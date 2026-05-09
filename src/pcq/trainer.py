"""pcq.trainer — high-level one-liner training API (T-CQPY-005).

Trainer 는 atom 들을 dict 로 모아 pcq.Experiment 를 자동 조립한다.
recipe = atom factory 들의 dict 를 반환하는 Python 함수 ("Tailwind preset" of cq).

사용 패턴:
  (A) preset 만:                Trainer(preset="vision/cifar10_smallcnn_baseline").fit()
  (B) preset + atom override:   Trainer(preset=..., sched_factory=...).fit()
  (C) atom 이름만:               Trainer(task="classification", dataset="fake", model="mlp").fit()
  (D) atom 객체 직접:            Trainer(model=pcq.examples.models.small_cnn(3,10),
                                          dataset_train=..., dataset_eval=...,
                                          optim_factory=...).fit()
"""
from __future__ import annotations

import importlib
import pkgutil
import pprint
from typing import Any, Callable, Optional

from torch import nn
from torch.utils.data import Dataset

from pcq import _registry
from pcq import metric as _metric
from pcq import optim as _optim
from pcq.experiment import Experiment

# task 이름 → loss registry 이름 매핑 (atom-only 모드에서 task 만으로 loss 결정)
_TASK_LOSSES: dict[str, str] = {"classification": "cross_entropy"}

# Trainer kwargs 중 cfg 로 전달되는 키 (나머지는 recipe dict override)
_CFG_KEYS = {"epochs", "batch_size", "seed", "output_dir", "resume_from", "resume", "lr"}

# 기본 학습률 (atom-only mode 의 adamw 기본값)
_DEFAULT_LR = 1e-3


def _import_recipe(name: str) -> Callable[[], dict]:
    """Recipe 함수 import. 'a/b/.../name' → pcq.recipes.a.b...name:name.

    중첩 그룹 지원: 'vision/seg/fake_seg_smoke' → pcq.recipes.vision.seg.fake_seg_smoke.
    함수 이름은 마지막 segment 와 동일해야 한다.
    """
    if "/" in name:
        parts = name.split("/")
        fname = parts[-1]
        module_path = "pcq.recipes." + ".".join(parts)
    else:
        fname = name
        module_path = f"pcq.recipes.{fname}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"recipe '{name}' not found (tried import {module_path})"
        ) from e
    if not hasattr(mod, fname):
        raise ValueError(
            f"recipe module {module_path} missing function {fname!r}"
        )
    return getattr(mod, fname)


def list_presets() -> list[str]:
    """등록된 recipe 이름 목록 (qualified, 예: 'vision/cifar10_smallcnn_baseline').

    임의 깊이의 nested group 지원. 'pcq.recipes.vision.seg.fake_seg_smoke' →
    'vision/seg/fake_seg_smoke'.
    """
    import pcq.recipes as recipes_pkg

    presets: list[str] = []
    for _finder, mod_name, ispkg in pkgutil.walk_packages(
        recipes_pkg.__path__, prefix="pcq.recipes."
    ):
        if ispkg:
            continue
        # pcq.recipes 의 prefix 'pcq.recipes.' 제거 → 'vision.seg.fake_seg_smoke'
        rel = mod_name[len("pcq.recipes.") :]
        if not rel:
            continue
        presets.append(rel.replace(".", "/"))
    return sorted(presets)


def print_recipe(name: str) -> None:
    """Recipe 내용 출력 (학습 자료성 — 어떤 atom 으로 구성되는지 투명하게)."""
    fn = _import_recipe(name)
    d = fn()
    print(f"# Recipe: {name}\n")
    pprint.pprint(
        {
            k: (
                type(v).__name__
                if isinstance(v, nn.Module)
                else (repr(v) if not callable(v) else f"<callable {k}>")
            )
            for k, v in d.items()
        }
    )


class _ComposedExperiment(Experiment):
    """Atom dict 으로 구동되는 Experiment subclass (Trainer 내부용).

    Generic step 은 (logits, label) classification 가정. 다른 task 는
    사용자 정의 Experiment 를 직접 작성한다.
    """

    def __init__(self, recipe_dict: dict, cfg: Optional[dict] = None) -> None:
        super().__init__(cfg=cfg)
        self._recipe = recipe_dict
        # recipe 가 epochs/batch_size 를 명시했고 cfg 에 없으면 cfg 로 승격
        if "epochs" in recipe_dict and "epochs" not in self.cfg:
            self.cfg["epochs"] = recipe_dict["epochs"]
        if "batch_size" in recipe_dict and "batch_size" not in self.cfg:
            self.cfg["batch_size"] = recipe_dict["batch_size"]

    def build_dataset(self, split: str) -> Dataset:
        # split 별 개별 dataset 우선, 없으면 공용 'dataset' 사용
        key = "dataset_train" if split == "train" else "dataset_eval"
        if key in self._recipe:
            ds = self._recipe[key]
            return ds(split) if callable(ds) else ds
        if "dataset" in self._recipe:
            ds = self._recipe["dataset"]
            return ds(split) if callable(ds) else ds
        raise KeyError(f"recipe missing dataset for split={split!r}")

    def build_model(self) -> nn.Module:
        m = self._recipe["model"]
        # 이미 nn.Module 이면 그대로, factory 면 호출
        if isinstance(m, nn.Module):
            return m
        return m() if callable(m) else m

    def build_loss(self) -> nn.Module:
        loss_obj = self._recipe["loss"]
        if isinstance(loss_obj, nn.Module):
            return loss_obj
        return loss_obj() if callable(loss_obj) else loss_obj

    def build_optimizer(self, params):
        factory = self._recipe.get("optim_factory")
        if factory is None:
            return _optim.adamw(params)
        return factory(params)

    def build_scheduler(self, optimizer):
        factory = self._recipe.get("sched_factory")
        if factory is None:
            return None
        return factory(optimizer)

    def training_step(self, batch) -> tuple:
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        task = self._recipe.get("task", "classification")
        if task == "segmentation":
            return loss, {"iou": _metric.iou(logits, y).item()}
        # default: classification
        return loss, {"acc": _metric.accuracy(logits, y).item()}

    def eval_step(self, batch) -> dict:
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        task = self._recipe.get("task", "classification")
        if task == "segmentation":
            return {
                "loss": loss.item(),
                "iou": _metric.iou(logits, y).item(),
            }
        return {
            "loss": loss.item(),
            "acc": _metric.accuracy(logits, y).item(),
        }


class Trainer:
    """High-level one-liner training API.

    내부적으로 _ComposedExperiment(=pcq.Experiment) 를 조립해 fit() 실행한다.
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        task: Optional[str] = None,
        dataset: Any = None,
        model: Any = None,
        cfg: Optional[dict] = None,
        **overrides: Any,
    ) -> None:
        self.cfg: dict = dict(cfg) if cfg else {}
        # Provenance — 어떤 preset 을 base 로, 어떤 atom 을 override 했는지
        self._preset_name: Optional[str] = preset
        # Recipe 시작점 — preset 이 있으면 그 dict 로, 없으면 빈 dict
        if preset is not None:
            recipe_fn = _import_recipe(preset)
            recipe_dict = dict(recipe_fn())
        else:
            recipe_dict = {}

        # 사용자 override key 추적 (model/dataset/task + overrides)
        overridden_keys: set[str] = set()

        # Atom 이름 해석 (Case C)
        if model is not None:
            recipe_dict["model"] = self._resolve_model(model)
            overridden_keys.add("model")
        if dataset is not None:
            recipe_dict["dataset"] = self._resolve_dataset(dataset)
            overridden_keys.add("dataset")
        if task is not None:
            if "loss" not in recipe_dict:
                if task not in _TASK_LOSSES:
                    raise ValueError(
                        f"unknown task {task!r}; supported: {list(_TASK_LOSSES)}"
                    )
                loss_name = _TASK_LOSSES[task]
                recipe_dict["loss"] = _registry.losses.build(loss_name)
            overridden_keys.add("task")

        # Override kwargs 분리: cfg 키는 self.cfg 로, 나머지는 recipe override
        for k, v in overrides.items():
            if k in _CFG_KEYS:
                self.cfg[k] = v
            else:
                recipe_dict[k] = v
                overridden_keys.add(k)

        # Atom-only 모드를 위한 합리적 기본값
        if "loss" not in recipe_dict:
            recipe_dict["loss"] = _registry.losses.build("cross_entropy")
        if "optim_factory" not in recipe_dict:
            lr = float(self.cfg.get("lr", _DEFAULT_LR))
            recipe_dict["optim_factory"] = lambda p, lr=lr: _optim.adamw(p, lr=lr)

        self._recipe_dict = recipe_dict
        self._user_overrides: set[str] = overridden_keys
        # cfg 에 provenance 메타 주입 (Experiment 가 config.json 으로 저장)
        if self._preset_name is not None:
            self.cfg["_recipe"] = self._preset_name
        if self._user_overrides:
            self.cfg["_overrides"] = sorted(self._user_overrides)

    @staticmethod
    def _resolve_model(model: Any) -> Any:
        # AtomRef → registry build, 문자열 → registry lookup + factory 호출,
        # 그 외 → 그대로 (nn.Module 객체 또는 callable factory)
        from pcq.registry.spec import AtomRef

        if isinstance(model, AtomRef):
            return _registry.models.build_ref(model)
        if isinstance(model, str):
            return _registry.models.build(model)
        return model

    @staticmethod
    def _resolve_dataset(dataset: Any) -> Any:
        from pcq.registry.spec import AtomRef

        if isinstance(dataset, AtomRef):
            # AtomRef 는 split-aware factory 가 아닌 단일 dataset 인스턴스 build.
            # 이 경우 dataset_train/dataset_eval 를 같은 인스턴스로 공유 — 사용자
            # 책임. split 별 분리가 필요하면 dataset_train= dataset_ref(...) 직접.
            return _registry.datasets.build_ref(dataset)
        if isinstance(dataset, str):
            return _registry.datasets.get(dataset).factory
        return dataset

    def fit(self) -> None:
        exp = _ComposedExperiment(self._recipe_dict, cfg=self.cfg)
        exp.fit()
        # Inspection 용 상태 노출
        self.history = exp.history
        self.output_dir = exp.output_dir

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Trainer":
        """cfg 에서 preset/overrides 자동 인식해 Trainer 생성.

        cfg["preset"]: recipe 이름. 선택.
        cfg["_overrides_data"]: dict of {atom_key: AtomRef.to_dict()}.
                                apply-plan 이 cq.yaml 에 기록한 override.
        그 외 cfg 키는 pcq.Experiment cfg 로 그대로 전달 (output_dir, epochs,
        batch_size, lr 등).

        AtomRef 는 RecipeSpec._resolve_ref 와 동일한 규약으로 resolve:
          - model      → out["model"] = nn.Module
          - dataset_*  → out[key]     = callable(split) → Dataset
          - loss       → out["loss"]  = nn.Module
          - optim      → out["optim_factory"] = callable(params)
          - sched      → out["sched_factory"] = callable(optimizer)
          - metric     → out["metrics_callables"] 에 누적
        """
        from pcq import registry as registry_pkg
        from pcq.agent.schema import RecipeSpec
        from pcq.registry.spec import AtomRef

        preset = cfg.get("preset")
        overrides_data = cfg.get("_overrides_data") or {}

        # AtomRef 를 trainer kwargs 로 변환. RecipeSpec._resolve_ref 활용 — 같은 규약.
        kwargs: dict[str, Any] = {}
        # _resolve_ref 가 채워줄 임시 dict — 결과를 kwargs 로 옮긴다
        for atom_key, ref_data in overrides_data.items():
            if not isinstance(ref_data, dict) or "kind" not in ref_data or "name" not in ref_data:
                # 알 수 없는 형태 — 그대로 전달 (forward compat)
                kwargs[atom_key] = ref_data
                continue
            ref = AtomRef.from_dict(ref_data)
            tmp: dict[str, Any] = {}
            RecipeSpec._resolve_ref(tmp, atom_key, ref, registry_pkg)
            # _resolve_ref 는 자체 키를 정함 (loss → "loss", optim → "optim_factory" 등)
            for k, v in tmp.items():
                kwargs[k] = v

        # cfg 에서 trainer 가 직접 사용하는 키 제외 — clean cfg 만 전달
        clean_cfg = {
            k: v for k, v in cfg.items()
            if k not in ("preset", "_overrides_data")
        }
        return cls(preset=preset, cfg=clean_cfg, **kwargs)

    def dry_run(self) -> dict:
        """학습 안 하고 조립된 plan 반환. agent 가 실험 전 검증 시.

        Returns:
            dict — cfg, atoms (요약), task, epochs, batch_size, declared_metrics,
                    expected_artifacts. output_dir 에 어떤 파일도 쓰지 않음.
        """
        # 지연 import 로 순환 의존 회피
        from pcq import agent as _agent

        meta_keys = {
            "epochs", "batch_size", "metrics", "requires_extras",
            "smoke_safe", "smoke_overrides", "task",
        }
        atoms = {
            k: _agent._atom_summary(v)
            for k, v in self._recipe_dict.items()
            if k not in meta_keys
        }
        return {
            "cfg": dict(self.cfg),
            "atoms": atoms,
            "task": self._recipe_dict.get("task", "classification"),
            "epochs": int(
                self.cfg.get("epochs", self._recipe_dict.get("epochs", 1))
            ),
            "batch_size": int(
                self.cfg.get(
                    "batch_size", self._recipe_dict.get("batch_size", 32)
                )
            ),
            "declared_metrics": list(self._recipe_dict.get("metrics", [])),
            "expected_artifacts": [
                "model.pt", "config.json", "metrics.json",
                "last.ckpt", "best.ckpt", "manifest.json",
            ],
            "preset": self._preset_name,
            "overrides": sorted(self._user_overrides),
        }

    @staticmethod
    def list_presets() -> list[str]:
        return list_presets()

    @staticmethod
    def list_models() -> list[str]:
        return _registry.models.list()

    @staticmethod
    def list_datasets() -> list[str]:
        return _registry.datasets.list()

    @staticmethod
    def list_metrics() -> list[str]:
        return _registry.metrics.list()

    @staticmethod
    def print_recipe(name: str) -> None:
        print_recipe(name)
