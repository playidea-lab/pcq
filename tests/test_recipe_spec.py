"""RecipeSpec.build() 호환성 + seg recipe 변환 검증."""
from __future__ import annotations

import json

import pcq
from pcq.agent.schema import RecipeSpec


def test_seg_smoke_recipe_uses_spec():
    """fake_seg_smoke recipe 가 SPEC 기반인지 확인."""
    from pcq.recipes.vision.seg import fake_seg_smoke as mod

    assert hasattr(mod, "SPEC")
    assert isinstance(mod.SPEC, RecipeSpec)
    assert mod.SPEC.task == "segmentation"
    assert mod.SPEC.name == "vision/seg/fake_seg_smoke"


def test_seg_recipe_spec_build_compatible(tmp_path):
    """SPEC.build() 로 만들어진 dict 가 _ComposedExperiment 호환."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 4}
    pcq.Trainer(preset="vision/seg/fake_seg_smoke", cfg=cfg).fit()
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_recipe_spec_to_dict_json_safe():
    from pcq.recipes.vision.seg.fake_seg_smoke import SPEC

    d = SPEC.to_dict()
    json.dumps(d)  # 직렬화 가능
    assert d["name"] == "vision/seg/fake_seg_smoke"
    assert "atoms" in d
    # AtomRef 는 dict 로 직렬화됨
    assert d["atoms"]["model"]["kind"] == "model"
    assert d["atoms"]["model"]["name"] == "unet"
    # defaults 는 epochs/batch_size 포함
    assert d["defaults"]["epochs"] == 2


def test_voc_unet_recipe_uses_spec():
    from pcq.recipes.vision.seg import voc_unet as mod

    assert hasattr(mod, "SPEC")
    assert isinstance(mod.SPEC, RecipeSpec)


def test_voc_unet_recipe_spec_metadata():
    from pcq.recipes.vision.seg.voc_unet import SPEC

    assert SPEC.smoke_safe is False
    assert SPEC.requires_extras == ["vision"]
    assert SPEC.smoke_overrides is not None
    assert "dataset_train" in SPEC.smoke_overrides


def test_voc_unet_acceptance_with_smoke_overrides(tmp_path):
    """smoke_overrides 적용한 voc_unet acceptance run."""
    from pcq.testing import recipe_smoke

    report = recipe_smoke("vision/seg/voc_unet", tmp_path)
    assert report.passed, str(report)


def test_recipe_spec_build_resolves_optim_to_factory():
    """optim AtomRef 는 model.parameters() 기반 factory 로 변환."""
    from pcq.recipes.vision.seg.fake_seg_smoke import SPEC

    built = SPEC.build()
    assert "optim_factory" in built
    assert callable(built["optim_factory"])


def test_recipe_spec_build_lazy_dataset():
    """dataset AtomRef 는 split-aware lambda 로 lazy 변환 (fit() 까지 build 지연)."""
    from pcq.recipes.vision.seg.fake_seg_smoke import SPEC

    built = SPEC.build()
    # dataset_train/eval 은 callable 로 wrapping (split 인자 받음)
    assert callable(built["dataset_train"])
    assert callable(built["dataset_eval"])
    # 호출하면 dataset 인스턴스 반환
    ds = built["dataset_train"]("train")
    assert hasattr(ds, "__getitem__")
    assert hasattr(ds, "__len__")


def test_recipe_spec_metrics_preserved():
    """SPEC.metrics 가 build() dict 에 포함."""
    from pcq.recipes.vision.seg.fake_seg_smoke import SPEC

    built = SPEC.build()
    assert built["metrics"] == [
        "epoch", "train_loss", "train_iou", "eval_loss", "eval_iou"
    ]


# ── v1.9: 모든 built-in recipe 가 SPEC 사용 ─────────────────────────────────
import importlib  # noqa: E402

import pytest  # noqa: E402

from pcq.trainer import _import_recipe  # noqa: E402


_ALL_RECIPES = [
    "vision/fake_smoke",
    "vision/mnist_mlp",
    "vision/cifar10_smallcnn_baseline",
    "vision/cifar10_resnet18",
    "vision/seg/fake_seg_smoke",
    "vision/seg/voc_unet",
    "nlp/fake_text_classifier",
]


@pytest.mark.parametrize("recipe", _ALL_RECIPES)
def test_recipe_uses_spec(recipe):
    """v1.9: 모든 등록 recipe 는 SPEC = RecipeSpec(...) 형태."""
    fn = _import_recipe(recipe)
    mod = importlib.import_module(fn.__module__)
    assert hasattr(mod, "SPEC"), f"{recipe} 가 SPEC 보유하지 않음"
    assert isinstance(mod.SPEC, RecipeSpec), (
        f"{recipe} SPEC 이 RecipeSpec 인스턴스 아님: {type(mod.SPEC)}"
    )


def test_recipe_spec_to_dict_all_json_safe():
    """v1.9: 모든 recipe SPEC.to_dict() 가 JSON 직렬화 가능."""
    for recipe in _ALL_RECIPES:
        fn = _import_recipe(recipe)
        mod = importlib.import_module(fn.__module__)
        spec = getattr(mod, "SPEC", None)
        assert spec is not None
        # 직렬화 시도 — 실패하면 raise
        json.dumps(spec.to_dict())


def test_fake_smoke_v19_actually_trains(tmp_path):
    """v1.9 fake_smoke (SPEC 기반) 도 학습 가능."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(preset="vision/fake_smoke", cfg=cfg).fit()
    assert (tmp_path / "model.pt").exists()


def test_nlp_fake_text_classifier_v19_actually_trains(tmp_path):
    """v1.9 nlp/fake_text_classifier (SPEC 기반) 도 학습 가능."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(preset="nlp/fake_text_classifier", cfg=cfg).fit()
    assert (tmp_path / "model.pt").exists()
