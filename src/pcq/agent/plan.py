"""pcq.agent.plan — ExperimentPlan schema + change operations.

Plan은 agent가 LLM으로 생성, pcq apply-plan이 안전하게 적용한다.
v1.10 supported ops: set_config, set_atom.
v1.11 added: set_atom merge=true, set_dataset_transform (sugar for set_atom merge).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pcq.agent.schema import SCHEMA_VERSION


# v1.11 에서 지원하는 ChangeOp 종류
_SUPPORTED_OPS = {"set_config", "set_atom", "set_dataset_transform"}

# set_atom 의 atom 키 → registry kind 매핑 SSOT
_ATOM_KIND_MAP: dict[str, str] = {
    "model": "model",
    "dataset": "dataset",
    "dataset_train": "dataset",
    "dataset_eval": "dataset",
    "loss": "loss",
    "optim": "optim",
    "optim_factory": "optim",
    "sched": "sched",
    "sched_factory": "sched",
    "metric": "metric",
    "metrics": "metric",
}


@dataclass
class ChangeOp:
    """ExperimentPlan 의 단일 변경 단위.

    set_config:
      - key (str): cq.yaml configs 의 키. dot-notation 미지원 (flat key).
      - value (Any): JSON-safe scalar/list/dict.

    set_atom:
      - atom (str): recipe 키 ("model", "loss", "optim", "sched",
                    "dataset", "dataset_train", "dataset_eval", "metric")
      - name (str): registry 등록 이름 (예: "cross_entropy")
      - params (dict): atom factory 호출 파라미터
      - merge (bool, v1.11): True 면 기존 AtomRef 의 params 와 merge,
        False (default) 면 완전 교체.

    set_dataset_transform (v1.11, sugar):
      - split (str): "train" | "eval" → 내부적으로 atom="dataset_<split>"
      - params (dict): dataset 의 일부 params (예: {"image_size": 384})
      - 자동으로 set_atom merge=true 로 routing.
    """

    op: str
    # set_config 전용
    key: str | None = None
    value: Any = None
    # set_atom 전용
    atom: str | None = None
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    merge: bool = False                             # v1.11: set_atom params merge flag
    # set_dataset_transform 전용
    split: str | None = None                        # v1.11: "train" | "eval"

    def to_dict(self) -> dict:
        """JSON-safe 직렬화. op 종류에 따라 필드 선별."""
        d: dict[str, Any] = {"op": self.op}
        if self.op == "set_config":
            d["key"] = self.key
            d["value"] = self.value
        elif self.op == "set_atom":
            d["atom"] = self.atom
            d["name"] = self.name
            d["params"] = dict(self.params)
            # merge=False 는 기본값 — clean 직렬화를 위해 생략
            if self.merge:
                d["merge"] = True
        elif self.op == "set_dataset_transform":
            d["split"] = self.split
            d["params"] = dict(self.params)
        else:
            # 알 수 없는 op — 모든 필드를 보존 (추후 디버깅 용이)
            for k, v in asdict(self).items():
                if k != "op":
                    d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChangeOp":
        """JSON dict → ChangeOp. 알 수 없는 키는 무시."""
        return cls(
            op=d["op"],
            key=d.get("key"),
            value=d.get("value"),
            atom=d.get("atom"),
            name=d.get("name"),
            params=dict(d.get("params") or {}),
            merge=bool(d.get("merge", False)),
            split=d.get("split"),
        )

    def validate(self) -> list[str]:
        """ChangeOp 자체 검증. 빈 list 면 OK."""
        errors: list[str] = []
        if self.op not in _SUPPORTED_OPS:
            errors.append(
                f"unsupported op {self.op!r}; "
                f"supported: {sorted(_SUPPORTED_OPS)}"
            )
            return errors

        if self.op == "set_config":
            if not self.key or not isinstance(self.key, str):
                errors.append("set_config: key required (string)")
            # value 는 None 허용 (config 키 제거 의도일 수 있음). JSON-safe 만 보장.
            return errors

        if self.op == "set_atom":
            if not self.atom or not isinstance(self.atom, str):
                errors.append("set_atom: atom (recipe key) required")
            # name 은 merge=True + base 상속 사용 시 None 허용 (apply 단계에서 검증).
            if not self.merge and (not self.name or not isinstance(self.name, str)):
                errors.append("set_atom: name (registry atom name) required")
            if self.name is not None and not isinstance(self.name, str):
                errors.append("set_atom: name must be a string")
            if not isinstance(self.params, dict):
                errors.append("set_atom: params must be a dict")
            kind = self._infer_kind()
            valid_kinds = {"model", "dataset", "loss", "optim", "sched", "metric"}
            if kind not in valid_kinds:
                errors.append(
                    f"set_atom: cannot infer kind from atom key {self.atom!r} "
                    f"(supported: model, dataset, dataset_train, dataset_eval, "
                    f"loss, optim, sched, metric)"
                )
            return errors

        if self.op == "set_dataset_transform":
            if self.split not in ("train", "eval"):
                errors.append(
                    f"set_dataset_transform: split must be 'train' or 'eval', "
                    f"got {self.split!r}"
                )
            if not isinstance(self.params, dict) or not self.params:
                errors.append(
                    "set_dataset_transform: params required (e.g. {'image_size': 384})"
                )
            return errors

        return errors

    def _infer_kind(self) -> str:
        """atom 키 → registry kind 매핑.

        recipe 의 atom 키는 'dataset_train' 처럼 split 정보를 포함할 수 있어서
        kind 와 1:1 이 아니다. 이 매핑이 단일 SSOT.
        """
        if self.atom is None:
            return "unknown"
        return _ATOM_KIND_MAP.get(self.atom, "unknown")

    def to_set_atom(self) -> "ChangeOp":
        """set_dataset_transform → 등가 set_atom (merge=True) 변환.

        atom 키는 "dataset_<split>". name 은 None — apply 단계에서 기존
        override 또는 base recipe AtomRef 로부터 상속한다.
        """
        if self.op != "set_dataset_transform":
            raise ValueError(
                f"to_set_atom only valid for set_dataset_transform, got {self.op!r}"
            )
        atom_key = f"dataset_{self.split}"
        return ChangeOp(
            op="set_atom",
            atom=atom_key,
            name=None,                  # apply 단계에서 base 상속
            params=dict(self.params),
            merge=True,
        )


@dataclass
class ValidationPolicy:
    """Plan 적용 후 validation 정책. v1.10 에서는 schema 만 정의 (적용은 v1.11+)."""

    run_smoke: bool = True
    run_contract: bool = True
    allow_network: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentPlan:
    """Agent 가 LLM 으로 생성하는 실험 변경 계획.

    base.preset 을 시작점으로, changes 를 cq.yaml 에 적용한다.
    apply-plan 이 cq.yaml configs 만 수정 — train.py 와 recipes/ 는 사용자 영역.
    """

    schema_version: int = SCHEMA_VERSION
    id: str = ""                                          # e.g. "exp-001"
    intent: str = ""                                      # human-readable 설명
    base: dict[str, str] = field(default_factory=dict)    # {"preset": "vision/..."}
    target: dict[str, str] = field(default_factory=dict)  # {"metric": "...", "mode": "max"}
    changes: list[ChangeOp] = field(default_factory=list)
    validation_policy: ValidationPolicy = field(default_factory=ValidationPolicy)
    # v1.18 lineage: 이 plan 으로 만든 run 의 parent run 정보. apply_plan 이
    # cq.yaml.configs._parent_run_id / _parent_run_path 로 주입한다.
    parent_run_id: str | None = None
    parent_run_path: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "intent": self.intent,
            "base": dict(self.base),
            "target": dict(self.target),
            "changes": [c.to_dict() for c in self.changes],
            "validation_policy": self.validation_policy.to_dict(),
        }
        # 빈 값은 직렬화 생략 — clean shape.
        if self.parent_run_id:
            d["parent_run_id"] = self.parent_run_id
        if self.parent_run_path:
            d["parent_run_path"] = self.parent_run_path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentPlan":
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            id=d.get("id", ""),
            intent=d.get("intent", ""),
            base=dict(d.get("base") or {}),
            target=dict(d.get("target") or {}),
            changes=[ChangeOp.from_dict(c) for c in d.get("changes", [])],
            validation_policy=ValidationPolicy.from_dict(
                d.get("validation_policy") or {}
            ),
            parent_run_id=d.get("parent_run_id"),
            parent_run_path=d.get("parent_run_path"),
        )

    def validate(self) -> list[str]:
        """Plan 자체 검증. registry-aware 검증 (set_atom 의 ref) 은 apply 시 수행.

        여기서는 schema-level 만:
          - id 존재
          - changes 비어있지 않음
          - 각 change 의 op/key/atom/name 형식
          - base.preset 이 등록된 recipe 인지 (옵션)
        """
        errors: list[str] = []
        if not self.id or not isinstance(self.id, str):
            errors.append("plan.id required (non-empty string)")
        if not self.changes:
            errors.append("plan.changes empty")
        for i, c in enumerate(self.changes):
            for err in c.validate():
                errors.append(f"changes[{i}]: {err}")
        # base.preset 검증 (registered 여부)
        preset = self.base.get("preset")
        if preset:
            try:
                from pcq.trainer import Trainer

                if preset not in Trainer.list_presets():
                    errors.append(
                        f"base.preset {preset!r} not registered "
                        f"(use `pcq atoms list` to see presets)"
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(f"base.preset validation skipped: {e}")
        return errors


@dataclass
class ExperimentPlanSet:
    """Set of related ExperimentPlans (fork, grid, sweep — semantic up to agent).

    pcq expresses the set; the policy that generated it (random/grid/BO/agent
    LLM) is outside library scope.

    Fields:
      schema_version: pcq plan_set schema version (currently 1).
      id: set identifier (e.g. "sweep-001").
      intent: agent natural-language description of the set's purpose.
      base: shared base mapping (typically {"preset": "..."}); each member
            plan may override.
      parent_run_id / parent_run_path: lineage shared across the set —
            apply_planset propagates these to members that have not set
            their own.
      plans: list of ExperimentPlan (each member individually validatable).
    """

    schema_version: int = SCHEMA_VERSION
    id: str = ""
    intent: str | None = None
    base: dict[str, Any] = field(default_factory=dict)
    parent_run_id: str | None = None
    parent_run_path: str | None = None
    plans: list[ExperimentPlan] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "plans": [p.to_dict() for p in self.plans],
        }
        if self.intent:
            d["intent"] = self.intent
        if self.base:
            d["base"] = dict(self.base)
        if self.parent_run_id:
            d["parent_run_id"] = self.parent_run_id
        if self.parent_run_path:
            d["parent_run_path"] = self.parent_run_path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentPlanSet":
        plans_raw = d.get("plans") or []
        plans: list[ExperimentPlan] = []
        for entry in plans_raw:
            if isinstance(entry, ExperimentPlan):
                plans.append(entry)
            elif isinstance(entry, dict):
                plans.append(ExperimentPlan.from_dict(entry))
            # 다른 타입은 silent skip (validate 에서 빈 plans 로 fail).
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            id=d.get("id", ""),
            intent=d.get("intent"),
            base=dict(d.get("base") or {}),
            parent_run_id=d.get("parent_run_id"),
            parent_run_path=d.get("parent_run_path"),
            plans=plans,
        )

    def validate(self) -> list[str]:
        """PlanSet schema 검증.

        - id 존재
        - plans 비어있지 않음
        - 멤버 plan id 가 unique
        - 각 멤버 plan 의 schema validate 결과 누적
        """
        errors: list[str] = []
        if not self.id or not isinstance(self.id, str):
            errors.append("planset.id required (non-empty string)")
        if not self.plans:
            errors.append("planset.plans empty")

        seen_ids: dict[str, int] = {}
        for i, plan in enumerate(self.plans):
            if not isinstance(plan, ExperimentPlan):
                errors.append(f"plans[{i}]: not an ExperimentPlan")
                continue
            pid = plan.id or ""
            if pid:
                if pid in seen_ids:
                    errors.append(
                        f"plans[{i}]: duplicate plan id {pid!r} "
                        f"(also plans[{seen_ids[pid]}])"
                    )
                else:
                    seen_ids[pid] = i
            for err in plan.validate():
                errors.append(f"plans[{i}] ({pid or '<no-id>'}): {err}")
        return errors
