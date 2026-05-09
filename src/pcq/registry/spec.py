"""pcq.registry.spec — atom metadata schema.

AtomSpec: registry entry (factory + metadata)
ParamSpec: 한 파라미터의 type/range/required
AtomRef: serializable reference for plans/recipes
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class ParamSpec:
    """단일 파라미터 schema. type/range/choices/required 검증 포함."""

    type: str = "any"             # "int" | "float" | "str" | "bool" | "path" | "any"
    default: Any = None
    required: bool = False
    choices: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    description: str = ""

    def to_dict(self) -> dict:
        # default 는 None 이어도 노출 (사용자에게 명시적으로 보여주는 정보)
        d = asdict(self)
        out: dict[str, Any] = {"type": d["type"], "default": d["default"]}
        if d["required"]:
            out["required"] = True
        if d["choices"] is not None:
            out["choices"] = d["choices"]
        if d["min"] is not None:
            out["min"] = d["min"]
        if d["max"] is not None:
            out["max"] = d["max"]
        if d["description"]:
            out["description"] = d["description"]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        # 알 수 없는 키는 무시 (additive schema)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """단일 값 검증. (ok, error_msg)."""
        if value is None:
            if self.required:
                return False, "required param missing"
            return True, None
        # 타입 체크 — bool 은 int 의 subclass 이므로 별도 처리
        type_map: dict[str, Any] = {
            "int": int,
            "float": (int, float),
            "str": str,
            "bool": bool,
            "path": (str,),
        }
        expected = type_map.get(self.type)
        if expected is not None:
            if self.type == "int" and isinstance(value, bool):
                return False, "type mismatch: expected int, got bool"
            if not isinstance(value, expected):
                return False, (
                    f"type mismatch: expected {self.type}, "
                    f"got {type(value).__name__}"
                )
        if self.choices is not None and value not in self.choices:
            return False, f"value {value!r} not in choices {self.choices}"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.min is not None and value < self.min:
                return False, f"value {value} < min {self.min}"
            if self.max is not None and value > self.max:
                return False, f"value {value} > max {self.max}"
        return True, None


@dataclass
class AtomSpec:
    """Registry entry — factory + metadata."""

    kind: str                                            # model | dataset | loss | optim | sched | metric
    name: str
    factory: Callable[..., Any]                          # NOT serialized (callable)
    params: dict[str, ParamSpec] = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)
    input_contract: dict[str, Any] = field(default_factory=dict)
    output_contract: dict[str, Any] = field(default_factory=dict)
    label_contract: dict[str, Any] = field(default_factory=dict)
    metric_contract: dict[str, Any] = field(default_factory=dict)
    requires_extras: list[str] = field(default_factory=list)
    smoke_safe: bool = True
    description: str = ""
    metadata_status: str = "explicit"                    # "explicit" | "inferred"
    # v1.12: provenance — atom 의 origin 추적
    source: str = "builtin"                              # "builtin" | "project" | "generated" | "external"
    module: str = ""                                     # factory.__module__ (자동 추출)
    # v2.4: role — atom 의 의도적 위치(목적). builtin atom 은 contract example 이지
    #              production catalog 가 아니라는 사실을 명시.
    #   "reference_example" — pcq builtin (계약 검증 + 온보딩 + smoke baseline)
    #   "user"              — project / external 사용자 atom (production)
    role: str = "reference_example"

    @classmethod
    def from_meta(
        cls, kind: str, name: str, factory: Callable, meta: dict | None
    ) -> "AtomSpec":
        # factory.__module__ 자동 추출 — project atom 식별/디버깅용
        inferred_module = getattr(factory, "__module__", "") or ""
        if not meta:
            # best-effort inferred — meta 없이 등록한 atom 호환
            # source="builtin" default → role="reference_example" inferred
            return cls(
                kind=kind,
                name=name,
                factory=factory,
                metadata_status="inferred",
                source="builtin",
                module=inferred_module,
                role="reference_example",
            )
        params: dict[str, ParamSpec] = {}
        for pname, pmeta in (meta.get("params") or {}).items():
            params[pname] = (
                pmeta if isinstance(pmeta, ParamSpec) else ParamSpec.from_dict(pmeta)
            )
        source = str(meta.get("source", "builtin"))
        # role 우선순위: meta 에 명시 > source 기반 inferred
        # builtin → reference_example, 그 외 (project/generated/external) → user
        role = meta.get("role")
        if role is None:
            role = "reference_example" if source == "builtin" else "user"
        return cls(
            kind=kind,
            name=name,
            factory=factory,
            params=params,
            tasks=list(meta.get("tasks", [])),
            input_contract=dict(meta.get("input_contract", {})),
            output_contract=dict(meta.get("output_contract", {})),
            label_contract=dict(meta.get("label_contract", {})),
            metric_contract=dict(meta.get("metric_contract", {})),
            requires_extras=list(meta.get("requires_extras", [])),
            smoke_safe=bool(meta.get("smoke_safe", True)),
            description=str(meta.get("description", "")),
            metadata_status="explicit",
            source=source,
            module=inferred_module,
            role=str(role),
        )

    def to_dict(self) -> dict:
        """JSON-safe metadata. factory 는 제외."""
        return {
            "kind": self.kind,
            "name": self.name,
            "params": {n: p.to_dict() for n, p in self.params.items()},
            "tasks": self.tasks,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "label_contract": self.label_contract,
            "metric_contract": self.metric_contract,
            "requires_extras": self.requires_extras,
            "smoke_safe": self.smoke_safe,
            "description": self.description,
            "metadata_status": self.metadata_status,
            "source": self.source,
            "module": self.module,
            # v2.4: role — builtin atoms 가 production catalog 가 아니라
            #              contract example 임을 명시.
            "role": self.role,
        }

    def validate_params(self, supplied: dict) -> list[str]:
        """파라미터 검증. 빈 list 면 OK."""
        errors: list[str] = []
        # required missing
        for pname, pspec in self.params.items():
            if pspec.required and pname not in supplied:
                errors.append(f"required param {pname!r} missing")
        # unknown supplied (params 가 정의된 경우만 strict)
        if self.params:
            for pname in supplied:
                if pname not in self.params:
                    errors.append(
                        f"unknown param {pname!r} (allowed: {sorted(self.params)})"
                    )
        # per-param validate
        for pname, value in supplied.items():
            if pname in self.params:
                ok, msg = self.params[pname].validate(value)
                if not ok:
                    errors.append(f"param {pname!r}: {msg}")
        return errors


@dataclass
class AtomRef:
    """Serializable reference to a registered atom. Used in recipes/plans."""

    kind: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> "AtomRef":
        return cls(
            kind=d["kind"],
            name=d["name"],
            params=dict(d.get("params", {})),
        )
