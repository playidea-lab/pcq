"""pcq agent schemas — stdlib dataclasses with to_dict() helpers.

모든 schema 는 schema_version=1 을 필드로 포함. 향후 변경은 additive 우선.
Pydantic 의존 없이 stdlib dataclass 만 사용한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Public schema version. v1.x 동안은 additive 변경만 허용.
SCHEMA_VERSION = 1


def _clean(d: dict) -> dict:
    """None 값을 제거해 JSON 출력을 깔끔하게. list/dict 는 그대로 둔다."""
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class CqYamlSummary:
    path: str
    name: str | None = None
    cmd: str | None = None
    declared_metrics: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    # v1.15: structured cq.yaml — dict-style metrics + inputs section.
    # metrics_schema: {metric_name: {mode, split, aggregation, sample_count, ...}}
    # inputs:         {input_name:  {name, version, uri, sha256, split, ...}}
    metrics_schema: dict[str, dict] = field(default_factory=dict)
    inputs: dict[str, dict] = field(default_factory=dict)
    parse_error: str | None = None

    def to_dict(self) -> dict:
        out = _clean(asdict(self))
        # 빈 dict 는 _clean 으로 인해 유지됨 — 명시적으로 의미가 있을 때만 노출.
        # metrics_schema / inputs 가 비어 있으면 출력에서 제거 (legacy list-style 호환).
        if not self.metrics_schema:
            out.pop("metrics_schema", None)
        if not self.inputs:
            out.pop("inputs", None)
        return out


@dataclass
class EntrypointInfo:
    path: str | None
    kind: str | None = None      # "trainer" | "experiment" | "script" | None
    preset: str | None = None    # AST 정적 스캔으로 Trainer(preset=...) 검출 시
    detected_imports: list[str] = field(default_factory=list)   # v1.13: ML framework imports
    cq_calls: list[str] = field(default_factory=list)           # v1.13: pcq.X() 호출 목록

    def to_dict(self) -> dict:
        out = _clean(asdict(self))
        # 빈 list 도 출력에 명시적으로 포함 (script-aware 검사가 의존)
        if self.detected_imports:
            out["detected_imports"] = self.detected_imports
        if self.cq_calls:
            out["cq_calls"] = self.cq_calls
        return out


@dataclass
class RecipeInfo:
    name: str
    task: str | None = None
    requires_extras: list[str] = field(default_factory=list)
    smoke_safe: bool | None = None

    def to_dict(self) -> dict:
        return _clean(asdict(self))


@dataclass
class OutputsInfo:
    output_dir: str | None = None
    has_manifest: bool = False
    has_metrics: bool = False
    has_summary: bool = False
    # v1.14: manifest schema 인식 (v1 path/kind only, v2 + sha256/size_bytes).
    manifest_schema_version: int | None = None
    manifest_files_count: int | None = None
    # v1.16: RunRecord + post-run validation report presence.
    has_run_record: bool = False
    has_validation_report: bool = False
    # v2.5 (P2 #5): output_dir 의 상태 — "empty" | "partial" | "complete" | None.
    # None 이면 output_dir 자체가 없음 (legacy: output_dir == None).
    # "empty": output_dir 존재하지만 manifest/metrics/run_record 모두 없음.
    # "partial": 일부 artifact 있음. "complete": run_record + manifest + metrics 모두.
    status: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProjectInspection:
    schema_version: int = SCHEMA_VERSION
    project_root: str = ""
    project_type: str = "unknown"   # "pcq" | "cq" | "unknown"
    has_cq_yaml: bool = False
    cq_yaml: CqYamlSummary | None = None
    entrypoint: EntrypointInfo | None = None
    recipes: list[RecipeInfo] = field(default_factory=list)
    outputs: OutputsInfo | None = None
    project_atoms_loaded: dict | None = None   # v1.12: load_project_atoms 보고서
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "project_root": self.project_root,
            "project_type": self.project_type,
            "has_cq_yaml": self.has_cq_yaml,
            "recipes": [r.to_dict() for r in self.recipes],
            "warnings": self.warnings,
            "errors": self.errors,
        }
        if self.cq_yaml is not None:
            out["cq_yaml"] = self.cq_yaml.to_dict()
        if self.entrypoint is not None:
            out["entrypoint"] = self.entrypoint.to_dict()
        if self.outputs is not None:
            out["outputs"] = self.outputs.to_dict()
        if self.project_atoms_loaded is not None:
            out["project_atoms_loaded"] = self.project_atoms_loaded
        return out


@dataclass
class EpochSummary:
    epoch: int
    metrics: dict[str, float]
    checkpoint: str | None = None

    def to_dict(self) -> dict:
        return _clean(asdict(self))


@dataclass
class RunSummary:
    schema_version: int = SCHEMA_VERSION
    status: str = "unknown"       # "completed" | "failed" | "partial" | "unknown"
    recipe: str | None = None
    monitor: dict | None = None   # {"name": str, "mode": "min"|"max"}
    target: dict | None = None    # {"metric": str, "mode": str} — 보통 monitor 와 동일
    best: EpochSummary | None = None
    last: EpochSummary | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    early_stopped_at: int | None = None
    warnings: list[str] = field(default_factory=list)
    failure: dict | None = None    # {"category": str, "message": str, "suggested_fix": str}

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "recipe": self.recipe,
            "monitor": self.monitor,
            "target": self.target,
            "best": self.best.to_dict() if self.best else None,
            "last": self.last.to_dict() if self.last else None,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
            "early_stopped_at": self.early_stopped_at,
            "warnings": self.warnings,
            "failure": self.failure,
        }
        return _clean(out)


@dataclass
class ValidationCheck:
    id: str
    status: str             # "pass" | "warn" | "fail" | "skip"
    severity: str = "info"  # "blocking" | "warning" | "info"
    detail: str = ""
    evidence: dict = field(default_factory=dict)
    suggested_fix: str | None = None

    def to_dict(self) -> dict:
        return _clean(asdict(self))


@dataclass
class ValidationReport:
    schema_version: int = SCHEMA_VERSION
    status: str = "pass"      # "pass" | "warn" | "fail"
    strictness: int | None = None
    strictness_name: str | None = None
    checks: list[ValidationCheck] = field(default_factory=list)
    blocking_count: int = 0
    warning_count: int = 0

    def to_dict(self) -> dict:
        out = {
            "schema_version": self.schema_version,
            "status": self.status,
            "strictness": self.strictness,
            "strictness_name": self.strictness_name,
            "checks": [c.to_dict() for c in self.checks],
            "blocking_count": self.blocking_count,
            "warning_count": self.warning_count,
        }
        return _clean(out)

    def add(self, check: ValidationCheck) -> None:
        self.checks.append(check)
        # blocking failure 면 fail 로 격상, warning 은 warn 으로 격상 (pass 일 때만)
        if check.status == "fail" and check.severity == "blocking":
            self.blocking_count += 1
            self.status = "fail"
        elif check.status == "warn" or check.severity == "warning":
            self.warning_count += 1
            if self.status == "pass":
                self.status = "warn"


@dataclass
class RecipeSpec:
    """Metadata-first recipe contract (v1.8+).

    atoms 값은 AtomRef (agent 친화) 또는 직접 객체/callable (사용자 편의) 모두 가능.
    .build() 가 AtomRef 들을 _ComposedExperiment 호환 dict 로 resolve.
    """

    name: str
    task: str
    description: str = ""
    metrics: list[str] = field(default_factory=list)
    monitor_candidates: list[dict] = field(default_factory=list)
    requires_extras: list[str] = field(default_factory=list)
    smoke_safe: bool | None = None
    smoke_overrides: dict[str, Any] | None = None
    atoms: dict[str, Any] = field(default_factory=dict)        # AtomRef | object | callable
    defaults: dict[str, Any] = field(default_factory=dict)     # epochs, batch_size, ...

    def to_dict(self) -> dict:
        """JSON-safe metadata representation. AtomRef 만 직렬화, 객체/callable 은 repr."""
        atoms_json: dict[str, Any] = {}
        for k, v in self.atoms.items():
            # AtomRef 는 to_dict 메서드 + kind 속성 보유 → 직렬화
            if hasattr(v, "to_dict") and hasattr(v, "kind") and hasattr(v, "name"):
                atoms_json[k] = v.to_dict()
            else:
                atoms_json[k] = repr(v)
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "task": self.task,
            "description": self.description,
            "metrics": list(self.metrics),
            "monitor_candidates": list(self.monitor_candidates),
            "requires_extras": list(self.requires_extras),
            "smoke_safe": self.smoke_safe,
            "smoke_overrides_keys": (
                sorted(self.smoke_overrides) if self.smoke_overrides else []
            ),
            "atoms": atoms_json,
            "defaults": dict(self.defaults),
        }

    def build(self) -> dict:
        """Resolve refs to executable atoms; produce dict for _ComposedExperiment."""
        # 지연 import — 순환 회피 + heavy import 지연
        from pcq import registry as _registry_pkg
        from pcq.registry.spec import AtomRef

        out: dict[str, Any] = {"task": self.task}
        if self.metrics:
            out["metrics"] = list(self.metrics)
        if self.requires_extras:
            out["requires_extras"] = list(self.requires_extras)
        if self.smoke_safe is not None:
            out["smoke_safe"] = self.smoke_safe
        if self.smoke_overrides:
            out["smoke_overrides"] = dict(self.smoke_overrides)
        # defaults (epochs/batch_size/...) 은 recipe dict 직속으로 승격
        out.update(dict(self.defaults))

        for key, atom in self.atoms.items():
            if isinstance(atom, AtomRef):
                self._resolve_ref(out, key, atom, _registry_pkg)
            else:
                out[key] = atom
        return out

    @staticmethod
    def _resolve_ref(
        out: dict, key: str, atom: Any, registry: Any
    ) -> None:
        """AtomRef 한 개를 out dict 에 적절한 형태로 배치."""
        kind = atom.kind
        if kind == "model":
            out["model"] = registry.models.build_ref(atom)
        elif kind == "loss":
            out["loss"] = registry.losses.build_ref(atom)
        elif kind == "dataset":
            # Lazy build — fit() 시점까지 dataset 생성 지연 (heavy import/download 회피).
            # _ComposedExperiment.build_dataset 이 callable(split) 호출 → AtomRef build.
            ref = atom
            out[key] = (
                lambda _split, _ref=ref, _r=registry:
                _r.datasets.build_ref(_ref)
            )
        elif kind == "optim":
            # optim 은 model.parameters() 가 fit() 시점에 결정되므로 factory 로 wrap.
            params_kw = dict(atom.params)
            opt_name = atom.name
            out["optim_factory"] = (
                lambda model_params, _n=opt_name, _p=params_kw:
                registry.optims.build(_n, model_params, **_p)
            )
        elif kind == "sched":
            params_kw = dict(atom.params)
            sched_name = atom.name
            out["sched_factory"] = (
                lambda optimizer, _n=sched_name, _p=params_kw:
                registry.scheds.build(_n, optimizer, **_p)
            )
        elif kind == "metric":
            # 다중 metric — list 누적
            if "metrics_callables" not in out:
                out["metrics_callables"] = []
            out["metrics_callables"].append(registry.metrics.build_ref(atom))
        else:
            # 알 수 없는 kind — 보수적으로 그대로 저장 (forward compat)
            out[key] = atom
