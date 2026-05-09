"""pcq.agent.run_record — RunRecord schema (v1.16+).

RunRecord = execution + source + environment + input identity + metric schema
          + artifact manifest + agent provenance + validation + summary.

Single source of truth for one experiment run. Used by CQ service for
comparison, lineage, and reproducibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1


@dataclass
class RunInfo:
    id: str = ""
    name: str = ""
    status: str = "completed"      # "completed" | "failed" | "partial" | "running" | "checkpointed"
    started_at: str | None = None
    finished_at: str | None = None
    # v1.18: lineage tracking — parent run의 semantic id 와 path string.
    # CQ URI (cq://...) 인 경우 pcq은 opaque string으로만 보존.
    parent_run_id: str | None = None
    parent_run_path: str | None = None
    # v2.11: streaming partial RunRecord 시간 차원 evidence.
    # last_updated_at: ISO-8601 UTC (tmp+rename atomic write 시점에 갱신).
    # partial: True 면 학습 중 — finalize_run() 에서 False 로 flip.
    last_updated_at: str | None = None
    partial: bool = False

    def to_dict(self) -> dict:
        # 빈 문자열 / None 은 출력에서 제거 (RunRecord 는 minimal-shape).
        # v2.11: partial / last_updated_at 는 의미 있을 때만 노출 — partial=False 는 default 라
        # 굳이 출력하지 않아도 backward-compat. partial=True 또는 last_updated_at 존재 시 노출.
        d = {k: v for k, v in asdict(self).items() if v not in (None, "")}
        # 기본값 partial=False 는 stale RunRecord 와 호환 — 명시적 True 일 때만 키 보존.
        if not self.partial:
            d.pop("partial", None)
        return d


@dataclass
class ExecutionInfo:
    cmd: str = ""
    cwd: str = "."
    config_path: str = "cq.yaml"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


@dataclass
class SourceInfo:
    git_sha: str = ""
    dirty: bool = False
    patch_sha256: str | None = None
    changed_files: list[str] = field(default_factory=list)
    cq_yaml_path: str | None = None
    cq_yaml_sha256: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # None / empty list 은 제거 — clean repo 에서 잡음 줄임.
        if not self.patch_sha256:
            d.pop("patch_sha256", None)
        if not self.changed_files:
            d.pop("changed_files", None)
        if not self.cq_yaml_path:
            d.pop("cq_yaml_path", None)
        if not self.cq_yaml_sha256:
            d.pop("cq_yaml_sha256", None)
        return d


@dataclass
class EnvironmentInfo:
    python: str = ""
    platform: str = ""
    pcq_version: str = ""
    torch_version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    device: str | None = None
    gpu_count: int | None = None
    gpu_model: str | None = None
    world_size: int | None = None
    lockfile: str | None = None
    lockfile_sha256: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in (
            "torch_version",
            "cuda_available",
            "cuda_version",
            "device",
            "gpu_count",
            "gpu_model",
            "world_size",
        ):
            if d.get(key) is None:
                d.pop(key, None)
        if not self.lockfile:
            d.pop("lockfile", None)
        if not self.lockfile_sha256:
            d.pop("lockfile_sha256", None)
        return d


@dataclass
class MetricsInfo:
    # declared: [{name, mode, split, aggregation, sample_count}, ...]
    declared: list[dict] = field(default_factory=list)
    history_path: str = "metrics.json"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentInfo:
    plan_id: str | None = None
    intent: str | None = None
    approval_status: str | None = None
    overrides: list[str] = field(default_factory=list)
    recipe: str | None = None

    def to_dict(self) -> dict:
        # 빈 값 제거 — agent 메타가 비어 있으면 굳이 노출하지 않음.
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [])}


@dataclass
class ValidationInfo:
    status: str = "unknown"     # "pass" | "warn" | "fail" | "unknown"
    report_path: str = "validation_report.json"

    def to_dict(self) -> dict:
        return asdict(self)


# v2.11: machine-readable error code enum. agent/CQ service 는 error_code 로
# 분류, suggested_fix 는 자연어로 읽고 추론. pcq 은 enum 만 발급.
ERROR_CODES: frozenset[str] = frozenset({
    "ERR_MISSING_DEPENDENCY",
    "ERR_INVALID_CONFIG",
    "ERR_DATASET_UNAVAILABLE",
    "ERR_OUT_OF_MEMORY",
    "ERR_TIMEOUT",
    "ERR_RUNTIME",        # catch-all
})


# 자연어 category → machine-readable error_code 매핑. failure_classifier 가 채우는
# 카테고리 (oom, missing_dependency, ...) 를 enum 으로 derive.
_CATEGORY_TO_ERROR_CODE: dict[str, str] = {
    "missing_dependency": "ERR_MISSING_DEPENDENCY",
    "config_error": "ERR_INVALID_CONFIG",
    "dataset_missing": "ERR_DATASET_UNAVAILABLE",
    "dataset_shape": "ERR_INVALID_CONFIG",
    "label_contract": "ERR_INVALID_CONFIG",
    "loss_contract": "ERR_INVALID_CONFIG",
    "metric_contract": "ERR_INVALID_CONFIG",
    "oom": "ERR_OUT_OF_MEMORY",
    "nan_loss": "ERR_RUNTIME",
    "timeout": "ERR_TIMEOUT",
    "distributed_write_race": "ERR_RUNTIME",
    "unknown_exception": "ERR_RUNTIME",
}


def category_to_error_code(category: str | None) -> str:
    """자연어 category → machine-readable error_code. 미상이면 ERR_RUNTIME."""
    if not category:
        return "ERR_RUNTIME"
    return _CATEGORY_TO_ERROR_CODE.get(str(category), "ERR_RUNTIME")


@dataclass
class FailureInfo:
    """Structured failure record for agent consumption (v2.11).

    error_code: machine-readable enum (ERROR_CODES). agent 가 분류용으로 사용.
    category:   기존 자연어 카테고리. backward compat — 옛 RunRecord 와 호환.
    message:    원본 exception 메시지 또는 자연어 설명.
    evidence:   structured key/value (예: {"module": "torchvision"}).
                pcq 은 자동 분류 시 채움. agent 가 추가해도 무방.
    suggested_fix: 자연어 — agent 가 다음 행동을 추론할 때 읽는다.
                   pcq 은 명령으로 변환하지 않는다 (정책 영역).
    """

    error_code: str = ""
    category: str = ""
    message: str = ""
    evidence: dict = field(default_factory=dict)
    suggested_fix: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # 빈 값 제거 — backward compat. suggested_fix/evidence 는 의미 있을 때만.
        if not self.suggested_fix:
            d.pop("suggested_fix", None)
        if not self.evidence:
            d.pop("evidence", None)
        if not self.error_code:
            d.pop("error_code", None)
        if not self.category:
            d.pop("category", None)
        if not self.message:
            d.pop("message", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FailureInfo":
        """Old shape (category-only) 도 정상 read. error_code 는 derive."""
        category = d.get("category", "") or ""
        error_code = d.get("error_code", "") or ""
        # backward compat: category 만 있고 error_code 없으면 derive.
        if category and not error_code:
            error_code = category_to_error_code(category)
        evidence = d.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        return cls(
            error_code=str(error_code) if error_code else "",
            category=str(category) if category else "",
            message=str(d.get("message", "") or ""),
            evidence=dict(evidence),
            suggested_fix=d.get("suggested_fix"),
        )


@dataclass
class RunRecord:
    """One run의 모든 evidence를 담는 SSOT.

    schema_version=1 에서는 다음 키를 의무적으로 포함한다:
      schema_version, run, execution, source, environment, metrics, artifacts.
    inputs / summary / agent / validation 은 선택적으로 채워질 수 있다.
    """

    schema_version: int = SCHEMA_VERSION
    run: RunInfo = field(default_factory=RunInfo)
    execution: ExecutionInfo = field(default_factory=ExecutionInfo)
    source: SourceInfo = field(default_factory=SourceInfo)
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    config: dict = field(default_factory=dict)         # cq.yaml/config identity
    inputs: dict = field(default_factory=dict)         # opaque (from pcq.yaml.inputs)
    input_summary: dict = field(default_factory=dict)  # identity coverage summary
    metrics: MetricsInfo = field(default_factory=MetricsInfo)
    artifacts: list[dict] = field(default_factory=list)   # from manifest.json files
    summary: dict = field(default_factory=dict)        # {target_metric, best, last}
    agent: AgentInfo = field(default_factory=AgentInfo)
    validation: ValidationInfo = field(default_factory=ValidationInfo)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "execution": self.execution.to_dict(),
            "source": self.source.to_dict(),
            "environment": self.environment.to_dict(),
            "config": self.config,
            "inputs": self.inputs,
            "input_summary": self.input_summary,
            "metrics": self.metrics.to_dict(),
            "artifacts": self.artifacts,
            "summary": self.summary,
            "agent": self.agent.to_dict(),
            "validation": self.validation.to_dict(),
        }
