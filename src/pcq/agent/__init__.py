"""pcq.agent — agent metadata + inspection + validation + summary.

v1.7 (Phase B): inspect / validate / summary 추가. 기존 v1.4 API
(recipe_meta, diff_recipes, list_meta) 는 그대로 유지.
"""
from __future__ import annotations

from typing import Any

from torch import nn

from pcq.agent.apply import (
    ApplyResult,
    PlanSetApplyResult,
    apply_plan,
    apply_planset,
)
from pcq.agent.compare import RunDiff, compare_runs
from pcq.agent.describe import RunDescription, describe_run
from pcq.agent.failure_classifier import classify_failure, enrich_failure
from pcq.agent.init import InitResult, init_experiment
from pcq.agent.install import (
    AgentAssetStatus,
    AgentInstallOperation,
    AgentInstallResult,
    AgentStatusResult,
    agent_assets_status,
    install_agent_assets,
)
from pcq.agent.json_contracts import (
    JSON_CONTRACT_VERSION,
    JSON_CONTRACTS,
    get_json_contracts,
    validate_json_contract,
)
from pcq.agent.inspect import inspect_project
from pcq.agent.lineage import (
    LineageChain,
    LineageNode,
    is_descendant_of,
    lineage,
)
from pcq.agent.plan import (
    ChangeOp,
    ExperimentPlan,
    ExperimentPlanSet,
    ValidationPolicy,
)
from pcq.agent.resolver import (
    ResolvedConfig,
    RunContext,
    resolve_project,
    resolve_run_context,
)
from pcq.agent.run_record import (
    ERROR_CODES,
    AgentInfo,
    EnvironmentInfo,
    ExecutionInfo,
    FailureInfo,
    MetricsInfo,
    RunInfo,
    RunRecord,
    SourceInfo,
    ValidationInfo,
    category_to_error_code,
)
from pcq.agent.scaffold import ScaffoldResult, scaffold_atom, validate_local_atoms
from pcq.agent.schema import (
    SCHEMA_VERSION,
    CqYamlSummary,
    EntrypointInfo,
    EpochSummary,
    OutputsInfo,
    ProjectInspection,
    RecipeInfo,
    RecipeSpec,
    RunSummary,
    ValidationCheck,
    ValidationReport,
)
from pcq.agent.smoke import SmokeReport, smoke_atom
from pcq.agent.strictness import (
    STRICTNESS_EVIDENCE_MATRIX,
    strictness_evidence_matrix,
    strictness_required_evidence,
)
from pcq.agent.summary import build_run_summary, summarize_run
from pcq.agent.validate import validate_project
from pcq.agent.validate_run import validate_run


# recipe dict 의 metadata-only 키 (atom 이 아님)
_META_KEYS = {
    "task",
    "metrics",
    "requires_extras",
    "smoke_safe",
    "smoke_overrides",
    "epochs",
    "batch_size",
}


def _atom_summary(value: Any) -> str:
    """atom (model/dataset/loss/lambda) 을 사람·agent 가 읽을 수 있게 요약.

    nn.Module → "ClassName (instance)"
    callable  → "callable<name>"
    그 외     → 타입 이름
    """
    if isinstance(value, nn.Module):
        return f"{type(value).__name__} (instance)"
    if callable(value):
        name = getattr(value, "__name__", "<lambda>")
        return f"callable<{name}>"
    return type(value).__name__


def recipe_meta(preset: str) -> dict:
    """Recipe metadata 추출. 학습 안 함, side effect 없음.

    v1.4 API 유지. recipe 호출 실패(예: torchvision 미설치) 시 import_error
    필드만 채워진 degraded dict 반환.
    """
    from pcq.trainer import _import_recipe

    fn = _import_recipe(preset)
    try:
        d = fn()
    except Exception as e:
        return {
            "name": preset,
            "task": None,
            "declared_metrics": [],
            "requires_extras": [],
            "smoke_safe": None,
            "has_smoke_overrides": False,
            "atoms": {},
            "epochs": None,
            "batch_size": None,
            "import_error": f"{type(e).__name__}: {e}",
        }

    atom_keys = (
        "model",
        "dataset_train",
        "dataset_eval",
        "dataset",
        "loss",
        "optim_factory",
        "sched_factory",
    )
    atoms: dict[str, str] = {}
    for key in atom_keys:
        if key in d:
            atoms[key] = _atom_summary(d[key])

    return {
        "name": preset,
        "task": d.get("task", "classification"),
        "declared_metrics": list(d.get("metrics", [])),
        "requires_extras": list(d.get("requires_extras", [])),
        "smoke_safe": d.get("smoke_safe"),
        "has_smoke_overrides": bool(d.get("smoke_overrides")),
        "atoms": atoms,
        "epochs": d.get("epochs"),
        "batch_size": d.get("batch_size"),
    }


def diff_recipes(a: str, b: str) -> dict:
    """두 recipe 의 atom + 설정 diff. agent 가 두 baseline 비교 시."""
    ma, mb = recipe_meta(a), recipe_meta(b)
    diff: dict[str, dict] = {}
    for key in set(ma.keys()) | set(mb.keys()):
        va, vb = ma.get(key), mb.get(key)
        if va != vb:
            diff[key] = {"a": va, "b": vb}
    return {"a": a, "b": b, "diff": diff}


def list_meta() -> list[dict]:
    """모든 등록된 recipe 의 metadata. agent 가 catalog 를 둘러볼 때."""
    from pcq.trainer import Trainer

    return [recipe_meta(p) for p in Trainer.list_presets()]


__all__ = [
    "SCHEMA_VERSION",
    "AgentInfo",
    "AgentAssetStatus",
    "AgentInstallOperation",
    "AgentInstallResult",
    "AgentStatusResult",
    "ApplyResult",
    "ChangeOp",
    "CqYamlSummary",
    "EntrypointInfo",
    "EnvironmentInfo",
    "EpochSummary",
    "ERROR_CODES",
    "ExecutionInfo",
    "ExperimentPlan",
    "ExperimentPlanSet",
    "FailureInfo",
    "InitResult",
    "JSON_CONTRACT_VERSION",
    "JSON_CONTRACTS",
    "LineageChain",
    "LineageNode",
    "MetricsInfo",
    "OutputsInfo",
    "PlanSetApplyResult",
    "ProjectInspection",
    "RecipeInfo",
    "RecipeSpec",
    "ResolvedConfig",
    "RunContext",
    "RunDescription",
    "RunDiff",
    "RunInfo",
    "RunRecord",
    "RunSummary",
    "ScaffoldResult",
    "SmokeReport",
    "SourceInfo",
    "STRICTNESS_EVIDENCE_MATRIX",
    "ValidationCheck",
    "ValidationInfo",
    "ValidationPolicy",
    "ValidationReport",
    "_atom_summary",
    "apply_plan",
    "apply_planset",
    "build_run_summary",
    "category_to_error_code",
    "classify_failure",
    "compare_runs",
    "describe_run",
    "diff_recipes",
    "enrich_failure",
    "get_json_contracts",
    "init_experiment",
    "agent_assets_status",
    "install_agent_assets",
    "inspect_project",
    "is_descendant_of",
    "lineage",
    "list_meta",
    "recipe_meta",
    "resolve_project",
    "resolve_run_context",
    "scaffold_atom",
    "smoke_atom",
    "strictness_evidence_matrix",
    "strictness_required_evidence",
    "summarize_run",
    "validate_local_atoms",
    "validate_json_contract",
    "validate_project",
    "validate_run",
]
