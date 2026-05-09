"""pcq.agent — agent metadata + inspection + validation + summary surface.

v4.0: contract runtime + agent CLI surface only. recipes/registry 의존 제거.
"""
from __future__ import annotations

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
from pcq.agent.inspect import inspect_project
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
from pcq.agent.schema import (
    SCHEMA_VERSION,
    CqYamlSummary,
    EntrypointInfo,
    EpochSummary,
    OutputsInfo,
    ProjectInspection,
    RecipeInfo,
    RunSummary,
    ValidationCheck,
    ValidationReport,
)
from pcq.agent.strictness import (
    STRICTNESS_EVIDENCE_MATRIX,
    strictness_evidence_matrix,
    strictness_required_evidence,
)
from pcq.agent.summary import build_run_summary, summarize_run
from pcq.agent.validate import validate_project
from pcq.agent.validate_run import validate_run


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
    "ResolvedConfig",
    "RunContext",
    "RunDescription",
    "RunDiff",
    "RunInfo",
    "RunRecord",
    "RunSummary",
    "SourceInfo",
    "STRICTNESS_EVIDENCE_MATRIX",
    "ValidationCheck",
    "ValidationInfo",
    "ValidationPolicy",
    "ValidationReport",
    "apply_plan",
    "apply_planset",
    "build_run_summary",
    "category_to_error_code",
    "classify_failure",
    "compare_runs",
    "describe_run",
    "enrich_failure",
    "get_json_contracts",
    "init_experiment",
    "agent_assets_status",
    "install_agent_assets",
    "inspect_project",
    "is_descendant_of",
    "lineage",
    "resolve_project",
    "resolve_run_context",
    "strictness_evidence_matrix",
    "strictness_required_evidence",
    "summarize_run",
    "validate_json_contract",
    "validate_project",
    "validate_run",
]
