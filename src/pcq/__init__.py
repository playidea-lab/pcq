"""pcq — contract runtime + agent CLI surface.

This package wraps the cq.yaml runtime contract (CQ_CONFIG_JSON, stdout
@key=value, output_dir) with two surfaces:

  1. Contract script API (cq.config / cq.log / cq.save_all / ...) — used by
     train.py to honor the runtime contract while remaining framework-neutral.
  2. Agent CLI surface (pcq.agent.*) — JSON-envelope tooling for AI agents to
     plan, validate, and reason about experiment runs.

Note: cq.yaml / CQ_CONFIG_JSON / cq:// URI 스킴은 CQ Go service contract
specification 으로, pcq 라이브러리 이름과 무관하게 그대로 유지된다.
"""
from pcq import agent
from pcq.agent import (
    ResolvedConfig,
    RunContext,
    compare_runs,
    describe_run,
    inspect_project,
    resolve_project,
    resolve_run_context,
    summarize_run,
    validate_project,
)
from pcq.contract import (
    finalize_run,
    save_all,
    save_config_snapshot,
    save_manifest,
    save_metrics,
    save_partial_run_record,
    save_run_summary,
)
from pcq.core import config, input_dir, log, output_dir, seed_everything


__version__ = "4.1.0"
__all__ = [
    "ResolvedConfig",
    "RunContext",
    "agent",
    "compare_runs",
    "config",
    "describe_run",
    "finalize_run",
    "input_dir",
    "inspect_project",
    "log",
    "output_dir",
    "resolve_project",
    "resolve_run_context",
    "save_all",
    "save_config_snapshot",
    "save_manifest",
    "save_metrics",
    "save_partial_run_record",
    "save_run_summary",
    "seed_everything",
    "summarize_run",
    "validate_project",
]
