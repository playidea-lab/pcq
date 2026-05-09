"""pcq MCP tool definitions — wrap pcq Python API as MCP tools.

설계:
- 각 tool 은 ``PcqTool(name, descriptor, handler)`` 로 표현.
- handler 는 async dict-in / dict-out. subprocess 사용 안 함.
- ``run_experiment`` 만 사용자 cmd 를 실행해야 하므로 subprocess 사용.
- input schema 는 JSON_CONTRACTS 참고하되, MCP tool input 은 그보다 단순한
  CLI argument 형태이므로 JSON Schema 를 직접 작성.

각 handler 는 ``pcq.cli`` 의 ``cmd_*`` 함수와 동일한 contract 를 따른다.
한 입력은 dict, 출력은 dict (JSON-safe).
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    from mcp.types import Tool as MCPTool
except ImportError as e:  # pragma: no cover - covered by extras check
    raise ImportError(
        "pcq.mcp.tools requires the `mcp` extras. "
        "Install with: uv add 'pcq[mcp]'"
    ) from e


HandlerType = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class PcqTool:
    """A single MCP tool: name + JSON Schema descriptor + async handler."""

    name: str
    descriptor: MCPTool
    handler: HandlerType


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _coerce_str_path(value: Any, default: str = ".") -> str:
    if value is None:
        return default
    return str(value)


def _err(message: str, **extra: Any) -> dict[str, Any]:
    out = {"schema_version": 1, "status": "error", "error": message}
    out.update(extra)
    return out


def _load_plan_payload(args: dict[str, Any], key: str) -> tuple[Any, str | None]:
    """plan / planset payload 추출.

    inline ``plan`` (dict) 우선. 없으면 ``plan_file`` (path) 에서 JSON 로드.
    """
    inline = args.get(key)
    if inline is not None:
        return inline, None
    file_key = f"{key}_file"
    path_value = args.get(file_key)
    if path_value is None:
        return None, f"either `{key}` or `{file_key}` is required"
    p = Path(str(path_value))
    if not p.exists():
        return None, f"{file_key} not found: {p}"
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as e:
        return None, f"{file_key} JSON parse failed: {e}"


# ─────────────────────────────────────────────────────────────────────
# Tool factories
# ─────────────────────────────────────────────────────────────────────


def _resolve_project_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.resolver import resolve_project

        rc = resolve_project(
            path=_coerce_str_path(args.get("path"), "."),
            cq_yaml_path=args.get("cq_yaml_path"),
        )
        return rc.to_dict()

    return PcqTool(
        name="resolve_project",
        descriptor=MCPTool(
            name="resolve_project",
            description=(
                "Resolve cq.yaml + CQ_CONFIG_JSON env into a single "
                "ResolvedConfig view. Returns project_root, cq_yaml_path, name, "
                "cmd, cfg, declared_metrics, output_dir. Read-only — does not "
                "create directories or mutate state."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Project root path (cwd by default)",
                    },
                    "cq_yaml_path": {
                        "type": "string",
                        "description": (
                            "Optional explicit cq.yaml path "
                            "(otherwise discovered)"
                        ),
                    },
                },
            },
        ),
        handler=handler,
    )


def _inspect_project_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.inspect import inspect_project

        insp = inspect_project(_coerce_str_path(args.get("path"), "."))
        return insp.to_dict()

    return PcqTool(
        name="inspect_project",
        descriptor=MCPTool(
            name="inspect_project",
            description=(
                "Return project structure, entrypoint kind, contract state, "
                "and output evidence. Read-only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Project root to inspect",
                    },
                },
            },
        ),
        handler=handler,
    )


def _validate_project_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.plan import ExperimentPlan, ExperimentPlanSet
        from pcq.agent.schema import ValidationCheck
        from pcq.agent.validate import validate_project

        path = _coerce_str_path(args.get("path"), ".")
        strictness = args.get("strictness")
        report = validate_project(path, strictness=strictness)

        plan_data, plan_err = _load_plan_payload(args, "plan")
        if args.get("plan") is not None or args.get("plan_file") is not None:
            if plan_err:
                report.add(
                    ValidationCheck(
                        id="plan_validation",
                        status="fail",
                        severity="blocking",
                        detail=plan_err,
                    )
                )
            else:
                try:
                    plan = ExperimentPlan.from_dict(plan_data)
                    plan_errors = plan.validate()
                    if plan_errors:
                        for err in plan_errors:
                            report.add(
                                ValidationCheck(
                                    id="plan_validation",
                                    status="fail",
                                    severity="blocking",
                                    detail=err,
                                )
                            )
                    else:
                        report.add(
                            ValidationCheck(
                                id="plan_validation",
                                status="pass",
                                severity="info",
                                detail=f"plan {plan.id} valid",
                            )
                        )
                except Exception as e:  # noqa: BLE001
                    report.add(
                        ValidationCheck(
                            id="plan_validation",
                            status="fail",
                            severity="blocking",
                            detail=f"plan parse failed: {e}",
                        )
                    )

        ps_data, ps_err = _load_plan_payload(args, "planset")
        if (
            args.get("planset") is not None
            or args.get("planset_file") is not None
        ):
            if ps_err:
                report.add(
                    ValidationCheck(
                        id="planset_validation",
                        status="fail",
                        severity="blocking",
                        detail=ps_err,
                    )
                )
            else:
                try:
                    ps = ExperimentPlanSet.from_dict(ps_data)
                    from pcq.agent.validate import _validate_planset

                    for c in _validate_planset(ps):
                        report.add(c)
                except Exception as e:  # noqa: BLE001
                    report.add(
                        ValidationCheck(
                            id="planset_validation",
                            status="fail",
                            severity="blocking",
                            detail=f"planset parse failed: {e}",
                        )
                    )

        return report.to_dict()

    return PcqTool(
        name="validate_project",
        descriptor=MCPTool(
            name="validate_project",
            description=(
                "Run static and contract validation before execution. "
                "Optional inline ExperimentPlan / ExperimentPlanSet validation. "
                "Read-only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "strictness": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                        "description": (
                            "Validation strictness 0..4 (default: cq.yaml "
                            "configs.strictness or 2)"
                        ),
                    },
                    "plan": {
                        "type": "object",
                        "description": "Inline ExperimentPlan dict (optional)",
                    },
                    "plan_file": {
                        "type": "string",
                        "description": "Path to ExperimentPlan JSON file",
                    },
                    "planset": {
                        "type": "object",
                        "description": "Inline ExperimentPlanSet dict",
                    },
                    "planset_file": {
                        "type": "string",
                        "description": "Path to ExperimentPlanSet JSON file",
                    },
                },
            },
        ),
        handler=handler,
    )


def _validate_run_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.validate_run import validate_run

        output_dir = _coerce_str_path(args.get("output_dir"), "output")
        report = validate_run(
            Path(output_dir).resolve(),
            strictness=args.get("strictness", 2),
            rescan_manifest=bool(args.get("rescan_manifest", False)),
        )
        return report.to_dict()

    return PcqTool(
        name="validate_run",
        descriptor=MCPTool(
            name="validate_run",
            description=(
                "Run post-run validation gates (manifest / metrics / "
                "run_summary) on a completed output directory. Read-only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string", "default": "output"},
                    "strictness": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                        "default": 2,
                    },
                    "rescan_manifest": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Ignore manifest entries whose files no longer "
                            "exist (output_dir reuse / stale lock-in fix)"
                        ),
                    },
                },
            },
        ),
        handler=handler,
    )


def _describe_run_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.describe import describe_run

        output_dir = _coerce_str_path(args.get("output_dir"), "output")
        desc = describe_run(output_dir)
        return desc.to_dict()

    return PcqTool(
        name="describe_run",
        descriptor=MCPTool(
            name="describe_run",
            description=(
                "Return a compact, decision-facts oriented summary of a "
                "RunRecord. Includes best/last metrics, validation status, "
                "lineage, artifact counts, decision_facts. Read-only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string", "default": "output"},
                },
            },
        ),
        handler=handler,
    )


def _compare_runs_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.compare import compare_runs

        a = args.get("a")
        b = args.get("b")
        if a is None or b is None:
            return _err("compare_runs requires `a` and `b` paths")
        diff = compare_runs(str(a), str(b))
        return diff.to_dict()

    return PcqTool(
        name="compare_runs",
        descriptor=MCPTool(
            name="compare_runs",
            description=(
                "Diff two RunRecords (or output dirs). Returns metric deltas, "
                "config changes, lineage relation, decision_facts. Read-only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string",
                        "description": (
                            "Path to first run_record.json or output dir"
                        ),
                    },
                    "b": {
                        "type": "string",
                        "description": (
                            "Path to second run_record.json or output dir"
                        ),
                    },
                },
                "required": ["a", "b"],
            },
        ),
        handler=handler,
    )


def _lineage_chain_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.lineage import lineage

        output_dir = _coerce_str_path(args.get("output_dir"), "output")
        chain = lineage(output_dir, max_depth=int(args.get("max_depth", 100)))
        return chain.to_dict()

    return PcqTool(
        name="lineage_chain",
        descriptor=MCPTool(
            name="lineage_chain",
            description=(
                "Walk a RunRecord's parent chain. Returns ordered nodes from "
                "this run back to its earliest reachable ancestor. Read-only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string", "default": "output"},
                    "max_depth": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                    },
                },
            },
        ),
        handler=handler,
    )


def _apply_plan_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.apply import apply_plan
        from pcq.agent.plan import ExperimentPlan

        plan_data, err = _load_plan_payload(args, "plan")
        if err:
            return _err(err)

        # v4.2 (GM-3): schema 검증 친화적 envelope. raw TypeError /
        # KeyError 대신 {status: "rejected", reason: ..., detail/errors}
        # 를 반환해 agent 가 사유를 즉시 파악할 수 있게 함.
        try:
            plan = ExperimentPlan.from_dict(plan_data)
        except (TypeError, KeyError, ValueError, AttributeError) as e:
            return {
                "schema_version": 1,
                "status": "rejected",
                "reason": "schema_invalid",
                "detail": str(e),
                "raw_plan": plan_data,
                "expected_schema": (
                    "see tool descriptor for ExperimentPlan example"
                ),
            }

        validation_errors = plan.validate()
        if validation_errors:
            return {
                "schema_version": 1,
                "status": "rejected",
                "reason": "validation_failed",
                "errors": validation_errors,
                "plan_id": plan.id,
            }

        result = apply_plan(
            _coerce_str_path(args.get("path"), "."), plan
        )
        return result.to_dict()

    # v4.2 (GM-2): inputSchema.plan.description 에 minimal ExperimentPlan
    # example 을 inline 으로 박아 agent 가 grep 없이 한 번에 plan dict 를
    # 작성할 수 있게 한다 (research/mcp-dogfood GM-2).
    _plan_example_text = (
        "Inline ExperimentPlan dict. Minimal example:\n"
        "{\n"
        '  "schema_version": 1,\n'
        '  "id": "exp-001",\n'
        '  "intent": "try larger lr",\n'
        '  "base": {"baseline": "gen0"},\n'
        '  "parent_run_id": "run_...",\n'
        '  "parent_run_path": "/abs/path/output_gen0",\n'
        '  "changes": [\n'
        '    {"op": "set_config", "key": "lr", "value": 0.01}\n'
        "  ]\n"
        "}\n"
        "Required: id (non-empty string), changes (non-empty list of "
        "{op: 'set_config', key: <str>, value: <any>}). Optional: intent, "
        "base, target, parent_run_id, parent_run_path, validation_policy."
    )

    return PcqTool(
        name="apply_plan",
        descriptor=MCPTool(
            name="apply_plan",
            description=(
                "Apply ExperimentPlan to project (modifies cq.yaml.configs "
                "only — never train.py). Provenance recorded under "
                ".pcq/plans/<plan_id>.json. Returns rejected envelope with "
                "reason='schema_invalid'|'validation_failed' on bad input."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "plan": {
                        "type": "object",
                        "description": _plan_example_text,
                        "additionalProperties": True,
                    },
                    "plan_file": {
                        "type": "string",
                        "description": "Path to ExperimentPlan JSON file",
                    },
                },
            },
        ),
        handler=handler,
    )


def _apply_planset_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.apply import apply_planset
        from pcq.agent.plan import ExperimentPlanSet

        ps_data, err = _load_plan_payload(args, "planset")
        if err:
            return _err(err)

        # v4.2 (GM-3): apply_plan 과 동일한 친화적 envelope.
        try:
            planset = ExperimentPlanSet.from_dict(ps_data)
        except (TypeError, KeyError, ValueError, AttributeError) as e:
            return {
                "schema_version": 1,
                "status": "rejected",
                "reason": "schema_invalid",
                "detail": str(e),
                "raw_planset": ps_data,
                "expected_schema": (
                    "see tool descriptor for ExperimentPlanSet example"
                ),
            }
        validation_errors = planset.validate()
        if validation_errors:
            return {
                "schema_version": 1,
                "status": "rejected",
                "reason": "validation_failed",
                "errors": validation_errors,
                "set_id": planset.id,
            }

        result = apply_planset(
            _coerce_str_path(args.get("path"), "."),
            planset,
            output_pattern=args.get("output_pattern", "runs/exp{i}"),
            force=bool(args.get("force", False)),
        )
        return result.to_dict()

    # v4.2 (GM-2): inline ExperimentPlanSet example for self-sufficient
    # tool descriptor.
    _planset_example_text = (
        "Inline ExperimentPlanSet dict. Minimal example:\n"
        "{\n"
        '  "schema_version": 1,\n'
        '  "id": "sweep-001",\n'
        '  "intent": "lr sweep",\n'
        '  "parent_run_id": "run_baseline",\n'
        '  "plans": [\n'
        '    {"id": "exp-000", "changes": [\n'
        '      {"op": "set_config", "key": "lr", "value": 0.01}]},\n'
        '    {"id": "exp-001", "changes": [\n'
        '      {"op": "set_config", "key": "lr", "value": 0.001}]}\n'
        "  ]\n"
        "}\n"
        "Required: id, plans (non-empty, each member is an ExperimentPlan)."
    )

    return PcqTool(
        name="apply_planset",
        descriptor=MCPTool(
            name="apply_planset",
            description=(
                "Expand ExperimentPlanSet members into N output directories, "
                "each with its own cq.yaml + plan provenance. Returns rejected "
                "envelope with reason='schema_invalid'|'validation_failed' on "
                "bad input."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "planset": {
                        "type": "object",
                        "description": _planset_example_text,
                        "additionalProperties": True,
                    },
                    "planset_file": {
                        "type": "string",
                        "description": "Path to ExperimentPlanSet JSON file",
                    },
                    "output_pattern": {
                        "type": "string",
                        "default": "runs/exp{i}",
                        "description": (
                            "Pattern using {i} (zero-based index) and/or "
                            "{plan_id}"
                        ),
                    },
                    "force": {"type": "boolean", "default": False},
                },
            },
        ),
        handler=handler,
    )


def _init_experiment_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.init import init_experiment

        agent_target = args.get("agent")
        if agent_target == "none":
            agent_target = None
        result = init_experiment(
            output_dir=_coerce_str_path(args.get("output"), "."),
            name=args.get("name"),
            force=bool(args.get("force", False)),
            with_pyproject=bool(args.get("with_pyproject", False)),
            agent=agent_target,
        )
        return result.to_dict()

    return PcqTool(
        name="init_experiment",
        descriptor=MCPTool(
            name="init_experiment",
            description=(
                "Scaffold a CQ-runnable experiment (cq.yaml + train.py "
                "contract script, optionally pyproject.toml + agent runtime "
                "assets)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "output": {
                        "type": "string",
                        "default": ".",
                        "description": "Project directory to populate",
                    },
                    "name": {
                        "type": "string",
                        "description": "cq.yaml.name (default: pcq-experiment)",
                    },
                    "force": {"type": "boolean", "default": False},
                    "with_pyproject": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Generate pyproject.toml with pcq dep "
                            "(recommended for lockfile_sha256 evidence)"
                        ),
                    },
                    "agent": {
                        "type": "string",
                        "enum": ["none", "codex", "claude", "both"],
                        "default": "none",
                    },
                },
            },
        ),
        handler=handler,
    )


def _finalize_run_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.cli import _find_project_root_for_output_dir
        from pcq.contract import finalize_run

        output_dir_arg = args.get("output_dir")
        if output_dir_arg is None:
            return _err("finalize_run requires `output_dir`")
        output_dir = Path(str(output_dir_arg)).resolve()
        if not output_dir.exists():
            return _err(f"output_dir not found: {output_dir}")

        project_root = (
            Path(str(args["project_root"])).resolve()
            if args.get("project_root")
            else _find_project_root_for_output_dir(output_dir)
        )
        rr_path = finalize_run(
            history=None,
            status=str(args.get("status", "completed")),
            output_dir=output_dir,
            project_root=project_root,
        )
        return {
            "schema_version": 1,
            "run_record_path": str(rr_path),
            "validation_report_path": str(
                rr_path.parent / "validation_report.json"
            ),
            "project_root": str(project_root) if project_root else None,
        }

    return PcqTool(
        name="finalize_run",
        descriptor=MCPTool(
            name="finalize_run",
            description=(
                "Generate run_record.json + validation_report.json for an "
                "output directory. Walks ancestors to find project root if "
                "not provided. Writes to output_dir."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string"},
                    "project_root": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["completed", "failed", "partial"],
                        "default": "completed",
                    },
                },
                "required": ["output_dir"],
            },
        ),
        handler=handler,
    )


def _agent_install_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.install import install_agent_assets

        result = install_agent_assets(
            _coerce_str_path(args.get("path"), "."),
            target=str(args.get("target", "codex")),
            force=bool(args.get("force", False)),
            dry_run=bool(args.get("dry_run", False)),
            mcp=bool(args.get("mcp", False)),
        )
        return result.to_dict()

    return PcqTool(
        name="agent_install",
        descriptor=MCPTool(
            name="agent_install",
            description=(
                "Install pcq agent runtime assets (AGENTS.md / CLAUDE.md "
                "managed block, .agents|.claude/skills/pcq/SKILL.md). "
                "Optionally write .mcp.json to wire pcq MCP server."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "target": {
                        "type": "string",
                        "enum": ["codex", "claude", "both"],
                        "default": "codex",
                    },
                    "force": {"type": "boolean", "default": False},
                    "dry_run": {"type": "boolean", "default": False},
                    "mcp": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Also wire .mcp.json with `pcq mcp serve` entry"
                        ),
                    },
                },
            },
        ),
        handler=handler,
    )


def _agent_status_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        from pcq.agent.install import agent_assets_status

        result = agent_assets_status(
            _coerce_str_path(args.get("path"), "."),
            target=str(args.get("target", "codex")),
        )
        return result.to_dict()

    return PcqTool(
        name="agent_status",
        descriptor=MCPTool(
            name="agent_status",
            description=(
                "Inspect pcq agent runtime asset status (installed / missing "
                "/ stale / divergent / unmanaged) without writing. Read-only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "target": {
                        "type": "string",
                        "enum": ["codex", "claude", "both"],
                        "default": "codex",
                    },
                },
            },
        ),
        handler=handler,
    )


def _run_experiment_tool() -> PcqTool:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        """Execute cq.yaml.cmd with auto-wired CQ_CONFIG_JSON.

        Subprocess is unavoidable here — the user's cmd string must run.
        For long-running training, prefer the CQ service `cq_run_experiment`
        tool over this in-process variant.
        """
        from pcq.agent.resolver import resolve_project

        path = Path(_coerce_str_path(args.get("path"), ".")).resolve()
        config_only = bool(args.get("config_only", False))

        rc = resolve_project(path=path)
        if rc.cq_yaml_path is None:
            return _err(
                f"cq.yaml not found at {path} (or any ancestor). "
                "Initialize with `init_experiment` or pass path.",
                project_root=str(path),
                runtime_cfg_path="",
                cmd="",
            )
        project_root = rc.project_root or path
        cmd = rc.cmd or ""
        if not config_only and not cmd:
            return _err(
                f"cq.yaml at {rc.cq_yaml_path} has no `cmd` field — "
                "add e.g. `cmd: python train.py` or use config_only=true.",
                project_root=str(project_root),
                runtime_cfg_path="",
                cmd="",
            )

        pcq_dir = project_root / ".pcq"
        pcq_dir.mkdir(parents=True, exist_ok=True)
        runtime_cfg_path = pcq_dir / "runtime_cfg.json"
        runtime_cfg_path.write_text(
            json.dumps(rc.cfg, indent=2, default=str), encoding="utf-8"
        )

        out_payload: dict[str, Any] = {
            "schema_version": 1,
            "cmd": cmd,
            "runtime_cfg_path": str(runtime_cfg_path),
            "project_root": str(project_root),
        }
        if config_only:
            out_payload["status"] = "config_only"
            return out_payload

        env = dict(os.environ)
        env["CQ_CONFIG_JSON"] = str(runtime_cfg_path)

        stdout_path = pcq_dir / "run_stdout.log"
        stderr_path = pcq_dir / "run_stderr.log"

        # Run in worker thread so the event loop stays responsive.
        def _exec() -> int:
            with (
                stdout_path.open("w", encoding="utf-8") as so,
                stderr_path.open("w", encoding="utf-8") as se,
            ):
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(project_root),
                    env=env,
                    stdout=so,
                    stderr=se,
                )
            return completed.returncode

        rc_code = await asyncio.to_thread(_exec)

        from pcq.cli import _read_text_tail

        stdout_tail, stdout_truncated = _read_text_tail(stdout_path)
        stderr_tail, stderr_truncated = _read_text_tail(stderr_path)
        out_payload.update(
            {
                "status": "completed" if rc_code == 0 else "failed",
                "exit_code": rc_code,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stdout_tail_truncated": stdout_truncated,
                "stderr_tail_truncated": stderr_truncated,
            }
        )
        return out_payload

    return PcqTool(
        name="run_experiment",
        descriptor=MCPTool(
            name="run_experiment",
            description=(
                "Execute cq.yaml.cmd with auto-wired CQ_CONFIG_JSON env. "
                "Captures stdout/stderr to .pcq/run_*.log. For long-running "
                "GPU training, prefer the CQ service queue."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "config_only": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Write runtime_cfg.json only, do not exec cmd"
                        ),
                    },
                },
            },
        ),
        handler=handler,
    )


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────


def build_tools() -> list[PcqTool]:
    """Return the canonical 14 pcq MCP tools, in stable order."""
    return [
        _resolve_project_tool(),
        _inspect_project_tool(),
        _validate_project_tool(),
        _validate_run_tool(),
        _describe_run_tool(),
        _compare_runs_tool(),
        _lineage_chain_tool(),
        _apply_plan_tool(),
        _apply_planset_tool(),
        _init_experiment_tool(),
        _finalize_run_tool(),
        _agent_install_tool(),
        _agent_status_tool(),
        _run_experiment_tool(),
    ]
