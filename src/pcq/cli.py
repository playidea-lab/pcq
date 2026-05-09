"""pcq CLI — JSON-stable agent interface.

v4.0: contract runtime + agent CLI surface only. atoms/recipe-meta/dry-run 제거.

Commands:
  pcq inspect [PATH] [--json]
  pcq validate [PATH] [--plan PLAN_FILE] [--planset PLANSET_FILE]
                       [--strictness 0..4] [--json]
  pcq summarize-run OUTPUT_DIR [--json]
  pcq init-experiment [--output DIR] [--name NAME] [--force]
                       [--with-pyproject] [--agent codex|claude|both] [--json]
  pcq agent install [--target codex|claude|both] [--path DIR]
                     [--dry-run] [--force] [--json]
  pcq agent status [--target codex|claude|both] [--path DIR] [--json]
  pcq apply-plan PLAN_FILE [--path DIR] [--json]
  pcq apply-planset PLANSET_FILE [--path DIR] [--output-pattern PAT]
                                  [--force] [--json]
  pcq finalize [OUTPUT_DIR] [--project-root PATH] [--status STATUS] [--json]
  pcq validate-run [OUTPUT_DIR] [--json]
  pcq describe-run [OUTPUT_DIR] [--json]
  pcq compare-runs A B [--json]
  pcq lineage [OUTPUT_DIR] [--max-depth N] [--json]
  pcq resolve [PATH] [--cq-yaml PATH] [--json]
  pcq run [--path PATH] [--config-only] [--json] [--jsonl]
          [--events PATH]

Exit codes:
  0  success
  1  validation fail or operation error
  2  argparse error (auto)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


def _print_human(data: dict, title: str = "") -> None:
    if title:
        print(f"== {title} ==")
    print(json.dumps(data, indent=2, default=str))


def _read_text_tail(path: Path, *, max_chars: int = 20_000) -> tuple[str, bool]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", False
    if len(text) <= max_chars:
        return text, False
    return text[-max_chars:], True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_metric_line(line: str) -> dict[str, int | float] | None:
    """Parse pcq.log stdout lines: `@key=value @other=value`.

    Returns None when the line is not entirely made of metric tokens.
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith("@"):
        return None

    metrics: dict[str, int | float] = {}
    for token in stripped.split():
        if not token.startswith("@") or "=" not in token:
            return None
        key, raw_value = token[1:].split("=", 1)
        if not key:
            return None
        try:
            if any(ch in raw_value.lower() for ch in (".", "e")):
                value: int | float = float(raw_value)
            else:
                value = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        metrics[key] = value
    return metrics or None


def _emit_jsonl_line(sink: Any, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, separators=(",", ":"), default=str)
    if sink is not None:
        sink.write(line + "\n")
        sink.flush()
    else:
        print(line, flush=True)


def _resolve_events_path(
    raw_path: str | None,
    *,
    project_root: Path,
    pcq_dir: Path,
    default_when_enabled: bool,
) -> Path | None:
    if raw_path is not None:
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            p = project_root / p
        return p.resolve()
    if default_when_enabled:
        return (pcq_dir / "events.jsonl").resolve()
    return None


def _run_with_events(
    *,
    cmd: str,
    project_root: Path,
    env: dict[str, str],
    out_payload: dict[str, Any],
    stdout_path: Path,
    stderr_path: Path,
    events_path: Path | None,
    emit_jsonl_stdout: bool,
    mirror_child_streams: bool,
) -> tuple[int, dict[str, Any]]:
    """Execute child cmd while emitting structured run events.

    Captures child stdout/stderr to log files and emits NDJSON events to
    stdout (when emit_jsonl_stdout) and/or events_path. Honors
    mirror_child_streams for non-JSON modes.
    """
    import subprocess
    import threading

    seq = 0
    sink = None
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        sink = events_path.open("w", encoding="utf-8")

    # 동시 emit 방지 — stdout/stderr reader thread 가 같은 sink 에 쓴다.
    emit_lock = threading.Lock()

    def emit(event: dict[str, Any]) -> None:
        nonlocal seq
        with emit_lock:
            seq += 1
            payload = {
                "schema_version": 1,
                "seq": seq,
                "time": _utc_now_iso(),
                **event,
            }
            if emit_jsonl_stdout:
                _emit_jsonl_line(None, payload)
            if sink is not None:
                _emit_jsonl_line(sink, payload)

    emit(
        {
            "event": "run.started",
            "cmd": cmd,
            "project_root": str(project_root),
            "runtime_cfg_path": out_payload.get("runtime_cfg_path"),
        }
    )

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )
    except OSError as e:
        emit({"event": "run.error", "error": str(e)})
        if sink is not None:
            sink.close()
        out_payload.update(
            {
                "status": "error",
                "exit_code": 1,
                "error": str(e),
            }
        )
        return 1, out_payload

    files = {
        "stdout": stdout_path.open("w", encoding="utf-8"),
        "stderr": stderr_path.open("w", encoding="utf-8"),
    }

    def reader(name: str, stream: Any) -> None:
        out_file = files[name]
        for line in stream:
            out_file.write(line)
            out_file.flush()
            if mirror_child_streams:
                target = sys.stdout if name == "stdout" else sys.stderr
                target.write(line)
                target.flush()
            text = line.rstrip("\n")
            if name == "stdout":
                metrics = _parse_metric_line(text)
                if metrics is not None:
                    emit({"event": "metric", "metrics": metrics})
                    continue
            emit(
                {
                    "event": name,
                    "stream": name,
                    "line": text,
                }
            )

    threads = [
        threading.Thread(
            target=reader, args=("stdout", proc.stdout), daemon=True
        ),
        threading.Thread(
            target=reader, args=("stderr", proc.stderr), daemon=True
        ),
    ]
    for t in threads:
        t.start()
    rc = proc.wait()
    for t in threads:
        t.join()
    for f in files.values():
        f.close()

    stdout_tail, stdout_truncated = _read_text_tail(stdout_path)
    stderr_tail, stderr_truncated = _read_text_tail(stderr_path)
    out_payload.update(
        {
            "status": "completed" if rc == 0 else "failed",
            "exit_code": rc,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "stdout_tail_truncated": stdout_truncated,
            "stderr_tail_truncated": stderr_truncated,
        }
    )
    if events_path is not None:
        out_payload["events_path"] = str(events_path)

    end_event: dict[str, Any] = {
        "event": (
            "run.completed" if rc == 0 else "run.failed"
        ),
        "exit_code": rc,
        "status": out_payload["status"],
    }
    if events_path is not None:
        end_event["events_path"] = str(events_path)
    emit(end_event)

    if sink is not None:
        sink.close()
    return rc, out_payload


def cmd_inspect(args: argparse.Namespace) -> int:
    from pcq.agent import inspect_project

    insp = inspect_project(args.path)
    out = insp.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Project Inspection")
    return 1 if insp.errors else 0


def cmd_validate(args: argparse.Namespace) -> int:
    from pcq.agent import ExperimentPlan, ExperimentPlanSet, validate_project
    from pcq.agent.schema import ValidationCheck

    report = validate_project(args.path, strictness=args.strictness)

    # PlanSet 검증 (선택)
    planset_file = getattr(args, "planset", None)
    if planset_file:
        ps_path = Path(planset_file)
        if not ps_path.exists():
            report.add(
                ValidationCheck(
                    id="planset_validation",
                    status="fail",
                    severity="blocking",
                    detail=f"planset file not found: {ps_path}",
                )
            )
        else:
            try:
                ps_data = json.loads(ps_path.read_text(encoding="utf-8"))
                planset = ExperimentPlanSet.from_dict(ps_data)
            except json.JSONDecodeError as e:
                report.add(
                    ValidationCheck(
                        id="planset_validation",
                        status="fail",
                        severity="blocking",
                        detail=f"planset JSON parse failed: {e}",
                    )
                )
            else:
                from pcq.agent.validate import _validate_planset

                for c in _validate_planset(planset):
                    report.add(c)

    # Plan 검증 (선택)
    plan_file = getattr(args, "plan", None)
    if plan_file:
        plan_path = Path(plan_file)
        if not plan_path.exists():
            report.add(
                ValidationCheck(
                    id="plan_validation",
                    status="fail",
                    severity="blocking",
                    detail=f"plan file not found: {plan_path}",
                )
            )
        else:
            try:
                plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
                plan = ExperimentPlan.from_dict(plan_data)
            except json.JSONDecodeError as e:
                report.add(
                    ValidationCheck(
                        id="plan_validation",
                        status="fail",
                        severity="blocking",
                        detail=f"plan JSON parse failed: {e}",
                    )
                )
            else:
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

    out = report.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Validation")
    return 0 if report.status != "fail" else 1


def cmd_init_experiment(args: argparse.Namespace) -> int:
    from pcq.agent import init_experiment

    try:
        result = init_experiment(
            output_dir=args.output,
            name=args.name,
            force=args.force,
            with_pyproject=args.with_pyproject,
            agent=args.agent if args.agent != "none" else None,
        )
    except Exception as e:  # noqa: BLE001
        err = {"error": str(e)}
        if args.json:
            _print_json(err)
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1
    out = result.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Init Experiment")
    return 0


def cmd_apply_plan(args: argparse.Namespace) -> int:
    from pcq.agent import apply_plan

    plan_path = Path(args.plan_file)
    if not plan_path.exists():
        err = {"error": f"plan file not found: {plan_path}"}
        if args.json:
            _print_json(err)
        else:
            print(err["error"], file=sys.stderr)
        return 1
    try:
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        err = {"error": f"plan JSON parse failed: {e}"}
        if args.json:
            _print_json(err)
        else:
            print(err["error"], file=sys.stderr)
        return 1
    result = apply_plan(args.path, plan_data)
    out = result.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Apply Plan")
    return 0 if result.status != "rejected" else 1


def cmd_apply_planset(args: argparse.Namespace) -> int:
    """ExperimentPlanSet 적용 — N 개 멤버 plan 을 expand."""
    from pcq.agent import apply_planset

    ps_path = Path(args.planset_file)
    if not ps_path.exists():
        err = {"error": f"planset file not found: {ps_path}"}
        if args.json:
            _print_json(err)
        else:
            print(err["error"], file=sys.stderr)
        return 1
    try:
        ps_data = json.loads(ps_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        err = {"error": f"planset JSON parse failed: {e}"}
        if args.json:
            _print_json(err)
        else:
            print(err["error"], file=sys.stderr)
        return 1
    result = apply_planset(
        args.path,
        ps_data,
        output_pattern=args.output_pattern,
        force=args.force,
    )
    out = result.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Apply PlanSet")
    return 0 if result.status != "rejected" else 1


def cmd_summarize_run(args: argparse.Namespace) -> int:
    from pcq.agent import summarize_run

    summary = summarize_run(args.output_dir)
    out = summary.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Run Summary")
    return 0 if summary.status != "failed" else 1


def cmd_finalize(args: argparse.Namespace) -> int:
    """run_record.json + validation_report.json 작성.

    output_dir ancestor 를 walk-up 하며 cq.yaml 가진 디렉토리를 project_root 로 사용.
    """
    from pcq.contract import finalize_run

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        err = {"error": f"output_dir not found: {output_dir}"}
        if args.json:
            _print_json(err)
        else:
            print(err["error"], file=sys.stderr)
        return 1

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else _find_project_root_for_output_dir(output_dir)
    )

    rr_path = finalize_run(
        history=None,
        status=args.status,
        output_dir=output_dir,
        project_root=project_root,
    )

    out: dict[str, Any] = {
        "schema_version": 1,
        "run_record_path": str(rr_path),
        "validation_report_path": str(rr_path.parent / "validation_report.json"),
        "project_root": str(project_root) if project_root else None,
    }
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Finalize")
    return 0


def _find_project_root_for_output_dir(output_dir: Path) -> Path | None:
    """output_dir 의 ancestor walk-up 으로 cq.yaml 가진 디렉토리 탐색."""
    cur = output_dir.resolve()
    for d in [cur, *cur.parents][:8]:
        for name in ("cq.yaml", "pcq.yml"):
            if (d / name).exists():
                return d
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            return d
    return output_dir.parent


def cmd_validate_run(args: argparse.Namespace) -> int:
    from pcq.agent.validate_run import validate_run

    output_dir = Path(args.output_dir).resolve()
    report = validate_run(
        output_dir,
        strictness=args.strictness,
        rescan_manifest=args.rescan_manifest,
    )
    out = report.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Validate Run")
    return 0 if report.status != "fail" else 1


def cmd_describe_run(args: argparse.Namespace) -> int:
    """RunRecord 압축 요약 출력."""
    from pcq.agent.describe import describe_run

    desc = describe_run(args.output_dir)
    out = desc.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Run Description")
    return 0


def cmd_compare_runs(args: argparse.Namespace) -> int:
    """두 RunRecord 비교 diff 출력."""
    from pcq.agent.compare import compare_runs

    diff = compare_runs(args.a, args.b)
    out = diff.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Run Diff (a → b)")
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    """RunRecord parent chain 출력."""
    from pcq.agent.lineage import lineage

    chain = lineage(args.output_dir, max_depth=args.max_depth)
    out = chain.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Lineage")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    """Manage agent runtime assets in Codex/Claude discovery paths."""
    if args.agent_action == "install":
        from pcq.agent.install import install_agent_assets

        try:
            result = install_agent_assets(
                args.path,
                target=args.target,
                force=args.force,
                dry_run=args.dry_run,
            )
        except Exception as e:  # noqa: BLE001
            err = {"schema_version": 1, "error": str(e)}
            if args.json:
                _print_json(err)
            else:
                print(err["error"], file=sys.stderr)
            return 1

        out = result.to_dict()
        if args.json:
            _print_json(out)
        else:
            _print_human(out, "Agent Install")
        return 0
    if args.agent_action == "status":
        from pcq.agent.install import agent_assets_status

        try:
            result = agent_assets_status(args.path, target=args.target)
        except Exception as e:  # noqa: BLE001
            err = {"schema_version": 1, "error": str(e)}
            if args.json:
                _print_json(err)
            else:
                print(err["error"], file=sys.stderr)
            return 1

        out = result.to_dict()
        if args.json:
            _print_json(out)
        else:
            _print_human(out, "Agent Status")
        return 0
    return 1


def cmd_resolve(args: argparse.Namespace) -> int:
    """cq.yaml + CQ_CONFIG_JSON env → ResolvedConfig (debug view)."""
    from pcq.agent.resolver import resolve_project

    rc = resolve_project(path=args.path, cq_yaml_path=args.cq_yaml)
    out = rc.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Resolved Project")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """fresh-user first-class entry point.

    cq.yaml.cmd 읽어 실행. configs 를 .pcq/runtime_cfg.json 에 dump 후
    CQ_CONFIG_JSON env 자동 wiring. exit code 그대로 forward.

    --config-only 면 cmd 실행 안 함 — runtime_cfg.json 만 작성하고 path 출력.
    """
    import os
    import subprocess

    from pcq.agent.resolver import resolve_project

    if args.json and args.jsonl:
        err = {
            "schema_version": 1,
            "status": "error",
            "error": "--json and --jsonl are mutually exclusive",
            "project_root": str(Path(args.path).resolve()),
            "runtime_cfg_path": "",
            "cmd": "",
        }
        _print_json(err)
        return 2

    project_path = Path(args.path).resolve()
    rc = resolve_project(path=project_path)

    # cq.yaml 자체 부재 — reject.
    if rc.cq_yaml_path is None:
        err = {
            "schema_version": 1,
            "status": "error",
            "error": (
                f"cq.yaml not found at {project_path} (or any ancestor). "
                "Initialize with `pcq init-experiment` or pass --path."
            ),
            "project_root": str(project_path),
            "runtime_cfg_path": "",
            "cmd": "",
        }
        if args.jsonl:
            _emit_jsonl_line(
                None,
                {
                    "schema_version": 1,
                    "seq": 1,
                    "time": _utc_now_iso(),
                    "event": "run.error",
                    **err,
                },
            )
        elif args.json:
            _print_json(err)
        else:
            print(err["error"], file=sys.stderr)
        return 1

    project_root = rc.project_root or project_path
    cmd = rc.cmd or ""
    if not args.config_only and not cmd:
        err = {
            "schema_version": 1,
            "status": "error",
            "error": (
                f"cq.yaml at {rc.cq_yaml_path} has no `cmd` field — "
                "add e.g. `cmd: python train.py` or use --config-only."
            ),
            "project_root": str(project_root),
            "runtime_cfg_path": "",
            "cmd": "",
        }
        if args.jsonl:
            _emit_jsonl_line(
                None,
                {
                    "schema_version": 1,
                    "seq": 1,
                    "time": _utc_now_iso(),
                    "event": "run.error",
                    **err,
                },
            )
        elif args.json:
            _print_json(err)
        else:
            print(err["error"], file=sys.stderr)
        return 1

    # configs → .pcq/runtime_cfg.json (project_root 안).
    pcq_dir = project_root / ".pcq"
    pcq_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg_path = pcq_dir / "runtime_cfg.json"
    runtime_cfg_path.write_text(
        json.dumps(rc.cfg, indent=2, default=str), encoding="utf-8"
    )

    if args.config_only:
        out: dict[str, Any] = {
            "schema_version": 1,
            "status": "config_only",
            "runtime_cfg_path": str(runtime_cfg_path),
            "project_root": str(project_root),
            "cmd": cmd,
        }
        if args.jsonl:
            _emit_jsonl_line(
                None,
                {
                    "schema_version": 1,
                    "seq": 1,
                    "time": _utc_now_iso(),
                    "event": "run.config_only",
                    **out,
                },
            )
        elif args.json:
            _print_json(out)
        else:
            _print_human(out, "Run (config-only)")
        return 0

    # cmd 실행 — env 에 CQ_CONFIG_JSON 주입.
    env = dict(os.environ)
    env["CQ_CONFIG_JSON"] = str(runtime_cfg_path)

    out_payload: dict[str, Any] = {
        "schema_version": 1,
        "cmd": cmd,
        "runtime_cfg_path": str(runtime_cfg_path),
        "project_root": str(project_root),
    }

    events_path = _resolve_events_path(
        args.events,
        project_root=project_root,
        pcq_dir=pcq_dir,
        default_when_enabled=args.jsonl,
    )
    if args.jsonl or events_path is not None:
        stdout_path = pcq_dir / "run_stdout.log"
        stderr_path = pcq_dir / "run_stderr.log"
        returncode, out_payload = _run_with_events(
            cmd=cmd,
            project_root=project_root,
            env=env,
            out_payload=out_payload,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            events_path=events_path,
            emit_jsonl_stdout=args.jsonl,
            mirror_child_streams=not (args.json or args.jsonl),
        )
        if args.json:
            _print_json(out_payload)
        elif not args.jsonl:
            _print_human(out_payload, "Run")
        return returncode

    if args.json:
        stdout_path = pcq_dir / "run_stdout.log"
        stderr_path = pcq_dir / "run_stderr.log"
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            completed = subprocess.run(
                cmd,
                shell=True,
                cwd=str(project_root),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
            )

        stdout_tail, stdout_truncated = _read_text_tail(stdout_path)
        stderr_tail, stderr_truncated = _read_text_tail(stderr_path)
        out_payload.update(
            {
                "status": (
                    "completed" if completed.returncode == 0 else "failed"
                ),
                "exit_code": completed.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stdout_tail_truncated": stdout_truncated,
                "stderr_tail_truncated": stderr_truncated,
            }
        )
        _print_json(out_payload)
        return completed.returncode

    completed = subprocess.run(
        cmd,
        shell=True,
        cwd=str(project_root),
        env=env,
    )
    out_payload.update(
        {
            "status": "completed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
        }
    )
    _print_human(out_payload, "Run")
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pcq",
        description="pcq CLI — agent-operable JSON interface",
    )
    parser.add_argument("--version", action="version", version=_get_version())
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="inspect project structure")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--json", action="store_true", help="emit JSON only")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("validate", help="validate project before CQ run")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument(
        "--plan",
        default=None,
        help="optional ExperimentPlan JSON file to validate alongside project",
    )
    p.add_argument(
        "--planset",
        default=None,
        help=(
            "optional ExperimentPlanSet JSON file — set + member schema validation"
        ),
    )
    p.add_argument(
        "--strictness",
        type=int,
        choices=range(0, 5),
        default=None,
        help=(
            "validation strictness level 0..4 "
            "(default: cq.yaml configs.strictness or 2)"
        ),
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser(
        "summarize-run", help="summarize a completed output directory"
    )
    p.add_argument("output_dir")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_summarize_run)

    # init-experiment ────────────────────────────────────────────────
    p_init = sub.add_parser(
        "init-experiment", help="scaffold a CQ-runnable contract script"
    )
    p_init.add_argument("--output", default=".", help="output project directory")
    p_init.add_argument(
        "--name", default=None, help="cq.yaml name (default: pcq-experiment)"
    )
    p_init.add_argument(
        "--force", action="store_true", help="overwrite existing files"
    )
    p_init.add_argument(
        "--with-pyproject",
        action="store_true",
        help="also generate pyproject.toml (pcq dep)"
        " — recommended for reproducible lockfile_sha256 evidence",
    )
    p_init.add_argument(
        "--agent",
        choices=["none", "codex", "claude", "both"],
        default="none",
        help=(
            "also install agent runtime assets for Codex/Claude "
            "(default: none)"
        ),
    )
    p_init.add_argument("--json", action="store_true")
    p_init.set_defaults(func=cmd_init_experiment)

    # agent ─────────────────────────────────────────────────────────
    p_agent = sub.add_parser(
        "agent",
        help="manage pcq agent runtime assets",
    )
    agent_sub = p_agent.add_subparsers(dest="agent_action", required=True)
    p_agent_install = agent_sub.add_parser(
        "install",
        help="install Codex/Claude instructions and skills into a project",
    )
    p_agent_install.add_argument(
        "--target",
        choices=["codex", "claude", "both"],
        default="codex",
        help="agent runtime target (default: codex)",
    )
    p_agent_install.add_argument(
        "--path",
        default=".",
        help="project root to modify (default: cwd)",
    )
    p_agent_install.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned file operations without writing",
    )
    p_agent_install.add_argument(
        "--force",
        action="store_true",
        help="overwrite managed blocks and skill files when they differ",
    )
    p_agent_install.add_argument("--json", action="store_true")

    p_agent_status = agent_sub.add_parser(
        "status",
        help="inspect Codex/Claude instructions and skills without writing",
    )
    p_agent_status.add_argument(
        "--target",
        choices=["codex", "claude", "both"],
        default="codex",
        help="agent runtime target (default: codex)",
    )
    p_agent_status.add_argument(
        "--path",
        default=".",
        help="project root to inspect (default: cwd)",
    )
    p_agent_status.add_argument("--json", action="store_true")
    p_agent.set_defaults(func=cmd_agent)

    # apply-plan ─────────────────────────────────────────────────────
    p_apply = sub.add_parser(
        "apply-plan", help="apply ExperimentPlan to project (modifies cq.yaml only)"
    )
    p_apply.add_argument("plan_file", help="path to ExperimentPlan JSON")
    p_apply.add_argument(
        "--path", default=".", help="project root containing cq.yaml"
    )
    p_apply.add_argument("--json", action="store_true")
    p_apply.set_defaults(func=cmd_apply_plan)

    # apply-planset ──────────────────────────────────────────────────
    p_aps = sub.add_parser(
        "apply-planset",
        help=(
            "apply ExperimentPlanSet — expand member plans into N output dirs"
        ),
    )
    p_aps.add_argument("planset_file", help="path to ExperimentPlanSet JSON")
    p_aps.add_argument(
        "--path",
        default=".",
        help="project root containing cq.yaml (base for member dirs)",
    )
    p_aps.add_argument(
        "--output-pattern",
        default="runs/exp{i}",
        help=(
            "expand pattern using {i} (zero-based index) and/or {plan_id} "
            "(default: 'runs/exp{i}')"
        ),
    )
    p_aps.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing member output directories",
    )
    p_aps.add_argument("--json", action="store_true")
    p_aps.set_defaults(func=cmd_apply_planset)

    # finalize ───────────────────────────────────────────────────────
    p_fin = sub.add_parser(
        "finalize",
        help="generate run_record.json + validation_report.json",
    )
    p_fin.add_argument("output_dir", nargs="?", default="output")
    p_fin.add_argument("--project-root", default=None)
    p_fin.add_argument(
        "--status",
        choices=["completed", "failed", "partial"],
        default="completed",
    )
    p_fin.add_argument("--json", action="store_true")
    p_fin.set_defaults(func=cmd_finalize)

    # validate-run ───────────────────────────────────────────────────
    p_vr = sub.add_parser(
        "validate-run",
        help="post-run validation gates (manifest/metrics/summary)",
    )
    p_vr.add_argument("output_dir", nargs="?", default="output")
    p_vr.add_argument(
        "--strictness",
        type=int,
        choices=range(0, 5),
        default=2,
        help="post-run validation strictness level 0..4 (default: 2)",
    )
    p_vr.add_argument(
        "--rescan-manifest",
        action="store_true",
        help=(
            "ignore manifest entries whose files no longer exist in "
            "output_dir (for output_dir reuse / stale lock-in fix)"
        ),
    )
    p_vr.add_argument("--json", action="store_true")
    p_vr.set_defaults(func=cmd_validate_run)

    # describe-run ───────────────────────────────────────────────────
    p_dr = sub.add_parser(
        "describe-run",
        help="compact RunRecord summary",
    )
    p_dr.add_argument("output_dir", nargs="?", default="output")
    p_dr.add_argument("--json", action="store_true")
    p_dr.set_defaults(func=cmd_describe_run)

    # compare-runs ───────────────────────────────────────────────────
    p_cr = sub.add_parser(
        "compare-runs",
        help="diff two RunRecords",
    )
    p_cr.add_argument(
        "a", help="path to first run_record.json or output dir"
    )
    p_cr.add_argument(
        "b", help="path to second run_record.json or output dir"
    )
    p_cr.add_argument("--json", action="store_true")
    p_cr.set_defaults(func=cmd_compare_runs)

    # lineage ────────────────────────────────────────────────────────
    p_li = sub.add_parser(
        "lineage",
        help="walk RunRecord parent chain",
    )
    p_li.add_argument(
        "output_dir",
        nargs="?",
        default="output",
        help="output_dir or run_record.json path",
    )
    p_li.add_argument(
        "--max-depth",
        type=int,
        default=100,
        help="max chain depth (default 100)",
    )
    p_li.add_argument("--json", action="store_true")
    p_li.set_defaults(func=cmd_lineage)

    # resolve ────────────────────────────────────────────────────────────
    p_res = sub.add_parser(
        "resolve",
        help="resolve cq.yaml + env into single ResolvedConfig view",
    )
    p_res.add_argument("path", nargs="?", default=".")
    p_res.add_argument("--cq-yaml", default=None, help="explicit cq.yaml path")
    p_res.add_argument("--json", action="store_true")
    p_res.set_defaults(func=cmd_resolve)

    # run ────────────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help=(
            "execute cq.yaml.cmd with auto-wired CQ_CONFIG_JSON — "
            "fresh-user entry point"
        ),
    )
    p_run.add_argument(
        "--path",
        default=".",
        help="project root containing cq.yaml (default: current dir)",
    )
    p_run.add_argument(
        "--config-only",
        action="store_true",
        help="write runtime_cfg.json only; do not exec cmd",
    )
    p_run.add_argument(
        "--json",
        action="store_true",
        help=(
            "emit a pure JSON envelope; child stdout/stderr are captured to "
            ".pcq log files"
        ),
    )
    p_run.add_argument(
        "--jsonl",
        action="store_true",
        help=(
            "emit newline-delimited JSON run events; child stdout/stderr are "
            "captured to .pcq log files"
        ),
    )
    p_run.add_argument(
        "--events",
        default=None,
        help=(
            "write newline-delimited JSON run events to PATH; relative paths "
            "are resolved from the project root"
        ),
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


def _get_version() -> str:
    try:
        from pcq import __version__

        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
