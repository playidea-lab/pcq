"""pcq CLI — JSON-stable agent interface.

Commands:
  pcq inspect [PATH] [--load-project-atoms] [--json]
  pcq recipe-meta NAME [--json]
  pcq dry-run [PATH] [--json]
  pcq validate [PATH] [--plan PLAN_FILE] [--planset PLANSET_FILE]
                       [--strictness 0..4] [--json]
  pcq summarize-run OUTPUT_DIR [--json]
  pcq atoms list [--kind KIND] [--source SRC] [--load-project PATH] [--json]
  pcq atoms show KIND NAME [--json]
  pcq atoms validate-ref REF_FILE [--json]
  pcq atoms scaffold KIND NAME [--output FILE] [--path DIR] [--force] [--json]
  pcq atoms validate-local [PATH] [--json]
  pcq atoms smoke KIND NAME [--load-project PATH] [--json]
  pcq init-experiment [--style trainer|experiment|script] [--preset NAME]
                       [--output DIR] [--name NAME] [--force]
                       [--agent codex|claude|both] [--json]
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
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        return path
    if default_when_enabled:
        return pcq_dir / "events.jsonl"
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
    """Run a command while producing structured JSONL run events."""
    import queue
    import subprocess
    import threading

    seq = 0
    events_sink = None
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_sink = events_path.open("w", encoding="utf-8")

    def emit(event: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        payload = {
            "schema_version": 1,
            "seq": seq,
            "time": _utc_now_iso(),
            **event,
        }
        if events_sink is not None:
            _emit_jsonl_line(events_sink, payload)
        if emit_jsonl_stdout:
            _emit_jsonl_line(None, payload)

    try:
        emit(
            {
                "event": "run.started",
                "status": "running",
                "cmd": cmd,
                "project_root": str(project_root),
                "runtime_cfg_path": out_payload["runtime_cfg_path"],
            }
        )

        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        stream_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

        def reader(name: str, stream: Any) -> None:
            try:
                for line in stream:
                    stream_queue.put((name, line))
            finally:
                stream_queue.put((name, None))

        threads = [
            threading.Thread(
                target=reader, args=("stdout", process.stdout), daemon=True
            ),
            threading.Thread(
                target=reader, args=("stderr", process.stderr), daemon=True
            ),
        ]
        for thread in threads:
            thread.start()

        done_streams = 0
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            while done_streams < 2:
                stream_name, line = stream_queue.get()
                if line is None:
                    done_streams += 1
                    continue

                if stream_name == "stdout":
                    stdout_file.write(line)
                    stdout_file.flush()
                    if mirror_child_streams:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    metrics = _parse_metric_line(line)
                    if metrics is not None:
                        emit(
                            {
                                "event": "metric",
                                "stream": "stdout",
                                "metrics": metrics,
                                "raw": line.rstrip("\n"),
                            }
                        )
                    else:
                        emit(
                            {
                                "event": "stdout",
                                "stream": "stdout",
                                "text": line.rstrip("\n"),
                            }
                        )
                else:
                    stderr_file.write(line)
                    stderr_file.flush()
                    if mirror_child_streams:
                        sys.stderr.write(line)
                        sys.stderr.flush()
                    emit(
                        {
                            "event": "stderr",
                            "stream": "stderr",
                            "text": line.rstrip("\n"),
                        }
                    )

        returncode = process.wait()
        for thread in threads:
            thread.join(timeout=1)

        final_status = "completed" if returncode == 0 else "failed"
        final_event = "run.completed" if returncode == 0 else "run.failed"
        emit(
            {
                "event": final_event,
                "status": final_status,
                "exit_code": returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "events_path": str(events_path) if events_path else "",
            }
        )

        stdout_tail, stdout_truncated = _read_text_tail(stdout_path)
        stderr_tail, stderr_truncated = _read_text_tail(stderr_path)
        out_payload.update(
            {
                "status": final_status,
                "exit_code": returncode,
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
        return returncode, out_payload
    finally:
        if events_sink is not None:
            events_sink.close()


def _print_atoms_list_human(data: dict) -> None:
    """v2.4: `pcq atoms list` 인간 가독 출력.

    builtin atoms 에 [reference example] 태그를 붙여 contract example 임을 명시.
    project / generated / external atoms 는 [<source>] 로 표시.
    """
    print("== Atoms (registered) ==")
    for kind, entries in data.get("atoms", {}).items():
        print(f"\n  {kind}:")
        if not entries:
            print("    (none)")
            continue
        for e in entries:
            role = e.get("role", "")
            src = e.get("source", "")
            # role 우선 — reference_example 이면 명시 태그, 아니면 source 표시
            if role == "reference_example":
                role_tag = "[reference example]"
            elif src and src != "builtin":
                role_tag = f"[{src}]"
            else:
                role_tag = f"[{role or src or 'unknown'}]"
            tasks = e.get("tasks") or []
            tasks_str = f" tasks={tasks}" if tasks else ""
            extras = e.get("requires_extras") or []
            extras_str = f" requires={extras}" if extras else ""
            print(f"    {e['name']:24s} {role_tag:22s}{tasks_str}{extras_str}")
    print()
    print(
        "Note: 'reference example' atoms exist for contract verification + "
        "onboarding + smoke baselines."
    )
    print(
        "      Production atoms should be project-local — see "
        "`pcq atoms scaffold`."
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    from pcq.agent import inspect_project

    insp = inspect_project(args.path, load_project_atoms=args.load_project_atoms)
    out = insp.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Project Inspection")
    return 1 if insp.errors else 0


def cmd_recipe_meta(args: argparse.Namespace) -> int:
    from pcq.agent import recipe_meta

    try:
        meta = recipe_meta(args.name)
    except Exception as e:
        err = {"schema_version": 1, "error": str(e), "name": args.name}
        if args.json:
            _print_json(err)
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1
    out = {"schema_version": 1, **meta}
    if args.json:
        _print_json(out)
    else:
        _print_human(out, f"Recipe: {args.name}")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """프로젝트 entrypoint detection 후 Trainer 조립 + dry_run plan.

    v1.13: script/experiment style 은 graceful 처리 (preset 없음을 정상으로).
    """
    from pcq.agent import inspect_project
    from pcq.trainer import Trainer

    insp = inspect_project(args.path)
    entry = insp.entrypoint

    # v1.13: script / experiment / unknown — graceful exit (rc=0)
    if entry is None or entry.kind in ("script", "experiment", "unknown"):
        kind_str = entry.kind if entry else "unknown"
        if entry and entry.kind == "script":
            detail = (
                "contract script — no preset/recipe to dry-run. "
                "expected_artifacts are produced by save_all() / explicit "
                "writes during the run."
            )
        elif entry and entry.kind == "experiment":
            detail = (
                "experiment-style — no preset to dry-run. Output via "
                "Experiment.fit()."
            )
        else:
            detail = "no entrypoint detected."
        out = {
            "schema_version": 1,
            "kind": kind_str,
            "entrypoint": entry.to_dict() if entry else None,
            "detail": detail,
            "expected_artifacts": [
                "config.json", "metrics.json",
                "manifest.json", "run_summary.json",
            ],
        }
        if args.json:
            _print_json(out)
        else:
            _print_human(out, "Dry Run")
        return 0

    # trainer style — preset literal 필수
    if not entry.preset:
        out = {
            "schema_version": 1,
            "kind": "trainer",
            "detail": (
                "trainer entrypoint detected but no preset literal found"
            ),
        }
        if args.json:
            _print_json(out)
        else:
            _print_human(out, "Dry Run")
        return 0

    trainer = Trainer(preset=entry.preset)
    plan = trainer.dry_run()
    out = {"schema_version": 1, "preset": entry.preset, **plan}
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Dry Run")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from pcq.agent import ExperimentPlan, ExperimentPlanSet, validate_project
    from pcq.agent.schema import ValidationCheck

    report = validate_project(args.path, strictness=args.strictness)

    # PlanSet 검증 (선택) — --planset 시 schema + 멤버 검증 (registry-aware
    # 검사는 멤버 별 apply 시점에 수행).
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

    # Plan 검증 (선택) — --plan 옵션 시 ExperimentPlan + registry-aware 검사 추가
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
                # registry-aware 검사: set_atom 의 ref 가 실제로 build 가능한지
                try:
                    from pcq import registry as registry_pkg
                    from pcq.registry.spec import AtomRef

                    reg_map = {
                        "model": registry_pkg.models,
                        "dataset": registry_pkg.datasets,
                        "loss": registry_pkg.losses,
                        "optim": registry_pkg.optims,
                        "sched": registry_pkg.scheds,
                        "metric": registry_pkg.metrics,
                    }
                    for i, c in enumerate(plan.changes):
                        if c.op == "set_atom":
                            kind = c._infer_kind()
                            reg = reg_map.get(kind)
                            if reg is None:
                                continue
                            ref = AtomRef(
                                kind=kind,
                                name=c.name or "",
                                params=dict(c.params or {}),
                            )
                            errs = reg.validate_ref(ref)
                            for e in errs:
                                plan_errors.append(
                                    f"changes[{i}].set_atom: {e}"
                                )
                except Exception as e:  # noqa: BLE001
                    plan_errors.append(
                        f"registry-aware validation skipped: {e}"
                    )

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

                # v2.3: plan label contract simulation —
                # set_atom 변경 사항을 base RecipeSpec.atoms 위에 시뮬레이션해
                # ignore_index 충돌을 실행 전에 감지.
                try:
                    from pcq.agent.validate import _validate_plan_label_contracts

                    for check in _validate_plan_label_contracts(plan):
                        report.add(check)
                except Exception as e:  # noqa: BLE001 — best-effort
                    report.add(
                        ValidationCheck(
                            id="plan_label_contract",
                            status="warn",
                            severity="warning",
                            detail=(
                                f"plan label-contract simulation skipped: "
                                f"{type(e).__name__}: {e}"
                            ),
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
            preset=args.preset,
            name=args.name,
            force=args.force,
            style=args.style,
            with_pyproject=args.with_pyproject,
            agent=args.agent if args.agent != "none" else None,
        )
    except Exception as e:  # noqa: BLE001
        err = {"error": str(e), "preset": args.preset}
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
    """run_record.json + validation_report.json 작성 (v1.16, v2.5).

    v2.5: chdir/env tmp file 트릭 제거. finalize_run(output_dir=..., project_root=...)
    직접 호출. output_dir 이름이 "output" 인지 여부는 더 이상 detection 기준이
    아님 — output_dir 의 ancestors 를 walk-up 하며 cq.yaml 탐색.
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

    # output_dir 의 ancestor walk-up 으로 project_root (cq.yaml 가진 dir) 탐색.
    # 못 찾으면 output_dir 의 parent 사용 (legacy 호환).
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
    """output_dir 의 ancestor walk-up 으로 cq.yaml 가진 디렉토리 탐색.

    .git / pyproject.toml 만나면 stop (nested project safeguard).
    못 찾으면 output_dir.parent (legacy 호환).
    """
    cur = output_dir.resolve()
    for d in [cur, *cur.parents][:8]:
        for name in ("cq.yaml", "pcq.yml"):
            if (d / name).exists():
                return d
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            # 같은 디렉토리에 cq.yaml 없는데 root marker 만 — 그것이 root.
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
    """RunRecord 압축 요약 출력 (v1.17)."""
    from pcq.agent.describe import describe_run

    desc = describe_run(args.output_dir)
    out = desc.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Run Description")
    # no_record / corrupted 는 조용히 0 — agent 가 status 로 판단.
    return 0


def cmd_compare_runs(args: argparse.Namespace) -> int:
    """두 RunRecord 비교 diff 출력 (v1.17)."""
    from pcq.agent.compare import compare_runs

    diff = compare_runs(args.a, args.b)
    out = diff.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Run Diff (a → b)")
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    """RunRecord parent chain 출력 (v1.18)."""
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
    """v2.2: cq.yaml + CQ_CONFIG_JSON env → ResolvedConfig (debug view)."""
    from pcq.agent.resolver import resolve_project

    rc = resolve_project(path=args.path, cq_yaml_path=args.cq_yaml)
    out = rc.to_dict()
    if args.json:
        _print_json(out)
    else:
        _print_human(out, "Resolved Project")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """v2.12: fresh-user first-class entry point.

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
    # rc.cfg 는 이미 cq.yaml.configs + CQ_CONFIG_JSON merge 결과. 그대로 dump.
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

    # cmd 는 shell-string. shell=True 로 직접 실행 (cq.yaml 의 cmd 가 작성자
    # 의도대로 — 예: "uv run python train.py", "python -m foo args").
    #
    # JSON mode is machine contract mode: stdout must be parseable JSON only.
    # Child streams are captured to files and summarized in the envelope.
    #
    # JSONL/event mode is live agent mode: child streams are captured while
    # structured events are emitted to stdout and/or an events.jsonl file.
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


def _atom_registry_map() -> dict:
    # 6 카테고리 단일 인스턴스에 대한 kind 매핑.
    from pcq import registry

    return {
        "model": registry.models,
        "dataset": registry.datasets,
        "loss": registry.losses,
        "optim": registry.optims,
        "sched": registry.scheds,
        "metric": registry.metrics,
    }


def cmd_atoms(args: argparse.Namespace) -> int:
    """atom registry inspection — list / show / validate-ref / scaffold / validate-local / smoke."""
    reg_map = _atom_registry_map()

    if args.atom_action == "list":
        # --load-project 로 project atom 사전 로드
        load_path = getattr(args, "load_project", None)
        if load_path:
            from pcq.registry.loader import load_project_atoms

            load_project_atoms(load_path)
        source_filter = getattr(args, "source", None)
        out: dict = {"schema_version": 1, "atoms": {}}
        kinds = [args.kind] if args.kind else list(reg_map)
        for kind in kinds:
            reg = reg_map[kind]
            out["atoms"][kind] = []
            for name in reg.list():
                spec = reg.get(name)
                if source_filter and spec.source != source_filter:
                    continue
                out["atoms"][kind].append(
                    {
                        "name": name,
                        "tasks": spec.tasks,
                        "metadata_status": spec.metadata_status,
                        "requires_extras": spec.requires_extras,
                        "source": spec.source,
                        "module": spec.module,
                        # v2.4: role — atom 의 의도적 위치
                        "role": spec.role,
                    }
                )
        if args.json:
            _print_json(out)
        else:
            # v2.4: 인간 가독 출력 — builtin atoms 에 [reference example] 표시
            _print_atoms_list_human(out)
        return 0

    if args.atom_action == "show":
        if args.kind not in reg_map:
            err = {
                "error": f"unknown kind {args.kind!r}",
                "valid_kinds": sorted(reg_map),
            }
            if args.json:
                _print_json(err)
            else:
                print(err["error"], file=sys.stderr)
            return 1
        try:
            spec = reg_map[args.kind].get(args.name)
        except ValueError as e:
            err = {"error": str(e)}
            if args.json:
                _print_json(err)
            else:
                print(str(e), file=sys.stderr)
            return 1
        out = {"schema_version": 1, **spec.to_dict()}
        if args.json:
            _print_json(out)
        else:
            _print_human(out, f"{args.kind}/{args.name}")
        return 0

    if args.atom_action == "validate-ref":
        ref_path = Path(args.ref_file)
        if not ref_path.exists():
            err = {"error": f"file not found: {ref_path}"}
            if args.json:
                _print_json(err)
            else:
                print(err["error"], file=sys.stderr)
            return 1
        try:
            ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            err = {"error": f"JSON parse: {e}"}
            if args.json:
                _print_json(err)
            else:
                print(err["error"], file=sys.stderr)
            return 1
        from pcq.registry.spec import AtomRef

        try:
            ref = AtomRef.from_dict(ref_data)
        except (KeyError, TypeError) as e:
            err = {"error": f"invalid ref: {e}"}
            if args.json:
                _print_json(err)
            else:
                print(err["error"], file=sys.stderr)
            return 1
        reg = reg_map.get(ref.kind)
        if reg is None:
            err = {
                "error": f"unknown kind {ref.kind!r}",
                "valid_kinds": sorted(reg_map),
            }
            if args.json:
                _print_json(err)
            else:
                print(err["error"], file=sys.stderr)
            return 1
        errors = reg.validate_ref(ref)
        out = {
            "schema_version": 1,
            "kind": ref.kind,
            "name": ref.name,
            "params": ref.params,
            "valid": len(errors) == 0,
            "errors": errors,
        }
        if args.json:
            _print_json(out)
        else:
            _print_human(out, "Validate Ref")
        return 0 if not errors else 1

    if args.atom_action == "scaffold":
        from pcq.agent.scaffold import scaffold_atom

        result = scaffold_atom(
            kind=args.kind,
            name=args.name,
            output=args.output,
            project_root=args.path,
            force=args.force,
        )
        out = result.to_dict()
        if args.json:
            _print_json(out)
        else:
            _print_human(out, f"Scaffold {args.kind}/{args.name}")
        return 0 if result.status in ("created", "skipped") else 1

    if args.atom_action == "validate-local":
        from pcq.agent.scaffold import validate_local_atoms

        report = validate_local_atoms(args.path)
        out = report.to_dict()
        if args.json:
            _print_json(out)
        else:
            _print_human(out, "Project Atoms Validation")
        return 0 if report.status != "fail" else 1

    if args.atom_action == "smoke":
        # --load-project 로 project atom 사전 로드 (smoke 대상이 project atom 인 경우)
        load_path = getattr(args, "load_project", None)
        if load_path:
            from pcq.registry.loader import load_project_atoms

            load_project_atoms(load_path)
        from pcq.agent.smoke import smoke_atom

        report = smoke_atom(args.kind, args.name)
        out = report.to_dict()
        if args.json:
            _print_json(out)
        else:
            _print_human(out, f"Smoke {args.kind}/{args.name}")
        return 0 if report.passed else 1

    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pcq",
        description="pcq CLI — agent-operable JSON interface",
    )
    parser.add_argument("--version", action="version", version=_get_version())
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="inspect project structure")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument(
        "--load-project-atoms",
        action="store_true",
        help="import pcq_atoms.py / atoms/*.py during inspect (default: read-only)",
    )
    p.add_argument("--json", action="store_true", help="emit JSON only")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("recipe-meta", help="inspect one recipe metadata")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recipe_meta)

    p = sub.add_parser("dry-run", help="show assembled execution plan")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_dry_run)

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
            "optional ExperimentPlanSet JSON file (v2.11) — set + member "
            "schema validation"
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

    # `pcq atoms` — sub-subcommand for atom registry inspection.
    p_atoms = sub.add_parser("atoms", help="atom registry inspection")
    atom_sub = p_atoms.add_subparsers(dest="atom_action", required=True)

    p_list = atom_sub.add_parser("list", help="list registered atoms")
    p_list.add_argument(
        "--kind",
        choices=["model", "dataset", "loss", "optim", "sched", "metric"],
        default=None,
    )
    p_list.add_argument(
        "--source",
        choices=["builtin", "project", "generated", "external"],
        default=None,
        help="filter atoms by source",
    )
    p_list.add_argument(
        "--load-project",
        metavar="PATH",
        default=None,
        help="load project atoms from PATH before listing",
    )
    p_list.add_argument("--json", action="store_true")

    p_show = atom_sub.add_parser("show", help="show atom metadata")
    p_show.add_argument(
        "kind",
        choices=["model", "dataset", "loss", "optim", "sched", "metric"],
    )
    p_show.add_argument("name")
    p_show.add_argument("--json", action="store_true")

    p_val = atom_sub.add_parser(
        "validate-ref", help="validate an AtomRef JSON file"
    )
    p_val.add_argument("ref_file")
    p_val.add_argument("--json", action="store_true")

    # v1.12: scaffold — generate project-local atom file skeleton
    p_scaffold = atom_sub.add_parser(
        "scaffold", help="generate project-local atom file skeleton"
    )
    p_scaffold.add_argument(
        "kind",
        choices=["model", "dataset", "loss", "optim", "sched", "metric"],
    )
    p_scaffold.add_argument("name")
    p_scaffold.add_argument(
        "--output",
        default=None,
        help="output file path (default: atoms/<plural>.py)",
    )
    p_scaffold.add_argument(
        "--path",
        default=".",
        help="project root (default: cwd)",
    )
    p_scaffold.add_argument(
        "--force", action="store_true", help="overwrite existing file"
    )
    p_scaffold.add_argument("--json", action="store_true")

    # v1.12: validate-local — project atom contract validation
    p_local = atom_sub.add_parser(
        "validate-local",
        help="validate project-local atoms (pcq_atoms.py / atoms/*.py)",
    )
    p_local.add_argument("path", nargs="?", default=".")
    p_local.add_argument("--json", action="store_true")

    # v1.12: smoke — 1-step contract verification per atom
    p_smoke = atom_sub.add_parser(
        "smoke", help="smoke contract test for one atom"
    )
    p_smoke.add_argument(
        "kind",
        choices=["model", "dataset", "loss", "optim", "sched", "metric"],
    )
    p_smoke.add_argument("name")
    p_smoke.add_argument(
        "--load-project",
        metavar="PATH",
        default=None,
        help="load project atoms from PATH before smoke test",
    )
    p_smoke.add_argument("--json", action="store_true")

    p_atoms.set_defaults(func=cmd_atoms)

    # init-experiment ────────────────────────────────────────────────
    p_init = sub.add_parser(
        "init-experiment", help="scaffold a CQ-runnable ML experiment"
    )
    p_init.add_argument(
        "--style",
        choices=["trainer", "experiment", "script"],
        default="trainer",
        help="entrypoint style (default: trainer)",
    )
    p_init.add_argument(
        "--preset", help="registered recipe name (required for trainer style)"
    )
    p_init.add_argument("--output", default=".", help="output project directory")
    p_init.add_argument(
        "--name", default=None, help="cq.yaml name (default: preset with / → -)"
    )
    p_init.add_argument(
        "--force", action="store_true", help="overwrite existing files"
    )
    p_init.add_argument(
        "--with-pyproject",
        action="store_true",
        help="also generate pyproject.toml (pcq dep + preset extras)"
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
            "apply ExperimentPlanSet — expand member plans into N output dirs "
            "(v2.11)"
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
        help="generate run_record.json + validation_report.json (v1.16)",
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
            "output_dir (v2.12; for output_dir reuse / stale lock-in fix)"
        ),
    )
    p_vr.add_argument("--json", action="store_true")
    p_vr.set_defaults(func=cmd_validate_run)

    # describe-run ───────────────────────────────────────────────────
    p_dr = sub.add_parser(
        "describe-run",
        help="compact RunRecord summary (v1.17)",
    )
    p_dr.add_argument("output_dir", nargs="?", default="output")
    p_dr.add_argument("--json", action="store_true")
    p_dr.set_defaults(func=cmd_describe_run)

    # compare-runs ───────────────────────────────────────────────────
    p_cr = sub.add_parser(
        "compare-runs",
        help="diff two RunRecords (v1.17)",
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
        help="walk RunRecord parent chain (v1.18)",
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
        help="resolve cq.yaml + env into single ResolvedConfig view (v2.2)",
    )
    p_res.add_argument("path", nargs="?", default=".")
    p_res.add_argument("--cq-yaml", default=None, help="explicit cq.yaml path")
    p_res.add_argument("--json", action="store_true")
    p_res.set_defaults(func=cmd_resolve)

    # run ────────────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help=(
            "execute cq.yaml.cmd with auto-wired CQ_CONFIG_JSON (v2.12) — "
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
