"""pcq.agent.validate_run — post-run validation gates.

output_dir 의 contract artifact (manifest, metrics, run_summary, run_record) 를
읽어 일관성을 검사한다. RUN_RECORD.md §"Validation Gates" 의 post-run 단계
구현.

Returns ValidationReport — checks 의 상태가 fail 이면 report.status == "fail".
"""
from __future__ import annotations

import json
from pathlib import Path

from pcq.agent.schema import ValidationCheck, ValidationReport
from pcq.agent.strictness import (
    normalize_strictness,
    strictness_check,
    strictness_name,
)


def _has_service_grade_input_identity(meta: dict) -> bool:
    if meta.get("uri"):
        return True
    if meta.get("opaque") is True and meta.get("reason"):
        return True
    if meta.get("path") and (meta.get("sha256") or meta.get("manifest")):
        return True
    return False


def _add_run_reproducibility_gates(
    report: ValidationReport,
    rr: dict,
    strictness: int,
) -> None:
    if strictness < 3:
        return

    run = rr.get("run") if isinstance(rr.get("run"), dict) else {}

    # v2.11: partial run (still training) 은 reproducibility 평가 대상 아님.
    # source/env/lockfile/seed evidence gates 는 skip — partial 은 학습 중이라
    # 일부 evidence 가 미수집 상태일 수 있다. run_finalized gate (아래) 가
    # 별도로 partial=true 를 fail 처리한다.
    is_partial = bool(run.get("partial"))
    if is_partial:
        report.add(ValidationCheck(
            id="run_finalized",
            status="fail",
            severity="blocking",
            detail=(
                "run not yet finalized — call pcq.finalize_run() to complete "
                "(reproducibility evidence gates skipped while partial=true)"
            ),
            evidence={
                "strictness": strictness,
                "partial": True,
                "status": run.get("status"),
                "last_updated_at": run.get("last_updated_at"),
            },
            suggested_fix=(
                "call pcq.finalize_run() at end of training, or "
                "pcq.save_all(finalize=True)"
            ),
        ))
        return

    # finalize 된 run 은 명시적으로 pass — agent 가 키 존재 여부로 분기 가능.
    report.add(ValidationCheck(
        id="run_finalized",
        status="pass",
        severity="info",
        detail=f"run finalized with status={run.get('status')!r}",
        evidence={"strictness": strictness, "partial": False},
    ))
    execution = (
        rr.get("execution") if isinstance(rr.get("execution"), dict) else {}
    )
    source = rr.get("source") if isinstance(rr.get("source"), dict) else {}
    environment = (
        rr.get("environment") if isinstance(rr.get("environment"), dict) else {}
    )
    config = rr.get("config") if isinstance(rr.get("config"), dict) else {}
    metrics = rr.get("metrics") if isinstance(rr.get("metrics"), dict) else {}
    inputs = rr.get("inputs") if isinstance(rr.get("inputs"), dict) else None

    missing_identity: list[str] = []
    if not run.get("name"):
        missing_identity.append("run.name")
    if not execution.get("cmd"):
        missing_identity.append("execution.cmd")
    if missing_identity:
        report.add(ValidationCheck(
            id="run_record_execution_identity",
            status="fail",
            severity="blocking",
            detail=f"missing execution identity: {missing_identity}",
            evidence={"strictness": strictness, "missing": missing_identity},
            suggested_fix=(
                "set top-level `name` and `cmd` in cq.yaml or inject "
                "`_run_name`/`_cmd`"
            ),
        ))
    else:
        report.add(ValidationCheck(
            id="run_record_execution_identity",
            status="pass",
            severity="info",
            detail="run name and command are recorded",
            evidence={"strictness": strictness},
        ))

    missing_source: list[str] = []
    if not source.get("git_sha"):
        missing_source.append("source.git_sha")
    if "dirty" not in source:
        missing_source.append("source.dirty")
    if not source.get("cq_yaml_sha256") and not config.get("cq_yaml_sha256"):
        missing_source.append("cq_yaml_sha256")
    if missing_source:
        report.add(ValidationCheck(
            id="source_reproducibility",
            status="fail",
            severity="blocking",
            detail=f"missing source evidence: {missing_source}",
            evidence={"strictness": strictness, "missing": missing_source},
            suggested_fix=(
                "run inside a git worktree with cq.yaml present before "
                "finalizing the run"
            ),
        ))
    else:
        report.add(ValidationCheck(
            id="source_reproducibility",
            status="pass",
            severity="info",
            detail=(
                f"git_sha={source.get('git_sha')}, dirty={source.get('dirty')}"
            ),
            evidence={"strictness": strictness},
        ))

    missing_env = [
        key for key in ("python", "platform", "pcq_version")
        if not environment.get(key)
    ]
    if missing_env:
        report.add(ValidationCheck(
            id="environment_reproducibility",
            status="fail",
            severity="blocking",
            detail=f"missing environment evidence: {missing_env}",
            evidence={"strictness": strictness, "missing": missing_env},
            suggested_fix="finalize with current pcq so environment evidence is captured",
        ))
    else:
        report.add(ValidationCheck(
            id="environment_reproducibility",
            status="pass",
            severity="info",
            detail="python/platform environment evidence recorded",
            evidence={"strictness": strictness},
        ))

    if environment.get("lockfile") and environment.get("lockfile_sha256"):
        report.add(ValidationCheck(
            id="lockfile_evidence",
            status="pass",
            severity="info",
            detail=f"lockfile recorded: {environment.get('lockfile')}",
            evidence={"strictness": strictness},
        ))
    else:
        report.add(ValidationCheck(
            id="lockfile_evidence",
            status="fail",
            severity="blocking",
            detail="strictness>=3 requires lockfile and lockfile_sha256",
            evidence={"strictness": strictness},
            suggested_fix="create a lockfile, then rerun pcq.finalize_run()",
        ))

    if "seed" in config:
        report.add(ValidationCheck(
            id="seed_evidence",
            status="pass",
            severity="info",
            detail=f"seed recorded: {config.get('seed')}",
            evidence={"strictness": strictness},
        ))
    else:
        report.add(ValidationCheck(
            id="seed_evidence",
            status="fail",
            severity="blocking",
            detail="strictness>=3 requires run_record.config.seed",
            evidence={"strictness": strictness},
            suggested_fix="set `configs.seed` in cq.yaml and rerun finalize",
        ))

    declared = metrics.get("declared")
    if not isinstance(declared, list) or not declared:
        report.add(ValidationCheck(
            id="metrics_schema_evidence",
            status="fail",
            severity="blocking",
            detail="strictness>=3 requires declared metric evidence",
            evidence={"strictness": strictness},
            suggested_fix="declare metrics in cq.yaml or configure run_summary.monitor",
        ))
    else:
        report.add(ValidationCheck(
            id="metrics_schema_evidence",
            status="pass",
            severity="info",
            detail=f"{len(declared)} metric declaration(s) recorded",
            evidence={"strictness": strictness},
        ))

    if inputs is None:
        report.add(ValidationCheck(
            id="run_record_inputs_evidence",
            status="fail",
            severity="blocking",
            detail="run_record.inputs missing",
            evidence={"strictness": strictness},
            suggested_fix="finalize with current pcq so inputs are recorded",
        ))
    else:
        report.add(ValidationCheck(
            id="run_record_inputs_evidence",
            status="pass",
            severity="info",
            detail=f"inputs recorded: {sorted(inputs)}",
            evidence={
                "strictness": strictness,
                "input_count": len(inputs),
                "explicitly_empty": len(inputs) == 0,
            },
        ))

    if strictness < 4:
        return

    if not inputs:
        report.add(ValidationCheck(
            id="service_input_identity",
            status="fail",
            severity="blocking",
            detail="strictness=4 requires at least one service-grade input",
            evidence={"strictness": strictness, "input_count": 0},
            suggested_fix=(
                "record inputs with `uri`, `path`+`sha256`, `path`+`manifest`, "
                "or `opaque: true` with `reason`"
            ),
        ))
    else:
        missing = [
            name for name, meta in inputs.items()
            if not isinstance(meta, dict)
            or not _has_service_grade_input_identity(meta)
        ]
        if missing:
            report.add(ValidationCheck(
                id="service_input_identity",
                status="fail",
                severity="blocking",
                detail=f"inputs missing service-grade identity: {missing}",
                evidence={"strictness": strictness, "missing": missing},
                suggested_fix=(
                    "record each input with `uri`, `path`+`sha256`, "
                    "`path`+`manifest`, or `opaque: true` with `reason`"
                ),
            ))
        else:
            report.add(ValidationCheck(
                id="service_input_identity",
                status="pass",
                severity="info",
                detail="all inputs have service-grade identity",
                evidence={"strictness": strictness, "inputs": sorted(inputs)},
            ))

    invalid_metric_modes: list[str] = []
    for metric in declared or []:
        if not isinstance(metric, dict):
            invalid_metric_modes.append(str(metric))
            continue
        if metric.get("mode") not in ("min", "max"):
            invalid_metric_modes.append(str(metric.get("name", "<unknown>")))
    if declared and not invalid_metric_modes:
        report.add(ValidationCheck(
            id="service_metric_schema",
            status="pass",
            severity="info",
            detail="all declared metrics include min/max mode",
            evidence={"strictness": strictness},
        ))
    else:
        report.add(ValidationCheck(
            id="service_metric_schema",
            status="fail",
            severity="blocking",
            detail=(
                "strictness=4 requires declared metrics with mode=min|max"
                if not declared
                else f"metrics missing valid mode: {invalid_metric_modes}"
            ),
            evidence={"strictness": strictness},
            suggested_fix=(
                "declare metrics as a mapping, for example "
                "`metrics: {eval_acc: {mode: max}}`"
            ),
        ))

    hardware_keys = [
        key for key in (
            "device",
            "cuda",
            "cuda_available",
            "gpu",
            "gpu_model",
            "gpu_count",
            "world_size",
        )
        if key in environment
    ]
    if hardware_keys:
        report.add(ValidationCheck(
            id="service_hardware_evidence",
            status="pass",
            severity="info",
            detail=f"hardware evidence keys: {hardware_keys}",
            evidence={"strictness": strictness, "keys": hardware_keys},
        ))
    else:
        report.add(ValidationCheck(
            id="service_hardware_evidence",
            status="fail",
            severity="blocking",
            detail="strictness=4 requires hardware/device evidence",
            evidence={"strictness": strictness},
            suggested_fix=(
                "record CPU/GPU device, GPU count/model, or distributed "
                "world size in run_record.environment"
            ),
        ))

    agent = rr.get("agent") if isinstance(rr.get("agent"), dict) else {}
    lineage_present = any(
        value for value in (
            agent.get("plan_id"),
            agent.get("intent"),
            run.get("parent_run_id"),
            run.get("parent_run_path"),
        )
    )
    if lineage_present:
        report.add(ValidationCheck(
            id="service_lineage_evidence",
            status="pass",
            severity="info",
            detail="plan intent or parent lineage recorded",
            evidence={"strictness": strictness},
        ))
    else:
        report.add(ValidationCheck(
            id="service_lineage_evidence",
            status="fail",
            severity="blocking",
            detail="strictness=4 requires plan intent or lineage evidence",
            evidence={"strictness": strictness},
            suggested_fix=(
                "finalize with plan_id/intent or parent_run_id/parent_run_path"
            ),
        ))


def _read_partial_flag(output_dir: Path) -> bool:
    """run_record.json 의 run.partial 을 best-effort 로 read.

    파일 없거나 invalid 면 False — 정상 finalize 흐름과 동일.
    """
    rr_path = output_dir / "run_record.json"
    if not rr_path.exists():
        return False
    try:
        rr = json.loads(rr_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    run = rr.get("run") if isinstance(rr.get("run"), dict) else {}
    return bool(run.get("partial"))


def validate_run(
    output_dir: str | Path,
    strictness: int | None = 2,
    *,
    rescan_manifest: bool = False,
) -> ValidationReport:
    """post-run gates per RUN_RECORD.md §'Validation Gates'.

    Gates:
      - manifest_present + manifest_evidence (sha256 verify for v2)
      - metrics_present + metrics_well_formed
      - run_summary_present + summary_metrics_consistent
      - run_record_present + run_record_complete

    v2.11: when run_record.run.partial == True, manifest/run_summary
    presence checks are downgraded to skip (warn-info), since the run is
    still training. Reproducibility gates also skip with a single
    `run_finalized` fail at strictness>=3.

    v2.12: ``rescan_manifest=True`` skips manifest entries whose files no
    longer exist on disk (for output_dir reuse — stale lock-in fix).
    sha256 mismatch on present files is still failed.
    """
    strictness = normalize_strictness(strictness)
    out = Path(output_dir).resolve()
    report = ValidationReport(
        strictness=strictness,
        strictness_name=strictness_name(strictness),
    )
    report.add(strictness_check(strictness))

    # v2.11: partial 은 학습 중 — 일부 evidence 미수집 정상.
    is_partial = _read_partial_flag(out)

    # ── 1. manifest.json 존재 + 항목 sha256 검증 ─────────────────────
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        if is_partial:
            # 학습 중 — manifest 는 finalize 시점에 작성. 빠진 것이 정상.
            report.add(
                ValidationCheck(
                    id="manifest_present",
                    status="warn",
                    severity="warning",
                    detail="manifest.json missing (run is partial — expected before finalize)",
                    evidence={"partial": True},
                )
            )
        else:
            report.add(
                ValidationCheck(
                    id="manifest_present",
                    status="fail",
                    severity="blocking",
                    detail=f"manifest.json missing in {out}",
                )
            )
        manifest_data: dict | None = None
    else:
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            report.add(
                ValidationCheck(
                    id="manifest_parseable",
                    status="fail",
                    severity="blocking",
                    detail=f"manifest.json invalid JSON: {e}",
                )
            )
            manifest_data = None
        else:
            files = manifest_data.get("files", []) or []
            schema_v = manifest_data.get("schema_version", 1)
            missing: list[str] = []
            sha_mismatch: list[str] = []
            ignored_missing: list[str] = []
            for entry in files:
                rel = entry.get("path", "")
                p = out / rel
                if not p.exists():
                    if rescan_manifest:
                        # output_dir reuse 대응 — manifest 안의 stale 항목 무시.
                        ignored_missing.append(rel)
                        continue
                    missing.append(rel)
                    continue
                # schema v2 — sha256 round-trip.
                if schema_v >= 2 and entry.get("sha256"):
                    from pcq.contract import _sha256_file

                    actual = _sha256_file(p)
                    if actual != entry["sha256"]:
                        sha_mismatch.append(rel)
            if missing or sha_mismatch:
                detail_parts: list[str] = []
                if missing:
                    detail_parts.append(f"missing: {missing}")
                if sha_mismatch:
                    detail_parts.append(f"sha256 mismatch: {sha_mismatch}")
                report.add(
                    ValidationCheck(
                        id="manifest_evidence",
                        status="fail",
                        severity="blocking",
                        detail="; ".join(detail_parts),
                        suggested_fix=(
                            "rerun with --rescan-manifest to ignore stale "
                            "missing entries, or call pcq.save_manifest() to "
                            "refresh the manifest from current output_dir"
                        ),
                    )
                )
            else:
                detail = (
                    f"schema v{schema_v}, {len(files)} entries verified"
                )
                if ignored_missing:
                    detail += (
                        f" (rescan_manifest: ignored {len(ignored_missing)} "
                        f"stale entries)"
                    )
                evidence: dict = {}
                if ignored_missing:
                    evidence["ignored_missing"] = ignored_missing
                report.add(
                    ValidationCheck(
                        id="manifest_evidence",
                        status="pass",
                        severity="info",
                        detail=detail,
                        evidence=evidence,
                    )
                )

    # ── 2. metrics.json well-formed ─────────────────────────────────
    metrics_path = out / "metrics.json"
    metrics_data: dict | None = None
    if not metrics_path.exists():
        report.add(
            ValidationCheck(
                id="metrics_present",
                status="fail",
                severity="blocking",
                detail="metrics.json missing",
            )
        )
    else:
        try:
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            report.add(
                ValidationCheck(
                    id="metrics_well_formed",
                    status="fail",
                    severity="blocking",
                    detail=f"metrics.json invalid: {e}",
                )
            )
        else:
            history = metrics_data.get("history", [])
            report.add(
                ValidationCheck(
                    id="metrics_well_formed",
                    status="pass",
                    severity="info",
                    detail=f"{len(history)} epoch(s) recorded",
                )
            )

    # ── 3. run_summary.json best/last 가 metrics history 와 일치 ───
    rs_path = out / "run_summary.json"
    if not rs_path.exists():
        if is_partial:
            report.add(
                ValidationCheck(
                    id="run_summary_present",
                    status="warn",
                    severity="warning",
                    detail="run_summary.json missing (partial run)",
                    evidence={"partial": True},
                )
            )
        else:
            report.add(
                ValidationCheck(
                    id="run_summary_present",
                    status="fail",
                    severity="blocking",
                    detail="run_summary.json missing",
                )
            )
    else:
        try:
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            report.add(
                ValidationCheck(
                    id="run_summary_parseable",
                    status="fail",
                    severity="blocking",
                    detail=f"run_summary.json invalid: {e}",
                )
            )
        else:
            history = (
                metrics_data.get("history", []) if metrics_data else []
            )
            best = rs.get("best") or {}
            last = rs.get("last") or {}
            consistent = True
            issues: list[str] = []
            if best.get("epoch") is not None:
                epochs = [
                    e.get("epoch")
                    for e in history
                    if isinstance(e, dict)
                ]
                if best["epoch"] not in epochs:
                    consistent = False
                    issues.append(
                        f"best.epoch={best['epoch']} not in history"
                    )
            if last.get("epoch") is not None and history:
                last_h = history[-1] if history else {}
                if isinstance(last_h, dict) and last_h.get("epoch") != last["epoch"]:
                    consistent = False
                    issues.append(
                        f"last.epoch={last['epoch']} != "
                        f"history[-1].epoch={last_h.get('epoch')}"
                    )
            if consistent:
                report.add(
                    ValidationCheck(
                        id="summary_metrics_consistent",
                        status="pass",
                        severity="info",
                        detail="run_summary best/last align with metrics history",
                    )
                )
            else:
                report.add(
                    ValidationCheck(
                        id="summary_metrics_consistent",
                        status="fail",
                        severity="blocking",
                        detail="; ".join(issues),
                    )
                )

    # ── 4. run_record.json 존재 + 필수 키 ───────────────────────────
    rr_path = out / "run_record.json"
    if not rr_path.exists():
        missing_rr_status = "fail" if strictness >= 3 else "warn"
        missing_rr_severity = "blocking" if strictness >= 3 else "warning"
        report.add(
            ValidationCheck(
                id="run_record_present",
                status=missing_rr_status,
                severity=missing_rr_severity,
                detail=(
                    "run_record.json missing — call pcq.finalize_run() at end"
                ),
                evidence={"strictness": strictness},
                suggested_fix="call pcq.finalize_run() or pcq.save_all(finalize=True)",
            )
        )
    else:
        try:
            rr = json.loads(rr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            report.add(
                ValidationCheck(
                    id="run_record_parseable",
                    status="fail",
                    severity="blocking",
                    detail=f"run_record.json invalid: {e}",
                )
            )
        else:
            required = (
                "schema_version",
                "run",
                "execution",
                "source",
                "environment",
                "metrics",
                "artifacts",
            )
            missing_keys = [k for k in required if k not in rr]
            if missing_keys:
                report.add(
                    ValidationCheck(
                        id="run_record_complete",
                        status="fail",
                        severity="blocking",
                        detail=f"run_record missing keys: {missing_keys}",
                    )
                )
            else:
                report.add(
                    ValidationCheck(
                        id="run_record_complete",
                        status="pass",
                        severity="info",
                        detail=(
                            f"run_record schema v{rr.get('schema_version')}, "
                            "all required keys present"
                        ),
                    )
                )
                _add_run_reproducibility_gates(
                    report, rr, strictness
                )

    return report
