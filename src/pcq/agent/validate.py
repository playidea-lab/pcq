"""pcq.agent.validate — pre-run gates.

v4.0: contract-script focus only. recipe/atom/Trainer 의존 gate 제거.
유지되는 gate:
  - cq.yaml: cq_yaml_exists, cq_yaml_parseable, cmd_defined, metrics_declared,
             artifacts_declared, metric_schema_*, inputs_*, monitor_*
  - script: cq_config_called, cq_log_called, standard_artifacts_helper,
            detected_frameworks
  - post-run: manifest_evidence
  - reproducibility (strictness>=3): seed_evidence, lockfile_evidence,
            inputs_evidence
  - service-grade (strictness=4): service_input_identity, service_metric_schema,
            service_lineage_evidence
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pcq.agent.inspect import inspect_project
from pcq.agent.schema import ValidationCheck, ValidationReport
from pcq.agent.strictness import (
    normalize_strictness,
    strictness_check,
    strictness_name,
)


def _add_script_aware_gates(
    report: ValidationReport, insp: Any,
) -> None:
    """script style entrypoint 에 대한 gate 들 (v1.13).

    blocking: cq_config_called.
    warning: cq_log_called, standard_artifacts_helper.
    info: detected_frameworks.
    """
    cq_calls = (
        insp.entrypoint.cq_calls if insp.entrypoint else []
    )

    # 1. pcq.config 호출 (blocking)
    if "pcq.config" not in cq_calls:
        report.add(
            ValidationCheck(
                id="cq_config_called",
                status="fail",
                severity="blocking",
                detail=(
                    "contract script must call pcq.config() to read CQ_CONFIG_JSON"
                ),
                suggested_fix=(
                    "add `cfg = pcq.config()` near the top of the entrypoint"
                ),
            )
        )
    else:
        report.add(
            ValidationCheck(
                id="cq_config_called",
                status="pass",
                severity="info",
                detail="pcq.config() detected",
            )
        )

    # 2. pcq.log (warning)
    if "pcq.log" not in cq_calls:
        report.add(
            ValidationCheck(
                id="cq_log_called",
                status="warn",
                severity="warning",
                detail=(
                    "pcq.log() not detected — declared metrics may not be emitted"
                ),
                suggested_fix=(
                    "add `pcq.log(epoch=..., metric=value)` after evaluation"
                ),
            )
        )
    else:
        report.add(
            ValidationCheck(
                id="cq_log_called",
                status="pass",
                severity="info",
                detail="pcq.log() detected",
            )
        )

    # 3. pcq.save_* helper (warning)
    save_calls = {c for c in cq_calls if c.startswith("pcq.save_")}
    if not save_calls:
        report.add(
            ValidationCheck(
                id="standard_artifacts_helper",
                status="warn",
                severity="warning",
                detail=(
                    "no pcq.save_* helper detected — standard artifacts may not "
                    "be written"
                ),
                suggested_fix=(
                    "end the script with pcq.save_all(history=[...]) for full "
                    "contract compliance"
                ),
            )
        )
    else:
        report.add(
            ValidationCheck(
                id="standard_artifacts_helper",
                status="pass",
                severity="info",
                detail=f"detected: {sorted(save_calls)}",
            )
        )

    # 4. detected ML frameworks (info)
    detected = (
        insp.entrypoint.detected_imports if insp.entrypoint else []
    )
    report.add(
        ValidationCheck(
            id="detected_frameworks",
            status="pass",
            severity="info",
            detail=(
                f"detected frameworks: {detected}"
                if detected
                else "no ML frameworks detected in entrypoint imports"
            ),
        )
    )


def _validate_metric_schema(insp: Any) -> list[ValidationCheck]:
    """metric dict-style 사용 시 schema 완전성 검증 (v1.15).

    list-style legacy 인 경우 빈 list 반환 (skip).
    각 metric:
      - schema 가 mapping 이 아니면 warn
      - mode 미선언 warn
      - mode 가 min/max 가 아니면 fail (blocking)
    """
    checks: list[ValidationCheck] = []
    if not insp.cq_yaml or not insp.cq_yaml.metrics_schema:
        return checks
    schema_map = insp.cq_yaml.metrics_schema

    has_issue = False
    for name, schema in schema_map.items():
        if not isinstance(schema, dict):
            checks.append(ValidationCheck(
                id="metric_schema_format",
                status="warn",
                severity="warning",
                detail=f"metric {name!r}: schema not a mapping",
            ))
            has_issue = True
            continue
        if "mode" not in schema:
            checks.append(ValidationCheck(
                id="metric_schema_mode",
                status="warn",
                severity="warning",
                detail=f"metric {name!r}: 'mode' (min|max) not declared",
                suggested_fix=(
                    f"add `mode: min` or `mode: max` to metrics.{name}"
                ),
            ))
            has_issue = True
        elif schema["mode"] not in ("min", "max"):
            checks.append(ValidationCheck(
                id="metric_schema_mode_value",
                status="fail",
                severity="blocking",
                detail=(
                    f"metric {name!r}: mode={schema['mode']!r}, "
                    "expected 'min' or 'max'"
                ),
                suggested_fix=(
                    f"set metrics.{name}.mode to 'min' or 'max'"
                ),
            ))
            has_issue = True

    if not has_issue:
        checks.append(ValidationCheck(
            id="metric_schema_complete",
            status="pass",
            severity="info",
            detail=(
                f"all {len(schema_map)} metrics have valid schema"
            ),
        ))
    return checks


def _validate_inputs(insp: Any) -> list[ValidationCheck]:
    """inputs 섹션이 있을 때 dataset identity 점검 (v1.15).

    inputs 미사용 → 빈 list (skip).
    각 input 의 mapping 형태 + name 필드 권장.
    cq URI 는 opaque — parse/fetch 안 함.
    """
    checks: list[ValidationCheck] = []
    if not insp.cq_yaml or not insp.cq_yaml.inputs:
        return checks

    has_issue = False
    for input_name, meta in insp.cq_yaml.inputs.items():
        if not isinstance(meta, dict):
            checks.append(ValidationCheck(
                id="input_format",
                status="warn",
                severity="warning",
                detail=f"input {input_name!r}: not a mapping",
            ))
            has_issue = True
            continue
        if "name" not in meta:
            checks.append(ValidationCheck(
                id="input_identity",
                status="warn",
                severity="warning",
                detail=f"input {input_name!r}: 'name' field missing",
                suggested_fix=(
                    f"add `name: <dataset>` to inputs.{input_name}"
                ),
            ))
            has_issue = True

    if not has_issue:
        checks.append(ValidationCheck(
            id="inputs_declared",
            status="pass",
            severity="info",
            detail=(
                f"declared {len(insp.cq_yaml.inputs)} input(s): "
                f"{sorted(insp.cq_yaml.inputs)}"
            ),
        ))
    return checks


def _validate_monitor_in_metric_schema(insp: Any) -> ValidationCheck | None:
    """cfg.monitor 가 metric_schema 의 key 인지 + cfg.mode 와 schema.mode 일치 (v1.15).

    metrics_schema 미사용 또는 monitor 미선언 → None (skip).
    """
    if not insp.cq_yaml or not insp.cq_yaml.metrics_schema:
        return None
    # cq.yaml.configs.monitor / configs.mode 추출 — yaml_io 한 번 더 호출.
    try:
        from pcq.agent.yaml_io import read_yaml
        data = read_yaml(insp.cq_yaml.path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    configs = data.get("configs")
    if not isinstance(configs, dict):
        return None
    monitor = configs.get("monitor")
    if not monitor:
        return None
    schema_keys = set(insp.cq_yaml.metrics_schema.keys())
    if monitor not in schema_keys:
        return ValidationCheck(
            id="monitor_in_metric_schema",
            status="warn",
            severity="warning",
            detail=(
                f"monitor={monitor!r} not in metrics dict {sorted(schema_keys)}"
            ),
            suggested_fix=(
                f"add `{monitor}: {{mode: ...}}` to metrics or change monitor"
            ),
        )
    cfg_mode = configs.get("mode")
    schema_mode = insp.cq_yaml.metrics_schema.get(monitor, {}).get("mode")
    if cfg_mode and schema_mode and cfg_mode != schema_mode:
        return ValidationCheck(
            id="monitor_mode_consistency",
            status="warn",
            severity="warning",
            detail=(
                f"cfg.mode={cfg_mode!r} != metrics.{monitor}.mode="
                f"{schema_mode!r}"
            ),
            suggested_fix="align cfg.mode with metric schema",
        )
    return ValidationCheck(
        id="monitor_in_metric_schema",
        status="pass",
        severity="info",
        detail=f"monitor {monitor!r} declared with mode={schema_mode!r}",
    )


def _find_output_dir(project_root: Path) -> Path | None:
    """post-run gate 용 output 디렉토리 탐지.

    v2.5: ResolvedConfig.output_dir 우선 (cq.yaml.configs.output_dir 또는
    CQ_CONFIG_JSON.output_dir 기반). 없으면 legacy 'output' fallback.
    READ-ONLY: 디렉토리 만들지 않음. 존재하지 않으면 None.
    """
    from pcq.agent.resolver import resolve_project

    rc = resolve_project(path=project_root)
    if rc.output_dir is not None and rc.output_dir.exists() and rc.output_dir.is_dir():
        return rc.output_dir
    legacy = project_root / "output"
    if legacy.exists() and legacy.is_dir():
        return legacy
    return None


def _validate_manifest_post_run(project_root: Path) -> ValidationCheck | None:
    """post-run: manifest entries 가 real file 가리키는지 + sha256 verify (v2).

    output 디렉토리 또는 manifest.json 가 없으면 None (skip).
    schema v1: file 존재 검증만. v2: sha256 round-trip 추가.
    """
    output_dir = _find_output_dir(project_root)
    if output_dir is None:
        return None
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ValidationCheck(
            id="manifest_evidence",
            status="fail",
            severity="blocking",
            detail=f"manifest.json invalid JSON: {e}",
            suggested_fix="rebuild manifest with pcq.save_manifest()",
        )

    schema_v = m.get("schema_version", 1)
    files = m.get("files", [])
    missing: list[str] = []
    sha_mismatch: list[str] = []

    # v2 경로에서만 hashlib 필요 — 지연 import 로 v1 호환 비용 회피.
    sha_fn = None
    if schema_v >= 2:
        from pcq.contract import _sha256_file as sha_fn  # type: ignore[no-redef]

    for entry in files:
        path_str = entry.get("path")
        if not path_str:
            continue
        full = output_dir / path_str
        if not full.exists():
            missing.append(path_str)
            continue
        if schema_v >= 2 and entry.get("sha256") and sha_fn is not None:
            actual = sha_fn(full)
            declared = entry["sha256"]
            if actual != declared:
                sha_mismatch.append(
                    f"{path_str}: declared={declared[:8]} actual={actual[:8]}"
                )

    if missing or sha_mismatch:
        details: list[str] = []
        if missing:
            details.append(f"missing files: {missing}")
        if sha_mismatch:
            details.append(f"sha256 mismatch: {sha_mismatch}")
        return ValidationCheck(
            id="manifest_evidence",
            status="fail",
            severity="blocking",
            detail="; ".join(details),
            suggested_fix=(
                "re-run experiment or rebuild manifest with pcq.save_manifest()"
            ),
        )
    return ValidationCheck(
        id="manifest_evidence",
        status="pass",
        severity="info",
        detail=f"manifest schema v{schema_v}, {len(files)} entries verified",
    )


_LOCKFILE_CANDIDATES = (
    "uv.lock",
    "poetry.lock",
    "pdm.lock",
    "conda-lock.yml",
    "requirements.lock",
)


def _cq_yaml_has_top_level_key(path: Path | None, key: str) -> bool:
    if path is None:
        return False
    try:
        from pcq.agent.yaml_io import read_yaml

        data = read_yaml(path)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(data, dict) and key in data


def _has_service_grade_input_identity(meta: dict) -> bool:
    if meta.get("uri"):
        return True
    if meta.get("opaque") is True and meta.get("reason"):
        return True
    if meta.get("path") and (meta.get("sha256") or meta.get("manifest")):
        return True
    return False


def _add_project_reproducibility_gates(
    report: ValidationReport,
    project_root: Path,
    strictness: int,
) -> None:
    """Level 3/4 pre-run missing-evidence gates.

    These checks do not collect new evidence; they make the selected strictness
    level explicit so agents know which project metadata must be added before
    a run can be treated as reproducible or service-grade.
    """
    if strictness < 3:
        return

    from pcq.agent.resolver import resolve_project

    rc = resolve_project(path=project_root)
    cfg = rc.cfg

    seed_keys = [k for k in ("seed", "random_seed") if k in cfg]
    if seed_keys:
        report.add(ValidationCheck(
            id="seed_evidence",
            status="pass",
            severity="info",
            detail=f"seed configured via {seed_keys[0]}",
            evidence={"strictness": strictness, "keys": seed_keys},
        ))
    else:
        report.add(ValidationCheck(
            id="seed_evidence",
            status="fail",
            severity="blocking",
            detail="strictness>=3 requires an explicit seed in cq.yaml.configs",
            evidence={"strictness": strictness, "expected": "configs.seed"},
            suggested_fix="add `configs.seed: 42` to cq.yaml",
        ))

    lockfiles = [
        name for name in _LOCKFILE_CANDIDATES
        if (project_root / name).exists()
    ]
    if lockfiles:
        report.add(ValidationCheck(
            id="lockfile_evidence",
            status="pass",
            severity="info",
            detail=f"lockfile present: {lockfiles[0]}",
            evidence={"strictness": strictness, "lockfiles": lockfiles},
        ))
    else:
        report.add(ValidationCheck(
            id="lockfile_evidence",
            status="fail",
            severity="blocking",
            detail="strictness>=3 requires dependency lockfile evidence",
            evidence={
                "strictness": strictness,
                "expected": list(_LOCKFILE_CANDIDATES),
            },
            suggested_fix="create and commit a lockfile, for example `uv lock`",
        ))

    inputs_key_present = _cq_yaml_has_top_level_key(rc.cq_yaml_path, "inputs")
    if inputs_key_present:
        report.add(ValidationCheck(
            id="inputs_evidence",
            status="pass",
            severity="info",
            detail=f"inputs declared: {sorted(rc.inputs)}",
            evidence={
                "strictness": strictness,
                "input_count": len(rc.inputs),
                "explicitly_empty": len(rc.inputs) == 0,
            },
        ))
    else:
        report.add(ValidationCheck(
            id="inputs_evidence",
            status="fail",
            severity="blocking",
            detail=(
                "strictness>=3 requires cq.yaml.inputs to declare input "
                "identity or explicitly record no inputs"
            ),
            evidence={"strictness": strictness, "expected": "inputs"},
            suggested_fix="add `inputs: {}` or declare dataset inputs in cq.yaml",
        ))

    if strictness < 4:
        return

    if not rc.inputs:
        report.add(ValidationCheck(
            id="service_input_identity",
            status="fail",
            severity="blocking",
            detail="strictness=4 requires at least one service-grade input",
            evidence={"strictness": strictness, "input_count": 0},
            suggested_fix=(
                "declare inputs with `uri`, `path`+`sha256`, "
                "`path`+`manifest`, or `opaque: true` with `reason`"
            ),
        ))
    else:
        missing_identity = [
            name for name, meta in rc.inputs.items()
            if not _has_service_grade_input_identity(meta)
        ]
        if missing_identity:
            report.add(ValidationCheck(
                id="service_input_identity",
                status="fail",
                severity="blocking",
                detail=(
                    "inputs missing service-grade identity: "
                    f"{missing_identity}"
                ),
                evidence={
                    "strictness": strictness,
                    "missing": missing_identity,
                },
                suggested_fix=(
                    "add `uri`, `path`+`sha256`, `path`+`manifest`, or "
                    "`opaque: true` with `reason` for each input"
                ),
            ))
        else:
            report.add(ValidationCheck(
                id="service_input_identity",
                status="pass",
                severity="info",
                detail="all inputs have service-grade identity",
                evidence={
                    "strictness": strictness,
                    "inputs": sorted(rc.inputs),
                },
            ))

    invalid_metric_modes = [
        name for name, schema in rc.metrics_schema.items()
        if schema.get("mode") not in ("min", "max")
    ]
    if rc.metrics_schema and not invalid_metric_modes:
        report.add(ValidationCheck(
            id="service_metric_schema",
            status="pass",
            severity="info",
            detail="all dict-style metrics declare min/max mode",
            evidence={
                "strictness": strictness,
                "metrics": sorted(rc.metrics_schema),
            },
        ))
    else:
        detail = (
            "strictness=4 requires dict-style metrics with mode=min|max"
            if not rc.metrics_schema
            else f"metrics missing valid mode: {invalid_metric_modes}"
        )
        report.add(ValidationCheck(
            id="service_metric_schema",
            status="fail",
            severity="blocking",
            detail=detail,
            evidence={"strictness": strictness},
            suggested_fix=(
                "declare metrics as a mapping, for example "
                "`metrics: {eval_acc: {mode: max}}`"
            ),
        ))

    lineage_keys = [
        k for k in (
            "_plan_id",
            "plan_id",
            "_plan_intent",
            "intent",
            "_parent_run_id",
            "parent_run_id",
            "_parent_run_path",
            "parent_run_path",
        )
        if cfg.get(k)
    ]
    if lineage_keys:
        report.add(ValidationCheck(
            id="service_lineage_evidence",
            status="pass",
            severity="info",
            detail=f"lineage/plan evidence configured via {lineage_keys}",
            evidence={"strictness": strictness, "keys": lineage_keys},
        ))
    else:
        report.add(ValidationCheck(
            id="service_lineage_evidence",
            status="fail",
            severity="blocking",
            detail="strictness=4 requires plan intent or lineage evidence",
            evidence={"strictness": strictness},
            suggested_fix=(
                "set `_plan_id`, `_plan_intent`, `parent_run_id`, or "
                "`parent_run_path` in the injected config"
            ),
        ))


def _strictness_from_project(path: str | Path, explicit: int | None) -> int:
    if explicit is not None:
        return normalize_strictness(explicit)
    try:
        from pcq.agent.resolver import resolve_project

        return normalize_strictness(resolve_project(path=path).cfg.get("strictness"))
    except Exception:  # noqa: BLE001
        return normalize_strictness(None)


def validate_project(
    path: str | Path = ".", strictness: int | None = None
) -> ValidationReport:
    """Static + script-aware 검증.

    v4.0: contract-script 만 지원 — recipe/atom-level gates 제거.

    strictness:
      0: cq.yaml/cmd parseability only.
      1: static cq.yaml + script contract checks, no post-run.
      2: default; static + script contract + post-run evidence when present.
      3: level 2 + reproducibility evidence requirements.
      4: level 3 + service-grade input, metric, and lineage requirements.
    """
    strictness = _strictness_from_project(path, strictness)
    insp = inspect_project(path)
    report = ValidationReport(
        strictness=strictness,
        strictness_name=strictness_name(strictness),
    )
    report.add(strictness_check(strictness))

    # — Gate 1: Static Project Contract —
    report.add(
        ValidationCheck(
            id="cq_yaml_exists",
            status="pass" if insp.has_cq_yaml else "fail",
            severity="blocking",
            detail="cq.yaml present"
            if insp.has_cq_yaml
            else "cq.yaml missing at project root",
            suggested_fix=None
            if insp.has_cq_yaml
            else "run `pcq init-experiment`",
        )
    )

    # v2.5 (P2 #4): cq.yaml parse_errors gate — malformed YAML 명시 reporting.
    if insp.has_cq_yaml and insp.cq_yaml is not None and insp.cq_yaml.parse_error:
        report.add(
            ValidationCheck(
                id="cq_yaml_parseable",
                status="fail",
                severity="blocking",
                detail=f"cq.yaml parse error: {insp.cq_yaml.parse_error}",
                suggested_fix="fix YAML syntax — see ruamel.yaml diagnostics",
            )
        )

    if insp.has_cq_yaml and insp.cq_yaml is not None:
        cq_yaml = insp.cq_yaml
        # cmd 존재 여부
        cmd_ok = bool(cq_yaml.cmd)
        cmd_detail = f"cmd: {cq_yaml.cmd}" if cmd_ok else "cmd missing"
        report.add(
            ValidationCheck(
                id="cmd_defined",
                status="pass" if cmd_ok else "fail",
                severity="blocking",
                detail=cmd_detail,
            )
        )

    if strictness <= 0:
        return report

    if insp.has_cq_yaml and insp.cq_yaml is not None:
        cq_yaml = insp.cq_yaml

        # declared metrics
        report.add(
            ValidationCheck(
                id="metrics_declared",
                status="pass" if cq_yaml.declared_metrics else "warn",
                severity="warning" if not cq_yaml.declared_metrics else "info",
                detail=f"{len(cq_yaml.declared_metrics)} metrics declared"
                if cq_yaml.declared_metrics
                else "no declared metrics in cq.yaml",
            )
        )

        # artifacts
        report.add(
            ValidationCheck(
                id="artifacts_declared",
                status="pass" if cq_yaml.artifacts else "warn",
                severity="warning" if not cq_yaml.artifacts else "info",
                detail=f"{len(cq_yaml.artifacts)} artifact globs"
                if cq_yaml.artifacts
                else "no artifacts declared",
            )
        )

        # v1.15: structured cq.yaml gates — metric schema + inputs + monitor.
        for c in _validate_metric_schema(insp):
            report.add(c)
        for c in _validate_inputs(insp):
            report.add(c)
        m_check = _validate_monitor_in_metric_schema(insp)
        if m_check is not None:
            report.add(m_check)

    # v4.0: 모든 entrypoint 가 script 로 취급. script-aware gate 적용.
    _add_script_aware_gates(report, insp)

    if strictness <= 1:
        return report

    # post-run manifest evidence — 학습 후 산출물 검증 (있으면).
    post = _validate_manifest_post_run(Path(path).resolve())
    if post is not None:
        report.add(post)

    _add_project_reproducibility_gates(
        report, Path(path).resolve(), strictness
    )

    return report


def _validate_planset(planset: Any) -> list[ValidationCheck]:
    """ExperimentPlanSet 검증 — set + 모든 멤버 plan schema 검증 누적.

    v4.0: label-contract simulation 제거 (atom registry 부재).

    Returns:
        ValidationCheck 리스트.
    """
    checks: list[ValidationCheck] = []
    set_errors = planset.validate()
    if set_errors:
        for err in set_errors:
            checks.append(ValidationCheck(
                id="planset_validation",
                status="fail",
                severity="blocking",
                detail=err,
            ))
        return checks

    checks.append(ValidationCheck(
        id="planset_validation",
        status="pass",
        severity="info",
        detail=(
            f"planset {planset.id!r}: {len(planset.plans)} plans schema-valid"
        ),
    ))
    return checks
