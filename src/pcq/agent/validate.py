"""pcq.agent.validate — pre-run gates.

v1.7 구현 gate: Static Project Contract + Recipe Contract.
v1.8 추가: Label Contract (loss.ignore_index ↔ dataset.ignore_index 일치).
v1.9 추가:
  - model_dataset_channels: model.in_channels vs dataset.x[0] 일치
  - optional_extras_available: requires_extras best-effort import 검사
  - monitor_candidates_declared: SPEC.monitor_candidates 가 metrics 에 있나
v1.14 추가:
  - manifest_evidence (post-run): manifest entries 의 file 존재 + sha256 verify
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


# extras → 대표 모듈명 매핑. best-effort import 로 설치 여부 추정.
_EXTRAS_TO_MODULE: dict[str, str] = {
    "vision": "torchvision",
    "nlp": "transformers",
    "dist": "accelerate",
}


def _validate_label_contracts(recipe_dict: dict) -> ValidationCheck:
    """dataset 의 ignore_index 와 loss 의 ignore_index 일치 확인.

    recipe_dict 안의 dataset_train/dataset 와 loss 가 AtomRef 일 때만 검사 가능.
    legacy dict (callable/객체) 인 경우는 skip.
    """
    from pcq import registry as registry_pkg
    from pcq.registry.spec import AtomRef

    dataset_ignore: Any = None
    loss_ignore: Any = None

    # dataset_train 또는 dataset 키에서 ignore_index 조회
    for key in ("dataset_train", "dataset"):
        atom = recipe_dict.get(key)
        if isinstance(atom, AtomRef):
            try:
                spec = registry_pkg.datasets.get(atom.name)
                dataset_ignore = spec.label_contract.get("ignore_index")
            except ValueError:
                pass
            break

    # loss 에서 ignore_index 추출 (선언된 ignore_index_param 사용)
    atom = recipe_dict.get("loss")
    if isinstance(atom, AtomRef):
        try:
            spec = registry_pkg.losses.get(atom.name)
            ignore_param_name = spec.label_contract.get("ignore_index_param")
            if ignore_param_name:
                if ignore_param_name in atom.params:
                    loss_ignore = atom.params[ignore_param_name]
                elif ignore_param_name in spec.params:
                    loss_ignore = spec.params[ignore_param_name].default
        except ValueError:
            pass

    # 둘 다 알려진 경우만 mismatch 검사
    if (
        dataset_ignore is not None
        and loss_ignore is not None
        and dataset_ignore != loss_ignore
    ):
        return ValidationCheck(
            id="loss_label_ignore_index",
            status="fail",
            severity="blocking",
            detail=(
                f"dataset ignore_index={dataset_ignore} but "
                f"loss ignore_index={loss_ignore}"
            ),
            suggested_fix=(
                f"set loss.ignore_index={dataset_ignore} in recipe"
            ),
        )
    return ValidationCheck(
        id="loss_label_ignore_index",
        status="pass",
        severity="info",
        detail=(
            f"dataset_ignore={dataset_ignore}, loss_ignore={loss_ignore}"
        ),
    )


def _resolve_dataset_channels(
    dataset_atom: Any, dataset_spec: Any,
) -> int | None:
    """dataset 의 channel 수 추론. output_contract.x[0] + AtomRef.params 활용.

    output_contract.x[0] 가
      - 숫자 문자열("3") → int("3")
      - "channels" placeholder → AtomRef.params["channels"] 우선, 없으면 default
      - 그 외 → None (불명)
    """
    output_x = dataset_spec.output_contract.get("x", [])
    if not output_x:
        return None
    first = output_x[0]
    if isinstance(first, str):
        if first.isdigit():
            return int(first)
        if first == "channels":
            # AtomRef params 우선, 없으면 spec.params 의 default
            ch = dataset_atom.params.get("channels")
            if ch is None:
                pspec = dataset_spec.params.get("channels")
                if pspec is not None:
                    ch = pspec.default
            if isinstance(ch, int):
                return ch
    if isinstance(first, int):
        return first
    return None


def _resolve_model_in_channels(
    model_atom: Any, model_spec: Any,
) -> int | None:
    """model 의 in_channels 추론. AtomRef.params 우선, spec.params default 차선."""
    if "in_channels" in model_atom.params:
        v = model_atom.params["in_channels"]
        return v if isinstance(v, int) else None
    pspec = model_spec.params.get("in_channels")
    if pspec is not None and isinstance(pspec.default, int):
        return pspec.default
    # input_contract.x[1] 이 숫자 문자열이면 사용 (예: ["B","3","H","W"])
    x = model_spec.input_contract.get("x", [])
    if len(x) >= 2 and isinstance(x[1], str) and x[1].isdigit():
        return int(x[1])
    return None


def _validate_model_dataset_channels(
    recipe_dict: dict,
) -> ValidationCheck | None:
    """model.input_contract.in_channels 와 dataset.output_contract.x[0] 일치.

    예: small_cnn(in_channels=3) + mnist(1ch) → fail.
    AtomRef 가 아닌 경우(legacy 객체/callable) 또는 추론 불가 → None (skip).
    """
    from pcq import registry as registry_pkg
    from pcq.registry.spec import AtomRef

    model_atom = recipe_dict.get("model")
    dataset_atom = (
        recipe_dict.get("dataset_train") or recipe_dict.get("dataset")
    )
    if not isinstance(model_atom, AtomRef):
        return None
    if not isinstance(dataset_atom, AtomRef):
        return None
    try:
        model_spec = registry_pkg.models.get(model_atom.name)
        dataset_spec = registry_pkg.datasets.get(dataset_atom.name)
    except ValueError:
        return None

    model_in_ch = _resolve_model_in_channels(model_atom, model_spec)
    ds_channel = _resolve_dataset_channels(dataset_atom, dataset_spec)

    if model_in_ch is None or ds_channel is None:
        # 추론 불가 — silent skip
        return None
    if model_in_ch != ds_channel:
        return ValidationCheck(
            id="model_dataset_channels",
            status="fail",
            severity="blocking",
            detail=(
                f"model {model_atom.name} expects in_channels="
                f"{model_in_ch} but dataset {dataset_atom.name} outputs "
                f"{ds_channel} channels"
            ),
            suggested_fix=(
                f"set model in_channels={ds_channel} or change dataset"
            ),
        )
    return ValidationCheck(
        id="model_dataset_channels",
        status="pass",
        severity="info",
        detail=f"channels match ({model_in_ch})",
    )


def _validate_optional_extras(
    recipe_dict: dict,
) -> ValidationCheck:
    """recipe.requires_extras 에 명시된 extras 가 import 가능한지.

    extras 미설치 → warn (학습 시도 시 fail). 설치됨 → pass. 빈 list → pass.
    """
    extras_required = list(recipe_dict.get("requires_extras", []) or [])
    missing: list[str] = []
    for extra in extras_required:
        mod_name = _EXTRAS_TO_MODULE.get(extra)
        if mod_name is None:
            # 알 수 없는 extra — best-effort 로 모듈명을 extra 와 동일하게 시도
            mod_name = extra
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(f"{extra} (module: {mod_name})")
    if missing:
        return ValidationCheck(
            id="optional_extras_available",
            status="warn",
            severity="warning",
            detail=f"recipe requires extras not installed: {missing}",
            suggested_fix=(
                f"uv add pcq[{','.join(extras_required)}]"
            ),
        )
    if extras_required:
        return ValidationCheck(
            id="optional_extras_available",
            status="pass",
            severity="info",
            detail=f"all required extras available: {extras_required}",
        )
    return ValidationCheck(
        id="optional_extras_available",
        status="pass",
        severity="info",
        detail="no extras required",
    )


def _validate_monitor_in_metrics(
    spec_obj: Any, declared_metrics: list[str],
) -> ValidationCheck | None:
    """RecipeSpec.monitor_candidates 의 metric 이 declared metrics 안에 있나.

    spec_obj: RecipeSpec or None. declared_metrics: SPEC.metrics (또는 fallback).
    """
    candidates = list(getattr(spec_obj, "monitor_candidates", []) or [])
    if not candidates:
        return None
    declared = set(declared_metrics or [])
    if not declared:
        return None
    missing = [c["name"] for c in candidates if c.get("name") not in declared]
    if missing:
        return ValidationCheck(
            id="monitor_candidates_declared",
            status="warn",
            severity="warning",
            detail=(
                f"recipe monitor_candidates not in declared metrics: {missing}"
            ),
            suggested_fix=(
                "add to declared metrics or remove from monitor_candidates"
            ),
        )
    return ValidationCheck(
        id="monitor_candidates_declared",
        status="pass",
        severity="info",
        detail="all monitor_candidates declared",
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
    """Static + recipe-level 검증.

    strictness:
      0: cq.yaml/cmd parseability only.
      1: static cq.yaml + script contract checks, no recipe import/post-run.
      2: default; static + recipe metadata + post-run evidence when present.
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
            else "run `pcq init-experiment` (v1.8)",
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

    # — v1.13: script-style entrypoint 분기 —
    style = insp.entrypoint.kind if insp.entrypoint else None
    if style == "script":
        _add_script_aware_gates(report, insp)
        if strictness <= 1:
            return report
        # v1.14: post-run manifest evidence — 학습 후 산출물 검증 (있으면).
        post = _validate_manifest_post_run(Path(path).resolve())
        if post is not None:
            report.add(post)
        _add_project_reproducibility_gates(
            report, Path(path).resolve(), strictness
        )
        # script style 은 recipe gate 모두 skip — 일찍 return.
        return report

    if strictness <= 1:
        return report

    # — Gate 2: Recipe Contract (entrypoint preset 검출 시) —
    if insp.entrypoint and insp.entrypoint.preset:
        preset = insp.entrypoint.preset
        try:
            from pcq.agent import recipe_meta

            meta = recipe_meta(preset)
            report.add(
                ValidationCheck(
                    id="recipe_importable",
                    status="pass",
                    severity="info",
                    detail=f"recipe '{preset}' loaded, task={meta.get('task')}",
                )
            )
            # Gate 2b (v1.8+): RecipeSpec 기반 recipe 의 contract 검사들.
            try:
                from pcq.agent.schema import RecipeSpec
                from pcq.trainer import _import_recipe

                recipe_module = _import_recipe(preset).__module__
                import importlib

                mod = importlib.import_module(recipe_module)
                spec_obj = getattr(mod, "SPEC", None)
                if isinstance(spec_obj, RecipeSpec):
                    # SPEC.atoms 안의 AtomRef 들로 검사용 dict 합성
                    audit_dict: dict = dict(spec_obj.atoms)
                    audit_dict["requires_extras"] = list(
                        spec_obj.requires_extras
                    )
                    # v1.8: label ignore_index 일치
                    label_check = _validate_label_contracts(audit_dict)
                    report.add(label_check)
                    # v1.9: model-dataset channel 일치
                    ch_check = _validate_model_dataset_channels(audit_dict)
                    if ch_check is not None:
                        report.add(ch_check)
                    # v1.9: 필수 extras 설치 여부 (best-effort)
                    extras_check = _validate_optional_extras(audit_dict)
                    report.add(extras_check)
                    # v1.9: monitor_candidates 가 SPEC.metrics 에 포함되나
                    monitor_check = _validate_monitor_in_metrics(
                        spec_obj, list(spec_obj.metrics),
                    )
                    if monitor_check is not None:
                        report.add(monitor_check)
            except Exception:  # noqa: BLE001
                # spec-level gates 실패는 best-effort — 다른 검사는 계속 진행
                pass
            # recipe 의 declared_metrics 가 cq.yaml.metrics 에 있는지
            declared_set = set(meta.get("declared_metrics", []))
            if declared_set and insp.cq_yaml:
                yaml_set = set(insp.cq_yaml.declared_metrics)
                missing_in_yaml = declared_set - yaml_set
                if missing_in_yaml:
                    report.add(
                        ValidationCheck(
                            id="recipe_metrics_in_yaml",
                            status="warn",
                            severity="warning",
                            detail=(
                                "recipe declared metrics not in cq.yaml.metrics: "
                                f"{sorted(missing_in_yaml)}"
                            ),
                            suggested_fix="add to cq.yaml.metrics for full Hub coverage",
                        )
                    )
                else:
                    report.add(
                        ValidationCheck(
                            id="recipe_metrics_in_yaml",
                            status="pass",
                            severity="info",
                            detail="recipe metrics fully declared in cq.yaml",
                        )
                    )
        except Exception as e:
            report.add(
                ValidationCheck(
                    id="recipe_importable",
                    status="fail",
                    severity="blocking",
                    detail=f"recipe '{preset}' import failed: {e}",
                )
            )

    # — Gate 3: Optional dependency hint (informational) —
    extras: set[str] = set()
    for r in insp.recipes:
        for e in r.requires_extras:
            extras.add(e)
    if extras:
        report.add(
            ValidationCheck(
                id="optional_extras",
                status="pass",
                severity="info",
                detail=f"recipes reference extras: {sorted(extras)}",
                suggested_fix=f"install with `uv add pcq[{','.join(sorted(extras))}]`",
            )
        )

    # — v1.14: Post-run gate — manifest evidence 검증 (output 디렉토리 존재 시) —
    post = _validate_manifest_post_run(Path(path).resolve())
    if post is not None:
        report.add(post)

    _add_project_reproducibility_gates(
        report, Path(path).resolve(), strictness
    )

    return report


def _validate_planset(planset: Any) -> list[ValidationCheck]:
    """ExperimentPlanSet 검증 — set + 모든 멤버 plan schema 검증 누적.

    label-contract simulation 등 plan 별 추가 gate 도 동일하게 적용한다.
    set 자체가 schema fail 이면 single ValidationCheck 로 묶어 반환.

    Returns:
        ValidationCheck 리스트. 각 멤버 plan 의 label-contract 결과 포함.
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
    # 각 멤버 plan label-contract simulation
    for plan in planset.plans:
        member_checks = _validate_plan_label_contracts(plan)
        checks.extend(member_checks)
    return checks


def _validate_plan_label_contracts(plan: Any) -> list[ValidationCheck]:
    """v2.3: plan-only context 에서 label_contract gate 시뮬레이션.

    plan 의 set_atom 변경 사항을 base preset 의 RecipeSpec.atoms 위에 적용한 후
    `_validate_label_contracts` 를 돌려, 실행 전에 ignore_index 충돌을 감지한다.

    예: base preset (voc_unet) 의 dataset 이 ignore_index=-1 인데 plan 이
    `set_atom loss cross_entropy ignore_index=-100` 으로 바꾸면 → fail.

    Args:
        plan: ExperimentPlan instance.

    Returns:
        ValidationCheck 리스트. base preset 미존재 / RecipeSpec 부재 /
        import 실패 등 시뮬레이션 불가능한 경우 빈 리스트 (silent skip) —
        다른 plan_validation gate 가 fail 을 별도로 보고하므로 중복 noise 회피.
    """
    checks: list[ValidationCheck] = []
    base_preset = plan.base.get("preset") if plan.base else None
    if not base_preset:
        return checks

    # base preset 의 RecipeSpec.atoms 조회 — vision extras 미설치 환경 등에서
    # import 실패하면 silent skip (다른 gate 에서 import 실패를 별도 보고).
    try:
        from pcq.agent.schema import RecipeSpec
        from pcq.trainer import _import_recipe
        import importlib

        fn = _import_recipe(base_preset)
        mod = importlib.import_module(fn.__module__)
        spec = getattr(mod, "SPEC", None)
        if not isinstance(spec, RecipeSpec):
            return checks
    except Exception:  # noqa: BLE001 — silent skip on import failure
        return checks

    from pcq.registry.spec import AtomRef

    base_atoms = spec.atoms or {}
    sim_atoms: dict[str, AtomRef] = {}
    for k, v in base_atoms.items():
        if isinstance(v, AtomRef):
            sim_atoms[k] = AtomRef(
                kind=v.kind, name=v.name, params=dict(v.params)
            )

    # plan changes 적용 — set_atom 만 (set_config 는 cfg 영역, label contract 무관).
    for c in plan.changes:
        if c.op != "set_atom":
            continue
        atom_key = c.atom or ""
        kind = c._infer_kind()
        if kind == "unknown":
            continue
        if c.merge:
            existing = sim_atoms.get(atom_key)
            if existing is not None:
                merged_params = {**existing.params, **(c.params or {})}
                new_name = c.name if c.name else existing.name
                sim_atoms[atom_key] = AtomRef(
                    kind=kind, name=new_name, params=merged_params
                )
            elif c.name:
                # base 에 없으면 새로 생성 (merge 인데 base 부재).
                sim_atoms[atom_key] = AtomRef(
                    kind=kind, name=c.name, params=dict(c.params or {})
                )
        else:
            if c.name:
                sim_atoms[atom_key] = AtomRef(
                    kind=kind, name=c.name, params=dict(c.params or {})
                )

    # 시뮬된 atoms 로 label contract 검사.
    label_check = _validate_label_contracts(dict(sim_atoms))
    if label_check.status == "fail":
        checks.append(
            ValidationCheck(
                id="plan_label_contract",
                status="fail",
                severity="blocking",
                detail=f"plan {plan.id}: {label_check.detail}",
                suggested_fix=label_check.suggested_fix,
            )
        )
    elif label_check.status == "pass":
        checks.append(
            ValidationCheck(
                id="plan_label_contract",
                status="pass",
                severity="info",
                detail=(
                    f"plan {plan.id}: simulated atoms compatible "
                    f"({label_check.detail})"
                ),
            )
        )
    return checks
