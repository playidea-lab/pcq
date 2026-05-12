"""Stable JSON output contracts for agent-facing pcq surfaces.

The contracts here intentionally describe the minimum stable surface, not every
field a command may emit. Within ``schema_version == 1`` these required fields
are additive-only: callers may depend on them, and pcq may add more fields.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


JSON_CONTRACT_VERSION = 1


JSON_CONTRACTS: dict[str, dict[str, Any]] = {
    "pcq.run.envelope": {
        "schema_version": JSON_CONTRACT_VERSION,
        "description": "pcq run --json envelope for executed or config-only runs",
        "additive_only": True,
        "required": {
            "schema_version": "int",
            "status": "string",
            "project_root": "string",
            "runtime_cfg_path": "string",
            "cmd": "string",
        },
        "optional": {
            "exit_code": "int",
            "stdout_path": "string",
            "stderr_path": "string",
            "stdout_tail": "string",
            "stderr_tail": "string",
            "stdout_tail_truncated": "boolean",
            "stderr_tail_truncated": "boolean",
            "error": "string",
            # attribution 패스스루 필드 (T-PCQ-ATTR-3 additive)
            # describe-run 스키마의 attribution 객체와 동일한 형태로 전달됨
            "attribution": "object",
        },
        "enums": {
            "status": ["completed", "failed", "config_only", "error"],
        },
    },
    "pcq.run.event": {
        "schema_version": JSON_CONTRACT_VERSION,
        "description": "pcq run --jsonl newline-delimited event",
        "additive_only": True,
        "required": {
            "schema_version": "int",
            "seq": "int",
            "time": "string",
            "event": "string",
        },
        "optional": {
            "status": "string",
            "cmd": "string",
            "project_root": "string",
            "runtime_cfg_path": "string",
            "stream": "string",
            "text": "string",
            "raw": "string",
            "metrics": "object",
            "exit_code": "int",
            "stdout_path": "string",
            "stderr_path": "string",
            "events_path": "string",
            "error": "string",
        },
        "enums": {
            "event": [
                "run.started",
                "run.completed",
                "run.failed",
                "run.error",
                "run.config_only",
                "stdout",
                "stderr",
                "metric",
            ],
        },
    },
    "pcq.describe_run.record": {
        "schema_version": JSON_CONTRACT_VERSION,
        "description": "pcq describe-run --json for a readable RunRecord",
        "additive_only": True,
        "required": {
            "schema_version": "int",
            "status": "string",
            "output_dir": "string",
            "epochs_completed": "int",
            "partial": "boolean",
            "dirty": "boolean",
            "validation_status": "string",
            "decision_facts": "object",
        },
        "optional": {
            "run_id": "string",
            "name": "string",
            "cmd": "string",
            "target_metric": "string",
            "mode": "string",
            "best": "object",
            "best_value": "number",
            "best_epoch": "int",
            "last": "object",
            "last_value": "number",
            "last_epoch": "int",
            "duration_seconds": "number",
            "parent_run_id": "string",
            "parent_run_path": "string",
            "git_sha": "string",
            "python": "string",
            "platform": "string",
            "metrics_declared": "array",
            "artifacts": "array",
            "artifacts_summary": "object",
            "validation_report_path": "string",
            "reproducibility_evidence": "object",
            "failure": "object",
            # attribution 중첩 객체 (Phase 1 — 서명 제외)
            "attribution": "object",
            # attribution 플랫 표면 — 에이전트가 쿼리하기 쉽도록 최상위 노출
            "attribution_author_kind": "string",
            "attribution_committer_kind": "string",
            "attribution_operator": "string",
            "attribution_session_id": "string",
        },
        # attribution 필드의 상세 JSON Schema — property_overrides가 단순 타입 표기를 덮어씀
        "property_overrides": {
            "attribution": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "schema_version": {"type": "integer", "const": 1},
                    "author": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"type": "string", "enum": ["human", "agent"]},
                            "id": {"type": "string"},
                            "persona_id": {"type": ["string", "null"]},
                        },
                        "required": ["kind", "id"],
                    },
                    "committer": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"type": "string", "enum": ["human", "agent"]},
                            "id": {"type": "string"},
                            "persona_id": {"type": ["string", "null"]},
                        },
                        "required": ["kind", "id"],
                    },
                    "operator": {"type": "string"},
                    "session_id": {"type": ["string", "null"]},
                },
                "required": ["schema_version", "author", "committer", "operator"],
            },
            "attribution_author_kind": {"type": ["string", "null"]},
            "attribution_committer_kind": {"type": ["string", "null"]},
            "attribution_operator": {"type": ["string", "null"]},
            "attribution_session_id": {"type": ["string", "null"]},
        },
        "nested_required": {
            "decision_facts": {
                "run_completed": "boolean",
                "run_failed": "boolean",
                "run_partial": "boolean",
                "validation_passed": "boolean",
                "validation_failed": "boolean",
                "has_failure": "boolean",
                "has_target_metric": "boolean",
                "has_best": "boolean",
                "has_last": "boolean",
                "has_parent": "boolean",
                "artifact_count": "int",
                "metric_count": "int",
                "input_count": "int",
                "dirty_source": "boolean",
                "has_lockfile": "boolean",
                "has_cq_yaml_hash": "boolean",
            },
        },
    },
    "pcq.compare_runs.diff": {
        "schema_version": JSON_CONTRACT_VERSION,
        "description": "pcq compare-runs --json for two readable RunRecords",
        "additive_only": True,
        "required": {
            "schema_version": "int",
            "a_run_id": "string",
            "b_run_id": "string",
            "target_metric": "string",
            "a_target_metric": "string",
            "b_target_metric": "string",
            "mode": "string",
            "metric_direction": "string",
            "best": "object",
            "last": "object",
            "validation": "object",
            "artifacts": "object",
            "decision_facts": "object",
            "a_status": "string",
            "b_status": "string",
            "a_is_ancestor_of_b": "boolean",
            "b_is_ancestor_of_a": "boolean",
        },
        "optional": {
            "metric_delta": "number",
            "last_metric_delta": "number",
            "last_metric_direction": "string",
            "epochs_a": "int",
            "epochs_b": "int",
            "best_epoch_a": "int",
            "best_epoch_b": "int",
            "config_changes": "array",
            "input_changes": "array",
            "failure": "object",
            "source": "object",
            "notes": "array",
            # attribution diff 필드 (T-PCQ-ATTR-3 additive)
            "attribution_diff": "object",
            "attribution_author_changed": "boolean",
            "attribution_committer_changed": "boolean",
            "attribution_operator_changed": "boolean",
        },
        "nested_required": {
            "decision_facts": {
                "comparable": "boolean",
                "same_target_metric": "boolean",
                "best_improved": "boolean",
                "best_regressed": "boolean",
                "best_tied": "boolean",
                "last_improved": "boolean",
                "last_regressed": "boolean",
                "last_tied": "boolean",
                "candidate_completed": "boolean",
                "candidate_failed": "boolean",
                "candidate_validated": "boolean",
                "candidate_validation_failed": "boolean",
                "config_changed": "boolean",
                "input_changed": "boolean",
                "artifact_count_changed": "boolean",
                "source_changed": "boolean",
                "same_cq_yaml": "boolean",
                "has_lineage_relation": "boolean",
            },
        },
        "enums": {
            "mode": ["min", "max"],
            "metric_direction": ["improved", "regressed", "tied", "incomparable"],
        },
    },
    "pcq.validation_report": {
        "schema_version": JSON_CONTRACT_VERSION,
        "description": "pcq validate/validate-run --json ValidationReport",
        "additive_only": True,
        "required": {
            "schema_version": "int",
            "status": "string",
            "checks": "array",
            "blocking_count": "int",
            "warning_count": "int",
        },
        "optional": {
            "strictness": "int",
            "strictness_name": "string",
            # operator_required: L2+ strictness에서 attribution.operator 필수 게이트
            # 미설정 시 PII_PATTERN_DETECTED 경고 발생 가능 (R10 additive)
            "operator_required": "boolean",
            # warning_codes: 이번 보고서에서 발생한 경고 코드 목록
            # 포함 가능 코드: "PII_PATTERN_DETECTED"
            "warning_codes": "array",
        },
        "array_item_required": {
            "checks": {
                "id": "string",
                "status": "string",
                "severity": "string",
                "detail": "string",
            },
        },
        "enums": {
            "status": ["pass", "warn", "fail"],
            # warning_codes의 알려진 값 — 추가 코드는 하위 호환으로 허용
            # "PII_PATTERN_DETECTED": attribution.operator에 PII 패턴 감지
        },
    },
    "pcq.agent_install.result": {
        "schema_version": JSON_CONTRACT_VERSION,
        "description": "pcq agent install --json result",
        "additive_only": True,
        "required": {
            "schema_version": "int",
            "project_root": "string",
            "target": "string",
            "dry_run": "boolean",
            "force": "boolean",
            "files_created": "array",
            "files_updated": "array",
            "files_skipped": "array",
            "warnings": "array",
            "operations": "array",
        },
        "array_item_required": {
            "operations": {
                "path": "string",
                "action": "string",
                "kind": "string",
                "agent": "string",
                "reason": "string",
            },
        },
        "enums": {
            "target": ["codex", "claude", "both"],
        },
    },
    "pcq.agent_status.result": {
        "schema_version": JSON_CONTRACT_VERSION,
        "description": "pcq agent status --json read-only result",
        "additive_only": True,
        "required": {
            "schema_version": "int",
            "project_root": "string",
            "target": "string",
            "status": "string",
            "assets": "array",
            "repair_command": "string",
        },
        "array_item_required": {
            "assets": {
                "path": "string",
                "agent": "string",
                "kind": "string",
                "status": "string",
                "detail": "string",
                "suggested_fix": "string",
            },
        },
        "enums": {
            "target": ["codex", "claude", "both"],
            "status": [
                "missing",
                "installed",
                "partial",
                "stale",
                "unmanaged",
                "divergent",
            ],
        },
    },
}


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
}


def get_json_contracts() -> dict[str, dict[str, Any]]:
    """Return a JSON-serializable copy of the public contract registry."""
    return deepcopy(JSON_CONTRACTS)


def validate_json_contract(name: str, payload: Any) -> list[str]:
    """Validate ``payload`` against a named minimum JSON contract.

    This is deliberately a tiny stdlib validator rather than a full JSON Schema
    engine. It exists to freeze pcq's public agent surface without adding a
    runtime dependency.
    """
    spec = JSON_CONTRACTS.get(name)
    if spec is None:
        return [f"unknown contract: {name}"]
    if not isinstance(payload, dict):
        return [f"{name}: payload must be object"]

    errors: list[str] = []
    _validate_fields(errors, payload, spec.get("required", {}), required=True)
    _validate_fields(errors, payload, spec.get("optional", {}), required=False)
    _validate_enums(errors, payload, spec.get("enums", {}))
    _validate_nested_required(errors, payload, spec.get("nested_required", {}))
    _validate_array_items(errors, payload, spec.get("array_item_required", {}))
    return errors


def _validate_fields(
    errors: list[str],
    payload: dict[str, Any],
    fields: dict[str, str],
    *,
    required: bool,
    prefix: str = "",
) -> None:
    for key, type_name in fields.items():
        path = f"{prefix}{key}"
        if key not in payload:
            if required:
                errors.append(f"missing required field: {path}")
            continue
        if not _matches_type(payload[key], type_name):
            errors.append(
                f"{path}: expected {type_name}, got "
                f"{type(payload[key]).__name__}"
            )


def _validate_enums(
    errors: list[str],
    payload: dict[str, Any],
    enums: dict[str, list[Any]],
    *,
    prefix: str = "",
) -> None:
    for key, allowed in enums.items():
        if key in payload and payload[key] not in allowed:
            errors.append(
                f"{prefix}{key}: expected one of {allowed}, got {payload[key]!r}"
            )


def _validate_nested_required(
    errors: list[str],
    payload: dict[str, Any],
    nested: dict[str, dict[str, str]],
) -> None:
    for key, fields in nested.items():
        value = payload.get(key)
        if not isinstance(value, dict):
            errors.append(f"{key}: expected object for nested contract")
            continue
        _validate_fields(errors, value, fields, required=True, prefix=f"{key}.")


def _validate_array_items(
    errors: list[str],
    payload: dict[str, Any],
    arrays: dict[str, dict[str, str]],
) -> None:
    for key, fields in arrays.items():
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append(f"{key}: expected array")
            continue
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"{key}[{idx}]: expected object")
                continue
            _validate_fields(
                errors, item, fields, required=True, prefix=f"{key}[{idx}]."
            )


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "any":
        return True
    if type_name == "int":
        return type(value) is int
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = _TYPE_MAP.get(type_name)
    if expected is None:
        return False
    return isinstance(value, expected)
