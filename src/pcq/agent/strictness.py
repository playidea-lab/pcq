"""Shared validation strictness policy."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from pcq.agent.schema import ValidationCheck


STRICTNESS_NAMES: dict[int, str] = {
    0: "parse",
    1: "static",
    2: "standard",
    3: "reproducible",
    4: "service_grade",
}

STRICTNESS_EVIDENCE_MATRIX: dict[int, dict[str, Any]] = {
    0: {
        "name": "parse",
        "intended_use": "editor feedback and very early scaffolds",
        "pre_run": [
            "cq_yaml_exists",
            "cq_yaml_parseable",
            "cmd_defined",
        ],
        "post_run": [
            "manifest_present",
            "manifest_parseable",
            "metrics_present",
            "metrics_well_formed",
            "run_summary_present",
            "run_summary_parseable",
        ],
    },
    1: {
        "name": "static",
        "intended_use": "pre-run agent authoring",
        "pre_run": [
            "metrics_declared",
            "artifacts_declared",
            "cq_config_called",
            "cq_log_called",
            "standard_artifacts_helper",
        ],
        "post_run": [],
    },
    2: {
        "name": "standard",
        "intended_use": "default local and development validation",
        "pre_run": [
            "recipe_importable",
            "recipe_metrics_in_yaml",
            "loss_label_ignore_index",
            "model_dataset_channels",
            "optional_extras_available",
            "monitor_candidates_declared",
            "manifest_evidence",
        ],
        "post_run": [
            "manifest_evidence",
            "summary_metrics_consistent",
            "run_record_complete",
        ],
    },
    3: {
        "name": "reproducible",
        "intended_use": "CI and serious experiment records",
        "pre_run": [
            "seed_evidence",
            "lockfile_evidence",
            "inputs_evidence",
        ],
        "post_run": [
            "run_record_present",
            "run_finalized",
            "run_record_execution_identity",
            "source_reproducibility",
            "environment_reproducibility",
            "lockfile_evidence",
            "seed_evidence",
            "metrics_schema_evidence",
            "run_record_inputs_evidence",
        ],
    },
    4: {
        "name": "service_grade",
        "intended_use": "managed CQ runs and publishable comparisons",
        "pre_run": [
            "service_input_identity",
            "service_metric_schema",
            "service_lineage_evidence",
        ],
        "post_run": [
            "service_input_identity",
            "service_metric_schema",
            "service_hardware_evidence",
            "service_lineage_evidence",
        ],
    },
}


def normalize_strictness(value: Any, default: int = 2) -> int:
    """Return a clamped strictness level in the public 0..4 range."""
    if value is None:
        value = default
    try:
        level = int(value)
    except (TypeError, ValueError):
        level = default
    return max(0, min(4, level))


def strictness_name(level: int) -> str:
    """Human-readable stable name for a strictness level."""
    return STRICTNESS_NAMES.get(level, "standard")


def strictness_evidence_matrix() -> dict[int, dict[str, Any]]:
    """Return the full public strictness matrix as JSON-safe data."""
    return deepcopy(STRICTNESS_EVIDENCE_MATRIX)


def strictness_required_evidence(level: int) -> dict[str, list[str]]:
    """Return cumulative required evidence ids for ``level``.

    The returned ids are check ids used by `validate` / `validate-run` whenever
    that evidence is applicable. Some checks are style-dependent, so absence of
    a listed id in a report can still be valid when the inspected project does
    not use that style.
    """
    normalized = normalize_strictness(level)
    pre_run: list[str] = []
    post_run: list[str] = []
    for i in range(normalized + 1):
        item = STRICTNESS_EVIDENCE_MATRIX.get(i, {})
        pre_run.extend(item.get("pre_run", []))
        post_run.extend(item.get("post_run", []))
    return {
        "pre_run": _dedupe(pre_run),
        "post_run": _dedupe(post_run),
    }


def strictness_check(level: int) -> ValidationCheck:
    """Standard report check emitted by every validator."""
    name = strictness_name(level)
    return ValidationCheck(
        id="strictness_level",
        status="pass",
        severity="info",
        detail=f"strictness={level} ({name})",
        evidence={
            "level": level,
            "name": name,
            "required_evidence": strictness_required_evidence(level),
        },
    )


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
