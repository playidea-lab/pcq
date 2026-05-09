"""Strictness evidence matrix contract tests."""
from __future__ import annotations

import json
from pathlib import Path

from pcq.agent import (
    strictness_evidence_matrix,
    strictness_required_evidence,
    validate_project,
)
from pcq.agent.validate_run import validate_run


def _write_minimal_output(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0, "eval_acc": 0.5}]}),
        encoding="utf-8",
    )
    (out / "run_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "status": "completed",
            "monitor": {"name": "eval_acc", "mode": "max"},
            "best": {"epoch": 0, "metrics": {"eval_acc": 0.5}},
            "last": {"epoch": 0, "metrics": {"eval_acc": 0.5}},
        }),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "files": [{"path": "metrics.json", "kind": "metrics"}],
        }),
        encoding="utf-8",
    )


def test_strictness_matrix_is_json_serializable_and_cumulative():
    matrix = strictness_evidence_matrix()
    json.dumps(matrix)

    level3 = strictness_required_evidence(3)
    assert "cmd_defined" in level3["pre_run"]
    assert "seed_evidence" in level3["pre_run"]
    assert "source_reproducibility" in level3["post_run"]
    assert "service_input_identity" not in level3["post_run"]

    level4 = strictness_required_evidence(4)
    assert "service_input_identity" in level4["pre_run"]
    assert "service_hardware_evidence" in level4["post_run"]


def test_validate_project_strictness_check_exposes_required_evidence(tmp_path):
    (tmp_path / "cq.yaml").write_text(
        "name: strictness-contract\n"
        "cmd: uv run python train.py\n"
        "configs:\n"
        "  strictness: 3\n"
        "metrics: [eval_acc]\n"
        "artifacts: [output/]\n",
        encoding="utf-8",
    )
    report = validate_project(tmp_path, strictness=3)
    strict = next(c for c in report.checks if c.id == "strictness_level")

    required = strict.evidence["required_evidence"]
    assert strict.evidence["level"] == 3
    assert strict.evidence["name"] == "reproducible"
    assert "seed_evidence" in required["pre_run"]
    assert "run_record_execution_identity" in required["post_run"]


def test_validate_run_strictness_check_exposes_required_evidence(tmp_path):
    _write_minimal_output(tmp_path)
    report = validate_run(tmp_path, strictness=4)
    strict = next(c for c in report.checks if c.id == "strictness_level")

    required = strict.evidence["required_evidence"]
    assert strict.evidence["level"] == 4
    assert strict.evidence["name"] == "service_grade"
    assert "service_metric_schema" in required["pre_run"]
    assert "service_lineage_evidence" in required["post_run"]
