"""validate gates for v1.15 structured cq.yaml — metric_schema + monitor."""
from __future__ import annotations

import textwrap

from pcq.agent import validate_project


def _make(tmp_path, cq_yaml_content):
    """tmp_path 에 minimal pcq 프로젝트 셋업."""
    (tmp_path / "cq.yaml").write_text(
        textwrap.dedent(cq_yaml_content).lstrip(), encoding="utf-8"
    )
    (tmp_path / "train.py").write_text(
        "import pcq\ncq.config()\ncq.log(epoch=0)\n", encoding="utf-8"
    )


def test_validate_metric_schema_valid(tmp_path):
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          eval_iou: {mode: max, split: val}
          eval_loss: {mode: min}
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "metric_schema_complete"), None
    )
    assert check is not None
    assert check.status == "pass"


def test_validate_metric_missing_mode_warns(tmp_path):
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          eval_iou: {split: val}
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    assert any(
        c.id == "metric_schema_mode" and c.status == "warn"
        for c in report.checks
    )


def test_validate_metric_invalid_mode_fails(tmp_path):
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          eval_iou: {mode: bogus}
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    assert any(
        c.id == "metric_schema_mode_value" and c.status == "fail"
        for c in report.checks
    )
    # blocking → status=fail
    assert report.status == "fail"


def test_validate_monitor_must_be_in_metric_schema(tmp_path):
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs:
          monitor: eval_unknown
          mode: max
        metrics:
          eval_iou: {mode: max}
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    assert any(
        c.id == "monitor_in_metric_schema" and c.status == "warn"
        for c in report.checks
    )


def test_validate_monitor_mode_consistency(tmp_path):
    """cfg.mode 와 metric schema 의 mode 불일치 → warn."""
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs:
          monitor: eval_iou
          mode: min
        metrics:
          eval_iou: {mode: max}
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    assert any(
        c.id == "monitor_mode_consistency" and c.status == "warn"
        for c in report.checks
    )


def test_validate_monitor_in_schema_pass(tmp_path):
    """monitor 가 schema 에 있고 mode 일치 → pass."""
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs:
          monitor: eval_iou
          mode: max
        metrics:
          eval_iou: {mode: max}
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "monitor_in_metric_schema"), None
    )
    assert check is not None
    assert check.status == "pass"


def test_validate_legacy_list_metrics_no_schema_check(tmp_path):
    """list-style metrics 일 때 metric_schema gate 들 모두 skip (호환)."""
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          - epoch
          - eval_acc
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    schema_ids = {
        "metric_schema_complete",
        "metric_schema_mode",
        "metric_schema_mode_value",
        "monitor_in_metric_schema",
        "monitor_mode_consistency",
    }
    found = {c.id for c in report.checks} & schema_ids
    assert not found, (
        f"list-style 인 경우 dict-style gate 가 발생하면 안 됨: {found}"
    )
