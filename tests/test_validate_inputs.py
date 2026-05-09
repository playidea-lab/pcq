"""validate gates for v1.15 cq.yaml `inputs:` section."""
from __future__ import annotations

import textwrap

from pcq.agent import validate_project


def _make(tmp_path, cq_yaml_content):
    (tmp_path / "cq.yaml").write_text(
        textwrap.dedent(cq_yaml_content).lstrip(), encoding="utf-8"
    )
    (tmp_path / "train.py").write_text(
        "import pcq\ncq.config()\ncq.log(epoch=0)\n", encoding="utf-8"
    )


def test_validate_inputs_complete(tmp_path):
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics: [epoch]
        artifacts: [output/]
        inputs:
          dataset:
            name: dental
            version: v12
            uri: cq://datasets/dental/v12
    """)
    report = validate_project(tmp_path)
    assert any(
        c.id == "inputs_declared" and c.status == "pass"
        for c in report.checks
    )


def test_validate_inputs_missing_name_warns(tmp_path):
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics: [epoch]
        artifacts: [output/]
        inputs:
          dataset:
            version: v12
    """)
    report = validate_project(tmp_path)
    assert any(
        c.id == "input_identity" and c.status == "warn"
        for c in report.checks
    )


def test_validate_no_inputs_no_check(tmp_path):
    """inputs 섹션 없으면 input gates 모두 skip."""
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics: [epoch]
        artifacts: [output/]
    """)
    report = validate_project(tmp_path)
    input_ids = {"inputs_declared", "input_identity", "input_format"}
    assert not (
        {c.id for c in report.checks} & input_ids
    ), "inputs 미사용 시 input gate 발생 안 해야 함"


def test_validate_inputs_opaque_uri(tmp_path):
    """cq URI 는 opaque string — pcq 이 parse/fetch 안 함, 형태만 보존."""
    _make(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics: [epoch]
        artifacts: [output/]
        inputs:
          dataset:
            name: foo
            uri: cq://datasets/foo/v1?token=abc
    """)
    report = validate_project(tmp_path)
    # validate 는 URI 형식 검증 안 함 — 그냥 pass 여야.
    assert any(
        c.id == "inputs_declared" and c.status == "pass"
        for c in report.checks
    )
