"""inspect_project Python API 테스트 (CLI 통하지 않는 경로)."""
from __future__ import annotations

import json

from pcq.agent import inspect_project


def test_inspect_examples_returns_project_type_pcq():
    insp = inspect_project("examples")
    assert insp.project_type == "pcq"
    assert insp.has_cq_yaml is True
    assert insp.cq_yaml is not None
    assert insp.cq_yaml.name == "cq-python-smoke"
    assert "epoch" in insp.cq_yaml.declared_metrics


def test_inspect_detects_script_entrypoint():
    """v4.0: examples/train.py 는 contract script (Trainer 제거)."""
    insp = inspect_project("examples")
    assert insp.entrypoint is not None
    assert insp.entrypoint.kind == "script"
    assert insp.entrypoint.preset is None


def test_inspect_to_dict_serializable():
    insp = inspect_project("examples")
    d = insp.to_dict()
    s = json.dumps(d)
    assert json.loads(s) == d


def test_inspect_schema_version_present():
    insp = inspect_project("examples")
    d = insp.to_dict()
    assert d["schema_version"] == 1


def test_inspect_nonexistent_path_records_error(tmp_path):
    bad = tmp_path / "missing"
    insp = inspect_project(str(bad))
    assert any("does not exist" in e for e in insp.errors)


def test_inspect_artifacts_parsed():
    insp = inspect_project("examples")
    assert insp.cq_yaml is not None
    assert "output/" in insp.cq_yaml.artifacts
