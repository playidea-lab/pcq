"""validate_project 테스트."""
from __future__ import annotations

from pcq.agent import validate_project


def test_validate_examples_passes_or_warns():
    report = validate_project("examples")
    assert report.status in ("pass", "warn")
    ids = {c.id for c in report.checks}
    assert "cq_yaml_exists" in ids
    cq_yaml_check = next(c for c in report.checks if c.id == "cq_yaml_exists")
    assert cq_yaml_check.status == "pass"


def test_validate_no_cq_yaml_blocking_fail(tmp_path):
    report = validate_project(str(tmp_path))
    assert report.status == "fail"
    assert report.blocking_count >= 1
    cq_check = next(c for c in report.checks if c.id == "cq_yaml_exists")
    assert cq_check.status == "fail"
    assert cq_check.severity == "blocking"


def test_validate_to_dict_schema_version():
    report = validate_project("examples")
    d = report.to_dict()
    assert d["schema_version"] == 1
    assert d["strictness"] == 2
    assert d["strictness_name"] == "standard"
    assert "checks" in d
    assert "blocking_count" in d
    assert "warning_count" in d


def test_validate_strictness_zero_only_parse_contract(tmp_path):
    (tmp_path / "cq.yaml").write_text(
        "name: t\n"
        "cmd: uv run python train.py\n"
        "configs: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "train.py").write_text("print('no cq contract')\n")
    report = validate_project(tmp_path, strictness=0)
    ids = {c.id for c in report.checks}
    assert report.status == "pass"
    assert "cmd_defined" in ids
    assert "metrics_declared" not in ids
    assert "cq_config_called" not in ids


def test_validate_uses_cq_yaml_strictness_when_not_explicit(tmp_path):
    (tmp_path / "cq.yaml").write_text(
        "name: t\n"
        "cmd: uv run python train.py\n"
        "configs:\n"
        "  strictness: 3\n"
        "metrics: [eval_acc]\n"
        "artifacts: [output/]\n",
        encoding="utf-8",
    )
    report = validate_project(tmp_path)
    assert report.strictness == 3
    seed = next(c for c in report.checks if c.id == "seed_evidence")
    assert seed.status == "fail"
    assert seed.severity == "blocking"


def test_validate_strictness_four_requires_service_evidence(tmp_path):
    (tmp_path / "cq.yaml").write_text(
        "name: t\n"
        "cmd: uv run python train.py\n"
        "configs:\n"
        "  seed: 42\n"
        "metrics:\n"
        "  eval_acc:\n"
        "    mode: max\n"
        "artifacts: [output/]\n"
        "inputs:\n"
        "  dataset:\n"
        "    name: local\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("# fake lock\n", encoding="utf-8")
    report = validate_project(tmp_path, strictness=4)
    assert report.status == "fail"
    service_input = next(
        c for c in report.checks if c.id == "service_input_identity"
    )
    assert service_input.status == "fail"
    lineage = next(
        c for c in report.checks if c.id == "service_lineage_evidence"
    )
    assert lineage.status == "fail"
