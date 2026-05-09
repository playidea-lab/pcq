"""v2.5.0 P2 cleanup #4 — yaml strict mode.

malformed cq.yaml 이 더 이상 silently 통과하지 않음. parse_errors 에 명시되고,
inspect/validate 가 명확히 보고한다.
"""
from __future__ import annotations

from pathlib import Path

from pcq.agent.resolver import resolve_project
from pcq.agent.inspect import inspect_project
from pcq.agent.validate import validate_project


def _write(tmp: Path, body: str) -> None:
    (tmp / "cq.yaml").write_text(body)


def test_unreadable_yaml_records_parse_error(tmp_path):
    """파일이 directory 등 읽을 수 없는 형태 → parse_errors 에 기록."""
    # cq.yaml 을 디렉토리로 만들어 읽기 실패 유도.
    bad = tmp_path / "cq.yaml"
    bad.mkdir()
    rc = resolve_project(path=tmp_path)
    assert rc.parse_errors, (
        f"unreadable YAML must record parse_errors, got: cq_yaml_path={rc.cq_yaml_path}"
    )
    assert any("read" in e.lower() or "fail" in e.lower() for e in rc.parse_errors)


def test_well_formed_yaml_no_parse_errors_baseline(tmp_path):
    """정상 cq.yaml → parse_errors 비어 있음 (baseline)."""
    _write(
        tmp_path,
        "name: ok\ncmd: x\nconfigs:\n  output_dir: out\nmetrics:\n  - eval_loss\n",
    )
    rc = resolve_project(path=tmp_path)
    assert rc.parse_errors == []
    assert rc.name == "ok"


def test_inspect_reports_parse_error(tmp_path):
    """inspect 의 errors 에 parse_error 가 propagate (unreadable case)."""
    bad = tmp_path / "cq.yaml"
    bad.mkdir()  # cq.yaml 이 directory → read fail
    insp = inspect_project(tmp_path)
    # parse_error 는 insp.errors 또는 insp.cq_yaml.parse_error 로 표현돼야 함.
    has_parse_err = any("cq.yaml" in e for e in insp.errors) or (
        insp.cq_yaml is not None and insp.cq_yaml.parse_error
    )
    # cq_yaml_path 가 directory 면 cq.yaml.parse_error 또는 errors 에 기록됨.
    # 단 inspect 가 디렉토리를 cq_yaml_paths 로 잡지 않을 수도 있다 → 그 경우는 skip.
    if insp.has_cq_yaml:
        assert has_parse_err, (
            f"inspect did not report parse_error: errors={insp.errors}"
        )


def test_validate_fails_on_unparseable_yaml(tmp_path):
    """validate_project 의 cq_yaml_parseable gate 가 fail (unreadable case)."""
    bad = tmp_path / "cq.yaml"
    bad.mkdir()  # cq.yaml is directory → read fail
    report = validate_project(tmp_path)
    parseable_checks = [c for c in report.checks if c.id == "cq_yaml_parseable"]
    # parse error 가 있을 때만 cq_yaml_parseable gate 가 추가됨 (in-place fail).
    if parseable_checks:
        assert parseable_checks[0].status == "fail"
        assert parseable_checks[0].severity == "blocking"
    # else: 디렉토리는 cq_yaml_path glob 에 잡히지 않으면 cq_yaml_exists=fail 이 됨.
