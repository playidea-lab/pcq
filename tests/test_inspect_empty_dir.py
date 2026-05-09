"""v2.5.0 P2 cleanup #5 — inspect/validate handle empty output_dir explicitly.

빈 output_dir 에서 inspect/validate 가 명확한 메시지를 출력 (status='empty').
script gate 등이 기본 가정한 'output' 미존재를 안전하게 처리.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pcq.agent.inspect import inspect_project


def _write_yaml(tmp: Path, body: str) -> None:
    (tmp / "cq.yaml").write_text(body)


def test_inspect_empty_output_dir_status(tmp_path):
    """output_dir 존재하지만 어떤 artifact 도 없으면 status='empty'."""
    _write_yaml(tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: empty_out\n")
    (tmp_path / "empty_out").mkdir()  # empty dir, no artifact
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.status == "empty"
    assert insp.outputs.has_manifest is False
    assert insp.outputs.has_metrics is False
    assert insp.outputs.has_run_record is False


def test_inspect_partial_output_dir_status(tmp_path):
    """일부 artifact 있으면 status='partial'."""
    _write_yaml(tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: partial_out\n")
    out = tmp_path / "partial_out"
    out.mkdir()
    (out / "metrics.json").write_text('{"history": []}')
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.status == "partial"
    assert insp.outputs.has_metrics is True


def test_inspect_complete_output_dir_status(tmp_path):
    """manifest+metrics+run_record 모두 있으면 status='complete'."""
    _write_yaml(tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: full_out\n")
    out = tmp_path / "full_out"
    out.mkdir()
    (out / "metrics.json").write_text('{"history": []}')
    (out / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    (out / "run_record.json").write_text('{"run": {"id": "r"}}')
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.status == "complete"


def test_inspect_no_output_dir_no_status(tmp_path):
    """output_dir 자체가 없으면 status=None."""
    _write_yaml(tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: nope\n")
    # 'nope' dir 안 만듦
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    # nope 도 없고 legacy 'output' 도 없음
    assert insp.outputs.status is None
    assert insp.outputs.output_dir is None


def test_cli_inspect_empty_dir_returns_status(tmp_path):
    """CLI inspect JSON 이 status='empty' 출력."""
    _write_yaml(tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: empty_out\n")
    (tmp_path / "empty_out").mkdir()
    # script train.py 추가 — entrypoint 스캔이 실패하지 않게
    (tmp_path / "train.py").write_text("import pcq\ncq.config()\n")

    result = subprocess.run(
        [sys.executable, "-m", "pcq.cli", "inspect", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["outputs"]["status"] == "empty"


def test_inspect_does_not_mkdir_output_dir(tmp_path):
    """v2.5: read-only — inspect 호출 후에도 output_dir 자동 생성 안 됨."""
    _write_yaml(
        tmp_path, "name: t\ncmd: x\nconfigs:\n  output_dir: not_yet\n"
    )
    inspect_project(tmp_path)
    assert not (tmp_path / "not_yet").exists(), (
        "inspect must NOT create output_dir (read-only)"
    )
