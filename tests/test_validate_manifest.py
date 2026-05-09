"""validate_project — manifest evidence post-run gate (v1.14)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pcq.agent import validate_project


def _make_project(
    tmp_path: Path,
    manifest: dict,
    files: dict[str, bytes] | None = None,
) -> None:
    """post-run gate 검증용 minimal project layout."""
    (tmp_path / "cq.yaml").write_text(
        "name: t\n"
        "cmd: uv run python train.py\n"
        "configs: {}\n"
        "metrics: []\n"
        "artifacts:\n  - output/\n"
    )
    (tmp_path / "train.py").write_text(
        "import pcq\ncq.config()\ncq.log(epoch=0)\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(manifest))
    if files:
        for name, content in files.items():
            (output / name).write_bytes(content)


def test_manifest_v2_pass(tmp_path):
    """v2 manifest + 실제 file 일치 → manifest_evidence pass."""
    content = b"hello world"
    sha = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": 2,
        "files": [
            {
                "path": "model.pt",
                "kind": "model",
                "sha256": sha,
                "size_bytes": len(content),
            }
        ],
    }
    _make_project(tmp_path, manifest, files={"model.pt": content})
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "manifest_evidence"), None
    )
    assert check is not None
    assert check.status == "pass"
    assert "v2" in check.detail


def test_manifest_v2_sha_mismatch_fails(tmp_path):
    """v2 manifest 의 sha256 가 실제 파일과 불일치 → blocking fail."""
    content = b"hello"
    manifest = {
        "schema_version": 2,
        "files": [
            {
                "path": "model.pt",
                "kind": "model",
                "sha256": "0" * 64,
                "size_bytes": 5,
            }
        ],
    }
    _make_project(tmp_path, manifest, files={"model.pt": content})
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "manifest_evidence"), None
    )
    assert check is not None
    assert check.status == "fail"
    assert check.severity == "blocking"
    assert "mismatch" in check.detail


def test_manifest_v2_missing_file_fails(tmp_path):
    """v2 manifest entry 가 가리키는 file 이 없으면 blocking fail."""
    manifest = {
        "schema_version": 2,
        "files": [
            {
                "path": "ghost.pt",
                "kind": "model",
                "sha256": "x" * 64,
                "size_bytes": 0,
            }
        ],
    }
    _make_project(tmp_path, manifest)  # no actual file
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "manifest_evidence"), None
    )
    assert check is not None
    assert check.status == "fail"
    assert "missing" in check.detail


def test_manifest_v1_legacy_still_pass(tmp_path):
    """schema v1 legacy manifest — sha 검증 skip, file 존재만 확인."""
    content = b"x"
    manifest = {
        "schema_version": 1,
        "files": [{"path": "model.pt", "kind": "model"}],
    }
    _make_project(tmp_path, manifest, files={"model.pt": content})
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "manifest_evidence"), None
    )
    assert check is not None
    assert check.status == "pass"
    assert "v1" in check.detail


def test_manifest_v1_missing_file_still_fails(tmp_path):
    """v1 manifest 라도 file 누락은 fail."""
    manifest = {
        "schema_version": 1,
        "files": [{"path": "ghost.pt", "kind": "model"}],
    }
    _make_project(tmp_path, manifest)
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "manifest_evidence"), None
    )
    assert check is not None
    assert check.status == "fail"
    assert "missing" in check.detail


def test_no_output_dir_skips_post_run_gate(tmp_path):
    """output 디렉토리 없으면 manifest_evidence check 자체가 추가되지 않음."""
    (tmp_path / "cq.yaml").write_text(
        "name: t\n"
        "cmd: uv run python train.py\n"
        "metrics: []\n"
    )
    (tmp_path / "train.py").write_text(
        "import pcq\ncq.config()\ncq.log(epoch=0)\n"
    )
    report = validate_project(tmp_path)
    ids = {c.id for c in report.checks}
    assert "manifest_evidence" not in ids


def test_corrupt_manifest_json_fails(tmp_path):
    """망가진 manifest.json → blocking fail with parse error."""
    (tmp_path / "cq.yaml").write_text(
        "name: t\n"
        "cmd: uv run python train.py\n"
        "metrics: []\n"
    )
    (tmp_path / "train.py").write_text(
        "import pcq\ncq.config()\ncq.log(epoch=0)\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text("{ not valid json")
    report = validate_project(tmp_path)
    check = next(
        (c for c in report.checks if c.id == "manifest_evidence"), None
    )
    assert check is not None
    assert check.status == "fail"
    assert check.severity == "blocking"
