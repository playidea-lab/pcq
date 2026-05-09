"""inspect_project — outputs.manifest_schema_version exposed (v1.14)."""
from __future__ import annotations

import json
from pathlib import Path

from pcq.agent import inspect_project


def _make_min_project(tmp_path: Path) -> None:
    """script-style minimal project (cq.yaml + train.py)."""
    (tmp_path / "cq.yaml").write_text(
        "name: t\n"
        "cmd: uv run python train.py\n"
        "configs: {}\n"
        "metrics: []\n"
        "artifacts:\n  - output/\n"
    )
    (tmp_path / "train.py").write_text(
        "import pcq\ncq.config()\n"
    )


def test_inspect_recognizes_manifest_v2(tmp_path):
    _make_min_project(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "files": [
                    {
                        "path": "x.pt",
                        "kind": "model",
                        "sha256": "0" * 64,
                        "size_bytes": 0,
                    }
                ],
            }
        )
    )
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.has_manifest is True
    assert insp.outputs.manifest_schema_version == 2
    assert insp.outputs.manifest_files_count == 1


def test_inspect_recognizes_manifest_v1_legacy(tmp_path):
    _make_min_project(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [{"path": "x.pt", "kind": "model"}],
            }
        )
    )
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.manifest_schema_version == 1
    assert insp.outputs.manifest_files_count == 1


def test_inspect_no_manifest_leaves_schema_none(tmp_path):
    _make_min_project(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.has_manifest is False
    assert insp.outputs.manifest_schema_version is None
    assert insp.outputs.manifest_files_count is None


def test_inspect_corrupt_manifest_keeps_has_manifest_true(tmp_path):
    """망가진 manifest 도 has_manifest=True 는 유지하되 schema 필드는 None."""
    _make_min_project(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    (out / "manifest.json").write_text("{ broken json")
    insp = inspect_project(tmp_path)
    assert insp.outputs is not None
    assert insp.outputs.has_manifest is True
    assert insp.outputs.manifest_schema_version is None
    assert insp.outputs.manifest_files_count is None
