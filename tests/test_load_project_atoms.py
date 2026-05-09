"""Project atom auto-discovery (v1.12)."""
from __future__ import annotations

from pathlib import Path

import pcq
from pcq.registry.loader import list_sources, load_project_atoms


def _write_project_atom(tmp_path: Path, body: str) -> None:
    (tmp_path / "pcq_atoms.py").write_text(body, encoding="utf-8")


def test_load_cq_atoms_py_imports_and_marks_source(tmp_path):
    _write_project_atom(tmp_path, '''
import pcq

pcq.register_loss(
    "test_proj_loss_123",
    factory=lambda: __import__("torch").nn.CrossEntropyLoss(),
    meta={"tasks": ["classification"]},
)
''')
    report = load_project_atoms(tmp_path)
    assert "pcq_atoms" in report.modules_loaded
    assert any(
        a["name"] == "test_proj_loss_123" for a in report.atoms_registered
    )

    spec = pcq.registry.losses.get("test_proj_loss_123")
    assert spec.source == "project"
    # module 은 pcq_atoms (auto-load) 또는 그 reload 결과
    assert spec.module == "pcq_atoms" or "pcq_atoms" in spec.module


def test_load_atoms_glob(tmp_path):
    atoms_dir = tmp_path / "atoms"
    atoms_dir.mkdir()
    (atoms_dir / "models.py").write_text('''
import pcq
pcq.register_model(
    "test_proj_model_456",
    factory=lambda: None,
    meta={"tasks": ["classification"]},
)
''', encoding="utf-8")
    report = load_project_atoms(tmp_path)
    assert any(
        a["name"] == "test_proj_model_456" for a in report.atoms_registered
    )


def test_load_handles_import_errors(tmp_path):
    _write_project_atom(tmp_path, "raise RuntimeError('intentional')\n")
    report = load_project_atoms(tmp_path)
    assert len(report.errors) >= 1
    assert "intentional" in report.errors[0]["error"]


def test_load_handles_syntax_errors(tmp_path):
    _write_project_atom(tmp_path, "this is not valid python(\n")
    report = load_project_atoms(tmp_path)
    assert len(report.errors) >= 1


def test_load_no_atoms_no_errors(tmp_path):
    """pcq_atoms.py / atoms/ 둘 다 없는 경우 → 에러도 없고 등록도 없음."""
    report = load_project_atoms(tmp_path)
    assert report.atoms_registered == []
    assert report.errors == []


def test_load_nonexistent_path_no_errors(tmp_path):
    """존재하지 않는 path → silent."""
    report = load_project_atoms(tmp_path / "nonexistent")
    assert report.atoms_registered == []
    assert report.errors == []


def test_list_sources_separates_builtin_and_project(tmp_path):
    _write_project_atom(tmp_path, '''
import pcq
pcq.register_loss(
    "test_src_distinct_789",
    factory=lambda: __import__("torch").nn.CrossEntropyLoss(),
    meta={"tasks": ["classification"]},
)
''')
    load_project_atoms(tmp_path)
    sources = list_sources()
    assert "loss/cross_entropy" in sources["builtin"]
    assert "loss/test_src_distinct_789" in sources["project"]


def test_load_report_to_dict():
    report = load_project_atoms("/nonexistent/path/__")
    d = report.to_dict()
    assert d["schema_version"] == 1
    assert "modules_loaded" in d
    assert "atoms_registered" in d
    assert "errors" in d
