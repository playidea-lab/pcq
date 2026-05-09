"""pcq atoms scaffold (v1.12)."""
from __future__ import annotations

import pytest

from pcq.agent.scaffold import scaffold_atom


@pytest.mark.parametrize(
    "kind", ["model", "loss", "dataset", "metric", "optim", "sched"],
)
def test_scaffold_creates_runnable_atom(kind, tmp_path):
    result = scaffold_atom(kind, f"my_{kind}_v12", project_root=tmp_path)
    assert result.status == "created"
    # atoms/<plural>.py 또는 atoms/__init__.py 둘 다 보임
    assert any(f.startswith("atoms/") for f in result.files_changed)
    assert result.atom == {"kind": kind, "name": f"my_{kind}_v12"}
    assert result.next_checks


def test_scaffolded_atom_loadable_and_smokes(tmp_path):
    """scaffold → load_project_atoms → registry 등록 → smoke 통과."""
    scaffold_atom("model", "my_smoke_model_v12", project_root=tmp_path)

    from pcq.registry.loader import load_project_atoms
    report = load_project_atoms(tmp_path)
    assert any(
        a["name"] == "my_smoke_model_v12" for a in report.atoms_registered
    )

    from pcq.agent.smoke import smoke_atom
    smoke = smoke_atom("model", "my_smoke_model_v12")
    assert smoke.passed, smoke.error or smoke.detail


def test_scaffold_rejects_invalid_name(tmp_path):
    result = scaffold_atom("model", "123-bad", project_root=tmp_path)
    assert result.status == "error"
    assert "invalid atom name" in (result.error or "")


def test_scaffold_rejects_unknown_kind(tmp_path):
    result = scaffold_atom("bogus", "my_atom", project_root=tmp_path)
    assert result.status == "error"
    assert "unknown kind" in (result.error or "")


def test_scaffold_skip_existing_without_force(tmp_path):
    scaffold_atom("loss", "first_loss_v12", project_root=tmp_path)
    result2 = scaffold_atom("loss", "first_loss_v12", project_root=tmp_path)
    assert result2.status == "skipped"


def test_scaffold_force_overwrites(tmp_path):
    scaffold_atom("loss", "force_test_v12", project_root=tmp_path)
    result2 = scaffold_atom(
        "loss", "force_test_v12", project_root=tmp_path, force=True,
    )
    assert result2.status == "created"


def test_scaffold_appends_second_atom_to_existing_file(tmp_path):
    scaffold_atom("model", "model_a_v12", project_root=tmp_path)
    result = scaffold_atom("model", "model_b_v12", project_root=tmp_path)
    assert result.status == "created"
    text = (tmp_path / "atoms" / "models.py").read_text()
    assert "model_a_v12" in text
    assert "model_b_v12" in text


def test_scaffold_custom_output_path(tmp_path):
    target = tmp_path / "custom" / "loss_file.py"
    result = scaffold_atom(
        "loss", "custom_path_loss_v12",
        output=str(target),
        project_root=tmp_path,
    )
    assert result.status == "created"
    assert target.exists()
    assert "custom_path_loss_v12" in target.read_text()
