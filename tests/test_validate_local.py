"""validate_local_atoms — project atom 메타데이터 검증 (v1.12)."""
from __future__ import annotations

from pcq.agent.scaffold import validate_local_atoms


def test_validate_local_no_project_atoms(tmp_path):
    report = validate_local_atoms(tmp_path)
    # 경고는 있지만 fail 은 아님
    assert report.status in ("pass", "warn")
    assert any(c.id == "project_atoms_present" for c in report.checks)


def test_validate_local_complete_atom_passes(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text('''
import pcq

pcq.register_model(
    "complete_model_v12",
    factory=lambda: None,
    meta={
        "tasks": ["classification"],
        "params": {},
        "input_contract": {"x": ["B", "C", "H", "W"]},
        "output_contract": {"logits": ["B", "C"]},
    },
)
''', encoding="utf-8")
    report = validate_local_atoms(tmp_path)
    # complete_model_v12 has full contracts → status pass (or warn 만 있음)
    assert report.status in ("pass", "warn")
    # 결정적 fail 없음
    assert report.blocking_count == 0


def test_validate_local_missing_contract_fails(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text('''
import pcq

pcq.register_model(
    "incomplete_model_v12",
    factory=lambda: None,
    meta={"tasks": ["classification"]},
)
''', encoding="utf-8")
    report = validate_local_atoms(tmp_path)
    assert report.status == "fail"
    assert any(
        c.id == "project_atom_model_contract" for c in report.checks
    )


def test_validate_local_import_error_fails(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text(
        "raise ImportError('oops_v12')\n", encoding="utf-8",
    )
    report = validate_local_atoms(tmp_path)
    assert report.status == "fail"
    assert any(c.id == "project_atom_import" for c in report.checks)


def test_validate_local_dataset_missing_output_contract(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text('''
import pcq

pcq.register_dataset(
    "incomplete_dataset_v12",
    factory=lambda split=None: None,
    meta={"tasks": ["classification"]},
)
''', encoding="utf-8")
    report = validate_local_atoms(tmp_path)
    assert any(
        c.id == "project_atom_dataset_contract" for c in report.checks
    )


def test_validate_local_metric_missing_mode(tmp_path):
    (tmp_path / "pcq_atoms.py").write_text('''
import pcq

pcq.register_metric(
    "incomplete_metric_v12",
    factory=lambda: lambda l, t: None,
    meta={
        "tasks": ["classification"],
        "input_contract": {"logits": ["B","C"], "target": ["B"]},
    },
)
''', encoding="utf-8")
    report = validate_local_atoms(tmp_path)
    assert any(
        c.id == "project_atom_metric_mode" for c in report.checks
    )
