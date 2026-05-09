"""apply-plan: bounded mutation — v4.0.

v4.0: set_atom / set_dataset_transform op 제거. set_config 와 lineage(parent_run_*)
op 만 남는다. init_experiment 의 preset/style 인자 제거.
"""
from __future__ import annotations

import json

from pcq.agent import (
    ChangeOp,
    ExperimentPlan,
    apply_plan,
    init_experiment,
)
from pcq.agent.yaml_io import read_yaml


def _setup_project(tmp_path):
    init_experiment(tmp_path, force=True)
    return tmp_path


def test_apply_set_config_modifies_cq_yaml(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-001",
        intent="bigger run",
        base={"baseline": "init"},
        changes=[ChangeOp(op="set_config", key="epochs", value=5)],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied"
    assert "cq.yaml" in result.files_changed

    data = read_yaml(tmp_path / "cq.yaml")
    assert data["configs"]["epochs"] == 5


def test_apply_idempotent_no_changes(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-003",
        base={"baseline": "init"},
        changes=[ChangeOp(op="set_config", key="epochs", value=10)],
    )
    apply_plan(tmp_path, plan)
    result2 = apply_plan(tmp_path, plan)
    assert result2.status == "no_changes"
    assert result2.files_changed == []


def test_apply_rejects_invalid_op(tmp_path):
    _setup_project(tmp_path)
    plan_dict = {
        "schema_version": 1,
        "id": "exp-005",
        "base": {"baseline": "init"},
        "changes": [{"op": "drop_database"}],
    }
    result = apply_plan(tmp_path, plan_dict)
    assert result.status == "rejected"


def test_apply_rejects_when_no_cq_yaml(tmp_path):
    plan = ExperimentPlan(
        id="exp-006",
        base={"baseline": "init"},
        changes=[ChangeOp(op="set_config", key="epochs", value=1)],
    )
    result = apply_plan(tmp_path, plan)  # tmp_path 에는 cq.yaml 없음
    assert result.status == "rejected"
    assert any("cq.yaml" in r for r in result.rejected_reasons)


def test_apply_provenance_saved(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-006-prov",
        base={"baseline": "init"},
        changes=[ChangeOp(op="set_config", key="lr", value=0.005)],
    )
    apply_plan(tmp_path, plan)
    plan_file = tmp_path / ".pcq" / "plans" / "exp-006-prov.json"
    assert plan_file.exists()
    data = json.loads(plan_file.read_text())
    assert data["plan"]["id"] == "exp-006-prov"
    assert "applied_at" in data
    assert data["status"] == "applied"


def test_apply_rejected_no_provenance(tmp_path):
    """rejected plan 은 .pcq/plans 에 저장 안 됨 (실패한 변경 추적 X)."""
    _setup_project(tmp_path)
    # invalid op → rejected
    plan_dict = {
        "schema_version": 1,
        "id": "exp-rejected",
        "base": {},
        "changes": [{"op": "drop_database"}],
    }
    result = apply_plan(tmp_path, plan_dict)
    assert result.status == "rejected"
    plan_file = tmp_path / ".pcq" / "plans" / "exp-rejected.json"
    assert not plan_file.exists()


def test_apply_dict_input_works(tmp_path):
    """plan 인자로 dict 도 허용."""
    _setup_project(tmp_path)
    plan_dict = {
        "schema_version": 1,
        "id": "exp-007",
        "base": {"baseline": "init"},
        "changes": [{"op": "set_config", "key": "batch_size", "value": 32}],
    }
    result = apply_plan(tmp_path, plan_dict)
    assert result.status == "applied"
    data = read_yaml(tmp_path / "cq.yaml")
    assert data["configs"]["batch_size"] == 32


def test_apply_records_previous_value(tmp_path):
    """operations 항목에 previous_value 포함."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-008",
        base={"baseline": "init"},
        changes=[ChangeOp(op="set_config", key="epochs", value=99)],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied"
    op = result.operations[0]
    assert op["status"] == "applied"
    # init-experiment 가 epochs=1 로 시작했으므로 previous_value=1
    assert op["previous_value"] == 1


def test_apply_set_config_works_on_script_project(tmp_path):
    """script 프로젝트에서 set_config 정상 동작 (v4.0 의 유일 mutation op)."""
    init_experiment(tmp_path, force=True)
    plan = ExperimentPlan(
        id="exp-script-config",
        base={},
        changes=[
            ChangeOp(op="set_config", key="n_estimators", value=200),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons
    data = read_yaml(tmp_path / "cq.yaml")
    assert data["configs"]["n_estimators"] == 200


# ── v1.18 lineage: ExperimentPlan parent injection ────────────────────


def test_apply_plan_with_parent_run_id_injects_into_cq_yaml(tmp_path):
    """plan.parent_run_id / parent_run_path → cq.yaml.configs._parent_run_*."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-lineage-1",
        intent="iterate on prior run",
        base={"baseline": "init"},
        parent_run_id="run_parent_xyz",
        parent_run_path="../parent/output",
        changes=[ChangeOp(op="set_config", key="epochs", value=3)],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons

    data = read_yaml(tmp_path / "cq.yaml")
    assert data["configs"]["_parent_run_id"] == "run_parent_xyz"
    assert data["configs"]["_parent_run_path"] == "../parent/output"


def test_apply_plan_parent_only_no_changes(tmp_path):
    """parent 만 있고 changes 빈 plan 은 plan.validate() 에서 reject 되어야 함."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-lineage-only",
        base={"baseline": "init"},
        parent_run_id="run_parent_xyz",
        changes=[],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "rejected"


def test_apply_plan_parent_records_op_in_provenance(tmp_path):
    """set_parent_run_id / set_parent_run_path op 이 provenance 에 기록됨."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-lineage-2",
        base={"baseline": "init"},
        parent_run_id="run_p",
        parent_run_path="../p/output",
        changes=[ChangeOp(op="set_config", key="epochs", value=7)],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied"

    op_kinds = [op.get("op") for op in result.operations]
    assert "set_parent_run_id" in op_kinds
    assert "set_parent_run_path" in op_kinds


def test_apply_plan_parent_idempotent(tmp_path):
    """동일 parent 정보로 재적용하면 noop 처리."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-lineage-3",
        base={"baseline": "init"},
        parent_run_id="run_p",
        changes=[ChangeOp(op="set_config", key="epochs", value=3)],
    )
    apply_plan(tmp_path, plan)
    result2 = apply_plan(tmp_path, plan)
    # 두 번째 적용은 모두 noop → no_changes.
    assert result2.status == "no_changes"


def test_experiment_plan_round_trip_preserves_parent(tmp_path):
    """ExperimentPlan.to_dict / from_dict round-trip 시 parent 보존."""
    plan = ExperimentPlan(
        id="exp-rt",
        base={"baseline": "init"},
        parent_run_id="run_x",
        parent_run_path="cq://runs/x",
        changes=[ChangeOp(op="set_config", key="lr", value=0.001)],
    )
    d = plan.to_dict()
    assert d["parent_run_id"] == "run_x"
    assert d["parent_run_path"] == "cq://runs/x"
    plan2 = ExperimentPlan.from_dict(d)
    assert plan2.parent_run_id == "run_x"
    assert plan2.parent_run_path == "cq://runs/x"


def test_experiment_plan_to_dict_omits_empty_parent():
    """parent 가 None 이면 직렬화에서 제외."""
    plan = ExperimentPlan(
        id="exp-no-parent",
        base={"baseline": "init"},
        changes=[ChangeOp(op="set_config", key="lr", value=0.001)],
    )
    d = plan.to_dict()
    assert "parent_run_id" not in d
    assert "parent_run_path" not in d
