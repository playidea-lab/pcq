"""ExperimentPlan + ChangeOp 직렬화/검증 — v4.0.

v4.0 에서 set_atom, set_dataset_transform op 는 제거되었다 (atom registry 제거).
남은 op: set_config, set_parent_run_id, set_parent_run_path.
"""
from __future__ import annotations

import json

from pcq.agent.plan import ChangeOp, ExperimentPlan, ValidationPolicy


def test_change_op_set_config_to_dict():
    op = ChangeOp(op="set_config", key="epochs", value=80)
    d = op.to_dict()
    assert d == {"op": "set_config", "key": "epochs", "value": 80}


def test_change_op_unsupported_op_validation_fails():
    op = ChangeOp(op="unknown_op")
    errors = op.validate()
    assert errors
    assert any("unsupported" in e for e in errors)


def test_change_op_set_config_missing_key_fails():
    op = ChangeOp(op="set_config", value=10)
    errors = op.validate()
    assert any("key required" in e for e in errors)


def test_experiment_plan_round_trip():
    """v4.0: set_config 만 있는 plan 의 round-trip."""
    plan = ExperimentPlan(
        id="exp-001",
        intent="test plan",
        base={"baseline_run": "runs/exp-000"},
        target={"metric": "eval_acc", "mode": "max"},
        changes=[
            ChangeOp(op="set_config", key="epochs", value=5),
            ChangeOp(op="set_config", key="lr", value=1e-4),
        ],
    )
    d = plan.to_dict()
    # JSON-safe round-trip
    s = json.dumps(d)
    plan2 = ExperimentPlan.from_dict(json.loads(s))
    assert plan2.id == "exp-001"
    assert plan2.intent == "test plan"
    assert plan2.base["baseline_run"] == "runs/exp-000"
    assert plan2.target["mode"] == "max"
    assert len(plan2.changes) == 2
    assert plan2.changes[0].op == "set_config"
    assert plan2.changes[0].key == "epochs"
    assert plan2.changes[1].key == "lr"


def test_experiment_plan_empty_changes_invalid():
    plan = ExperimentPlan(id="exp-003")
    errors = plan.validate()
    assert any("empty" in e for e in errors)


def test_experiment_plan_missing_id_invalid():
    plan = ExperimentPlan(
        id="",
        changes=[ChangeOp(op="set_config", key="epochs", value=1)],
    )
    errors = plan.validate()
    assert any("id" in e for e in errors)


def test_experiment_plan_aggregates_change_errors():
    plan = ExperimentPlan(
        id="exp-bad",
        changes=[
            ChangeOp(op="bogus_op"),
            ChangeOp(op="set_config", value=1),  # missing key
        ],
    )
    errors = plan.validate()
    assert len(errors) >= 2
    assert any("changes[0]" in e for e in errors)
    assert any("changes[1]" in e for e in errors)


def test_validation_policy_round_trip():
    p = ValidationPolicy(run_smoke=False, run_contract=True, allow_network=True)
    d = p.to_dict()
    p2 = ValidationPolicy.from_dict(d)
    assert p2.run_smoke is False
    assert p2.allow_network is True


def test_change_op_from_dict_ignores_unknown_keys():
    d = {
        "op": "set_config",
        "key": "epochs",
        "value": 10,
        "future_field": "ignored",
    }
    op = ChangeOp.from_dict(d)
    assert op.op == "set_config"
    assert op.key == "epochs"
    assert op.value == 10
