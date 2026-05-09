"""ExperimentPlan + ChangeOp 직렬화/검증 — v1.10."""
from __future__ import annotations

import json

from pcq.agent.plan import ChangeOp, ExperimentPlan, ValidationPolicy


def test_change_op_set_config_to_dict():
    op = ChangeOp(op="set_config", key="epochs", value=80)
    d = op.to_dict()
    assert d == {"op": "set_config", "key": "epochs", "value": 80}


def test_change_op_set_atom_to_dict():
    op = ChangeOp(
        op="set_atom",
        atom="loss",
        name="cross_entropy",
        params={"ignore_index": -1},
    )
    d = op.to_dict()
    assert d["op"] == "set_atom"
    assert d["atom"] == "loss"
    assert d["name"] == "cross_entropy"
    assert d["params"] == {"ignore_index": -1}


def test_change_op_unsupported_op_validation_fails():
    op = ChangeOp(op="unknown_op")
    errors = op.validate()
    assert errors
    assert any("unsupported" in e for e in errors)


def test_change_op_set_config_missing_key_fails():
    op = ChangeOp(op="set_config", value=10)
    errors = op.validate()
    assert any("key required" in e for e in errors)


def test_change_op_set_atom_kind_inference():
    assert ChangeOp(op="set_atom", atom="loss", name="x")._infer_kind() == "loss"
    assert (
        ChangeOp(op="set_atom", atom="dataset_train", name="x")._infer_kind()
        == "dataset"
    )
    assert (
        ChangeOp(op="set_atom", atom="dataset_eval", name="x")._infer_kind()
        == "dataset"
    )
    assert (
        ChangeOp(op="set_atom", atom="dataset", name="x")._infer_kind() == "dataset"
    )
    assert ChangeOp(op="set_atom", atom="sched", name="x")._infer_kind() == "sched"
    assert ChangeOp(op="set_atom", atom="optim", name="x")._infer_kind() == "optim"
    assert ChangeOp(op="set_atom", atom="metric", name="x")._infer_kind() == "metric"
    # 알 수 없는 atom 키
    assert ChangeOp(op="set_atom", atom="bogus", name="x")._infer_kind() == "unknown"


def test_change_op_set_atom_unknown_kind_fails():
    op = ChangeOp(op="set_atom", atom="bogus_key", name="x")
    errors = op.validate()
    assert any("cannot infer kind" in e for e in errors)


def test_change_op_set_atom_missing_name_fails():
    op = ChangeOp(op="set_atom", atom="loss", name=None)
    errors = op.validate()
    assert any("name" in e for e in errors)


def test_experiment_plan_round_trip():
    plan = ExperimentPlan(
        id="exp-001",
        intent="test plan",
        base={"preset": "vision/fake_smoke"},
        target={"metric": "eval_acc", "mode": "max"},
        changes=[
            ChangeOp(op="set_config", key="epochs", value=5),
            ChangeOp(
                op="set_atom",
                atom="loss",
                name="cross_entropy",
                params={"ignore_index": -1},
            ),
        ],
    )
    d = plan.to_dict()
    # JSON-safe round-trip
    s = json.dumps(d)
    plan2 = ExperimentPlan.from_dict(json.loads(s))
    assert plan2.id == "exp-001"
    assert plan2.intent == "test plan"
    assert plan2.base["preset"] == "vision/fake_smoke"
    assert plan2.target["mode"] == "max"
    assert len(plan2.changes) == 2
    assert plan2.changes[0].op == "set_config"
    assert plan2.changes[1].atom == "loss"


def test_experiment_plan_validate_unknown_preset():
    plan = ExperimentPlan(
        id="exp-002",
        base={"preset": "does/not/exist"},
        changes=[ChangeOp(op="set_config", key="epochs", value=1)],
    )
    errors = plan.validate()
    assert any("preset" in e for e in errors)


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


# ─────────────────────────────────────────────────────────────────────
# v1.11: set_atom merge field + set_dataset_transform op
# ─────────────────────────────────────────────────────────────────────


def test_change_op_set_atom_merge_field():
    """merge=True 면 to_dict 에 merge 키 포함."""
    op = ChangeOp(
        op="set_atom",
        atom="dataset_train",
        name="voc_seg",
        params={"image_size": 384},
        merge=True,
    )
    d = op.to_dict()
    assert d["merge"] is True


def test_change_op_set_atom_merge_default_omitted():
    """merge=False (기본값) 는 dict 에서 생략 (clean serialization)."""
    op = ChangeOp(
        op="set_atom",
        atom="loss",
        name="cross_entropy",
        params={"ignore_index": -1},
    )
    d = op.to_dict()
    assert "merge" not in d


def test_change_op_set_atom_merge_round_trip():
    op = ChangeOp(
        op="set_atom",
        atom="dataset_train",
        name="voc_seg",
        params={"image_size": 384},
        merge=True,
    )
    d = op.to_dict()
    s = json.dumps(d)
    op2 = ChangeOp.from_dict(json.loads(s))
    assert op2.merge is True
    assert op2.atom == "dataset_train"
    assert op2.params == {"image_size": 384}


def test_change_op_set_atom_merge_allows_missing_name():
    """merge=True 에서는 name 미지정 허용 (apply 단계에서 base 상속)."""
    op = ChangeOp(
        op="set_atom",
        atom="dataset_train",
        name=None,
        params={"image_size": 384},
        merge=True,
    )
    errors = op.validate()
    assert not any("name" in e for e in errors)


def test_change_op_set_atom_replace_requires_name():
    """merge=False (기본) 에서는 name 필수."""
    op = ChangeOp(
        op="set_atom",
        atom="loss",
        name=None,
        params={},
    )
    errors = op.validate()
    assert any("name" in e for e in errors)


def test_change_op_set_dataset_transform_to_dict():
    op = ChangeOp(
        op="set_dataset_transform",
        split="train",
        params={"image_size": 384},
    )
    d = op.to_dict()
    assert d["op"] == "set_dataset_transform"
    assert d["split"] == "train"
    assert d["params"] == {"image_size": 384}


def test_change_op_set_dataset_transform_validate_split():
    op = ChangeOp(
        op="set_dataset_transform",
        split="bogus",
        params={"image_size": 384},
    )
    errors = op.validate()
    assert any("split" in e for e in errors)


def test_change_op_set_dataset_transform_validate_eval_split():
    op = ChangeOp(
        op="set_dataset_transform",
        split="eval",
        params={"image_size": 256},
    )
    errors = op.validate()
    assert errors == []


def test_change_op_set_dataset_transform_requires_params():
    op = ChangeOp(
        op="set_dataset_transform",
        split="train",
        params={},
    )
    errors = op.validate()
    assert any("params" in e for e in errors)


def test_change_op_set_dataset_transform_to_set_atom():
    """sugar → set_atom merge=true 변환."""
    op = ChangeOp(
        op="set_dataset_transform",
        split="train",
        params={"image_size": 384},
    )
    converted = op.to_set_atom()
    assert converted.op == "set_atom"
    assert converted.atom == "dataset_train"
    assert converted.merge is True
    assert converted.name is None
    assert converted.params == {"image_size": 384}


def test_change_op_set_dataset_transform_eval_to_set_atom():
    op = ChangeOp(
        op="set_dataset_transform",
        split="eval",
        params={"image_size": 256},
    )
    converted = op.to_set_atom()
    assert converted.atom == "dataset_eval"
    assert converted.merge is True


def test_change_op_set_dataset_transform_to_set_atom_only_for_sugar():
    """set_atom 에 to_set_atom 호출하면 ValueError."""
    import pytest

    op = ChangeOp(op="set_atom", atom="loss", name="x")
    with pytest.raises(ValueError):
        op.to_set_atom()


def test_experiment_plan_with_dataset_transform_round_trip():
    plan = ExperimentPlan(
        id="exp-dst",
        base={"preset": "vision/seg/voc_unet"},
        changes=[
            ChangeOp(
                op="set_dataset_transform",
                split="train",
                params={"image_size": 384},
            ),
        ],
    )
    d = plan.to_dict()
    # JSON-safe round-trip
    s = json.dumps(d)
    plan2 = ExperimentPlan.from_dict(json.loads(s))
    assert plan2.changes[0].op == "set_dataset_transform"
    assert plan2.changes[0].split == "train"
    assert plan2.changes[0].params == {"image_size": 384}


def test_experiment_plan_with_set_atom_merge_round_trip():
    plan = ExperimentPlan(
        id="exp-merge",
        base={"preset": "vision/seg/voc_unet"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="dataset_train",
                name="voc_seg",
                params={"image_size": 384},
                merge=True,
            ),
        ],
    )
    d = plan.to_dict()
    s = json.dumps(d)
    plan2 = ExperimentPlan.from_dict(json.loads(s))
    assert plan2.changes[0].merge is True
    assert plan2.changes[0].name == "voc_seg"
