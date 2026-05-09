"""apply-plan: bounded mutation — v1.10."""
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
    init_experiment(tmp_path, preset="vision/fake_smoke", force=True)
    return tmp_path


def test_apply_set_config_modifies_cq_yaml(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-001",
        intent="bigger run",
        base={"preset": "vision/fake_smoke"},
        changes=[ChangeOp(op="set_config", key="epochs", value=5)],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied"
    assert "cq.yaml" in result.files_changed

    data = read_yaml(tmp_path / "cq.yaml")
    assert data["configs"]["epochs"] == 5


def test_apply_set_atom_records_overrides_data(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-002",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="loss",
                name="cross_entropy",
                params={"ignore_index": -1},
            ),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons

    data = read_yaml(tmp_path / "cq.yaml")
    od = data["configs"].get("_overrides_data", {})
    assert "loss" in od
    assert od["loss"]["kind"] == "loss"
    assert od["loss"]["name"] == "cross_entropy"
    assert od["loss"]["params"]["ignore_index"] == -1


def test_apply_idempotent_no_changes(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-003",
        base={"preset": "vision/fake_smoke"},
        changes=[ChangeOp(op="set_config", key="epochs", value=10)],
    )
    apply_plan(tmp_path, plan)
    result2 = apply_plan(tmp_path, plan)
    assert result2.status == "no_changes"
    assert result2.files_changed == []


def test_apply_rejects_unknown_atom(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-004",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(op="set_atom", atom="loss", name="does_not_exist"),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "rejected"
    assert any("does_not_exist" in r for r in result.rejected_reasons)


def test_apply_rejects_invalid_op(tmp_path):
    _setup_project(tmp_path)
    plan_dict = {
        "schema_version": 1,
        "id": "exp-005",
        "base": {"preset": "vision/fake_smoke"},
        "changes": [{"op": "drop_database"}],
    }
    result = apply_plan(tmp_path, plan_dict)
    assert result.status == "rejected"


def test_apply_rejects_when_no_cq_yaml(tmp_path):
    plan = ExperimentPlan(
        id="exp-006",
        base={"preset": "vision/fake_smoke"},
        changes=[ChangeOp(op="set_config", key="epochs", value=1)],
    )
    result = apply_plan(tmp_path, plan)  # tmp_path 에는 cq.yaml 없음
    assert result.status == "rejected"
    assert any("cq.yaml" in r for r in result.rejected_reasons)


def test_apply_provenance_saved(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-006-prov",
        base={"preset": "vision/fake_smoke"},
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
    plan = ExperimentPlan(
        id="exp-rejected",
        base={"preset": "vision/fake_smoke"},
        changes=[ChangeOp(op="set_atom", atom="loss", name="nonexistent")],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "rejected"
    plan_file = tmp_path / ".pcq" / "plans" / "exp-rejected.json"
    assert not plan_file.exists()


def test_apply_dict_input_works(tmp_path):
    """plan 인자로 dict 도 허용."""
    _setup_project(tmp_path)
    plan_dict = {
        "schema_version": 1,
        "id": "exp-007",
        "base": {"preset": "vision/fake_smoke"},
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
        base={"preset": "vision/fake_smoke"},
        changes=[ChangeOp(op="set_config", key="epochs", value=99)],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied"
    op = result.operations[0]
    assert op["status"] == "applied"
    # init-experiment 가 epochs=1 로 시작했으므로 previous_value=1
    assert op["previous_value"] == 1


def test_apply_set_atom_with_optim_kind(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-009",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(op="set_atom", atom="optim", name="adamw", params={"lr": 1e-4}),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons
    data = read_yaml(tmp_path / "cq.yaml")
    assert data["configs"]["_overrides_data"]["optim"]["name"] == "adamw"


# ─────────────────────────────────────────────────────────────────────
# v1.11: set_atom merge + set_dataset_transform
# ─────────────────────────────────────────────────────────────────────


def test_apply_set_atom_merge_with_existing_override(tmp_path):
    """1차로 전체 ref 작성 → 2차로 merge=True 로 일부 params 만 변경.

    merge 시 기존 override 의 root/image_set/download 가 보존되고
    image_size 만 갱신된다.
    """
    _setup_project(tmp_path)
    # 1차: voc_seg 전체 ref (vision/fake_smoke base 라서 임의 dataset 이름 OK).
    # voc_seg 가 vision extras 필요해서 fake 로 대체 — fake dataset 의
    # num_samples + num_classes + image_size 를 merge 검증.
    plan1 = ExperimentPlan(
        id="exp-merge-1",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="dataset_train",
                name="fake",
                params={
                    "num_samples": 64,
                    "num_classes": 10,
                    "image_size": 32,
                },
            ),
        ],
    )
    apply_plan(tmp_path, plan1)

    # 2차: image_size 만 merge 로 변경
    plan2 = ExperimentPlan(
        id="exp-merge-2",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="dataset_train",
                name="fake",
                params={"image_size": 48},
                merge=True,
            ),
        ],
    )
    result = apply_plan(tmp_path, plan2)
    assert result.status == "applied", result.rejected_reasons

    data = read_yaml(tmp_path / "cq.yaml")
    od_ref = data["configs"]["_overrides_data"]["dataset_train"]
    assert od_ref["params"]["image_size"] == 48
    # 기존 params 보존
    assert od_ref["params"]["num_samples"] == 64
    assert od_ref["params"]["num_classes"] == 10


def test_apply_set_atom_merge_inherits_base_recipe_atom(tmp_path):
    """merge=True 인데 기존 override 가 없을 때 base recipe SPEC.atoms 에서 상속.

    fake_smoke 의 dataset_train 은 fake(num_samples=128, num_classes=10,
    image_size=32). merge 로 image_size 만 변경 → 나머지는 base 에서 상속.
    """
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-merge-base",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="dataset_train",
                name=None,                                # base 의 name 상속
                params={"image_size": 64},
                merge=True,
            ),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons

    data = read_yaml(tmp_path / "cq.yaml")
    od = data["configs"]["_overrides_data"]
    assert "dataset_train" in od
    # base 의 name 상속
    assert od["dataset_train"]["name"] == "fake"
    # base 의 num_samples (128) 보존, image_size 만 64 로 갱신
    assert od["dataset_train"]["params"]["image_size"] == 64
    assert od["dataset_train"]["params"]["num_samples"] == 128


def test_apply_set_dataset_transform_inherits_base_name(tmp_path):
    """vision/fake_smoke base 에 set_dataset_transform → fake name 상속."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-dst-1",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_dataset_transform",
                split="train",
                params={"image_size": 64},
            ),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons

    data = read_yaml(tmp_path / "cq.yaml")
    od = data["configs"]["_overrides_data"]
    assert "dataset_train" in od
    assert od["dataset_train"]["name"] == "fake"
    assert od["dataset_train"]["params"]["image_size"] == 64
    # base 의 num_classes 상속
    assert od["dataset_train"]["params"]["num_classes"] == 10


def test_apply_set_dataset_transform_eval_split(tmp_path):
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-dst-eval",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_dataset_transform",
                split="eval",
                params={"image_size": 48},
            ),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons

    data = read_yaml(tmp_path / "cq.yaml")
    od = data["configs"]["_overrides_data"]
    assert "dataset_eval" in od
    assert od["dataset_eval"]["name"] == "fake"


def test_apply_set_dataset_transform_records_sugar_op(tmp_path):
    """provenance 에 sugar_op="set_dataset_transform" 기록."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-dst-sugar",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_dataset_transform",
                split="train",
                params={"image_size": 48},
            ),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "applied", result.rejected_reasons
    op = result.operations[0]
    assert op.get("sugar_op") == "set_dataset_transform"
    assert op.get("sugar_split") == "train"


def test_apply_set_atom_merge_rejects_when_no_base_or_override(tmp_path):
    """merge=True 인데 atom 키가 base 에도 없고 기존 override 도 없으면 reject."""
    _setup_project(tmp_path)
    # vision/fake_smoke 에 sched 가 등록 안 돼있음 → base 상속 실패해야 함.
    plan = ExperimentPlan(
        id="exp-merge-fail",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="sched",
                name=None,
                params={"T_max": 100},
                merge=True,
            ),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "rejected"
    assert any(
        "cannot resolve base atom name" in r for r in result.rejected_reasons
    )


def test_apply_set_dataset_transform_idempotent(tmp_path):
    """동일 transform 두 번 적용 → 두 번째는 no_changes."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-dst-idem",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_dataset_transform",
                split="train",
                params={"image_size": 48},
            ),
        ],
    )
    apply_plan(tmp_path, plan)
    result2 = apply_plan(tmp_path, plan)
    assert result2.status == "no_changes"


def test_apply_set_atom_merge_then_replace(tmp_path):
    """merge 적용 후 replace (merge=False) 로 완전 교체 가능."""
    _setup_project(tmp_path)
    plan_merge = ExperimentPlan(
        id="exp-mr-1",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="dataset_train",
                name=None,
                params={"image_size": 64},
                merge=True,
            ),
        ],
    )
    apply_plan(tmp_path, plan_merge)

    plan_replace = ExperimentPlan(
        id="exp-mr-2",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(
                op="set_atom",
                atom="dataset_train",
                name="fake",
                params={
                    "num_samples": 32,
                    "num_classes": 10,
                    "image_size": 16,
                },
            ),
        ],
    )
    result = apply_plan(tmp_path, plan_replace)
    assert result.status == "applied", result.rejected_reasons
    data = read_yaml(tmp_path / "cq.yaml")
    od_ref = data["configs"]["_overrides_data"]["dataset_train"]
    # replace 이므로 image_size=16 — merge 의 64 가 남아있지 않다
    assert od_ref["params"]["image_size"] == 16
    assert od_ref["params"]["num_samples"] == 32


# ─────────────────────────────────────────────────────────────────────
# v1.13: script-style 프로젝트에서 atom-level op 거부
# ─────────────────────────────────────────────────────────────────────


def test_apply_rejects_set_atom_on_script_project(tmp_path):
    """script style 프로젝트에서 set_atom op 는 reject (atom 시스템 미사용)."""
    init_experiment(tmp_path, preset=None, style="script", force=True)
    plan = ExperimentPlan(
        id="exp-script-reject",
        base={},
        changes=[
            ChangeOp(op="set_atom", atom="loss", name="cross_entropy"),
        ],
    )
    result = apply_plan(tmp_path, plan)
    assert result.status == "rejected"
    assert any(
        "contract script" in r or "recipe-style" in r
        for r in result.rejected_reasons
    )


def test_apply_set_config_works_on_script_project(tmp_path):
    """script 프로젝트라도 set_config 는 정상 동작."""
    init_experiment(tmp_path, preset=None, style="script", force=True)
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


def test_apply_full_train_after_plan_set_config(tmp_path):
    """apply-plan 후 train.py 가 실제로 학습 가능 — 통합 검증."""
    import os
    import subprocess
    import sys

    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-010-train",
        base={"preset": "vision/fake_smoke"},
        changes=[
            ChangeOp(op="set_config", key="batch_size", value=8),
            ChangeOp(op="set_config", key="epochs", value=1),
        ],
    )
    apply_plan(tmp_path, plan)
    # cfg.json 으로 train.py 호출
    cfg_path = tmp_path / "smoke.json"
    cfg_path.write_text(json.dumps({
        "preset": "vision/fake_smoke",
        "output_dir": str(tmp_path / "out"),
        "epochs": 1,
        "batch_size": 8,
        "seed": 42,
        "_metrics_declared": [
            "epoch", "train_loss", "train_acc", "eval_loss", "eval_acc",
        ],
    }))
    env = {**os.environ, "CQ_CONFIG_JSON": str(cfg_path)}
    proc = subprocess.run(
        [sys.executable, "train.py"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"train.py failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert (tmp_path / "out" / "model.pt").exists()


# ── v1.18 lineage: ExperimentPlan parent injection ────────────────────


def test_apply_plan_with_parent_run_id_injects_into_cq_yaml(tmp_path):
    """plan.parent_run_id / parent_run_path → cq.yaml.configs._parent_run_*."""
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-lineage-1",
        intent="iterate on prior run",
        base={"preset": "vision/fake_smoke"},
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
    """parent 만 있고 changes 빈 plan 은 plan.validate() 에서 reject 되어야 함.

    parent_run_id 주입은 changes 와 별개이지만, plan.validate() 가 'changes
    empty' 를 막는다. 즉 lineage 만 갱신하려면 dummy noop change 또는 직접
    cq.yaml 편집이 필요.
    """
    _setup_project(tmp_path)
    plan = ExperimentPlan(
        id="exp-lineage-only",
        base={"preset": "vision/fake_smoke"},
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
        base={"preset": "vision/fake_smoke"},
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
        base={"preset": "vision/fake_smoke"},
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
        base={"preset": "vision/fake_smoke"},
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
        base={"preset": "vision/fake_smoke"},
        changes=[ChangeOp(op="set_config", key="lr", value=0.001)],
    )
    d = plan.to_dict()
    assert "parent_run_id" not in d
    assert "parent_run_path" not in d
