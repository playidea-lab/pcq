"""v2.3: plan-only context 에서 label_contract gate 작동 검증.

audit P1 #3 fix: plan 의 set_atom 변경 사항을 base preset 의 RecipeSpec.atoms
위에 시뮬레이션하여 ignore_index 충돌을 실행 전에 감지.
"""
from __future__ import annotations

import json

from pcq.agent import ExperimentPlan
from pcq.agent.validate import _validate_plan_label_contracts


def _make_plan(plan_id: str, base_preset: str, *, ignore_index: int) -> ExperimentPlan:
    """voc_unet base + cross_entropy(ignore_index=...) override plan."""
    return ExperimentPlan.from_dict({
        "schema_version": 1,
        "id": plan_id,
        "base": {"preset": base_preset},
        "changes": [
            {
                "op": "set_atom",
                "atom": "loss",
                "name": "cross_entropy",
                "params": {"ignore_index": ignore_index},
            },
        ],
    })


def test_validate_plan_detects_ignore_index_mismatch():
    """voc_unet (dataset ignore=-1) + plan (loss ignore=-100) → fail.

    audit P1 #3 의 실제 시나리오 — plan-only validation 이 이 충돌을 놓쳤음.
    """
    plan = _make_plan("exp-bad-ignore", "vision/seg/voc_unet", ignore_index=-100)
    checks = _validate_plan_label_contracts(plan)
    assert len(checks) == 1
    check = checks[0]
    assert check.id == "plan_label_contract"
    assert check.status == "fail"
    assert check.severity == "blocking"
    # detail 에 plan id 와 ignore_index 키워드 포함
    assert "exp-bad-ignore" in check.detail
    assert "ignore_index" in check.detail.lower()


def test_validate_plan_label_contract_passes_when_consistent():
    """voc_unet (dataset ignore=-1) + plan (loss ignore=-1) → pass."""
    plan = _make_plan("exp-good-ignore", "vision/seg/voc_unet", ignore_index=-1)
    checks = _validate_plan_label_contracts(plan)
    assert len(checks) == 1
    check = checks[0]
    assert check.id == "plan_label_contract"
    assert check.status == "pass"
    assert "exp-good-ignore" in check.detail


def test_validate_plan_no_base_preset_skips():
    """base.preset 없으면 silent skip — empty list."""
    plan = ExperimentPlan.from_dict({
        "schema_version": 1,
        "id": "exp-no-base",
        "base": {},
        "changes": [
            {"op": "set_config", "key": "epochs", "value": 10},
        ],
    })
    checks = _validate_plan_label_contracts(plan)
    assert checks == []


def test_validate_plan_unknown_preset_silent_skip():
    """존재하지 않는 base preset → silent skip (다른 gate 가 별도 보고)."""
    plan = ExperimentPlan.from_dict({
        "schema_version": 1,
        "id": "exp-unknown-base",
        "base": {"preset": "totally/not/a/preset"},
        "changes": [
            {
                "op": "set_atom",
                "atom": "loss",
                "name": "cross_entropy",
                "params": {"ignore_index": -100},
            },
        ],
    })
    checks = _validate_plan_label_contracts(plan)
    assert checks == []


def test_validate_plan_label_contract_via_cli(tmp_path):
    """CLI integration: pcq validate --plan 으로 label_contract gate 작동.

    voc_unet base + plan (loss ignore_index=-100) → plan_label_contract fail.
    """
    import subprocess
    import sys

    # minimal cq.yaml — validate 가 cq_yaml_exists gate 통과만 하면 됨.
    project_root = tmp_path / "proj"
    project_root.mkdir()
    (project_root / "cq.yaml").write_text(
        "name: voc-bad-plan\n"
        "cmd: uv run python train.py\n"
        "configs:\n"
        "  output_dir: output\n"
        "  preset: vision/seg/voc_unet\n"
        "metrics:\n"
        "  - epoch\n"
        "  - eval_iou\n"
        "artifacts: [output/]\n",
        encoding="utf-8",
    )
    # train.py — script-style entrypoint (validate 진입 가능)
    (project_root / "train.py").write_text(
        "import pcq\n"
        "cfg = pcq.config()\n"
        "pcq.log(epoch=0, eval_iou=0.0)\n"
        "pcq.save_all(history=[])\n",
        encoding="utf-8",
    )

    plan_path = project_root / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": 1,
        "id": "exp-bad-ignore",
        "base": {"preset": "vision/seg/voc_unet"},
        "changes": [
            {
                "op": "set_atom",
                "atom": "loss",
                "name": "cross_entropy",
                "params": {"ignore_index": -100},
            },
        ],
    }), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, "-m", "pcq.cli", "validate", str(project_root),
            "--plan", str(plan_path), "--json",
        ],
        capture_output=True, text=True, timeout=60,
    )
    # validate 는 fail 시 exit 1 — plan_label_contract fail 가 trigger 가능.
    out = json.loads(result.stdout) if result.stdout else {}
    label_check = next(
        (c for c in out.get("checks", []) if c["id"] == "plan_label_contract"),
        None,
    )
    assert label_check is not None, (
        f"plan_label_contract check missing. checks: {[c['id'] for c in out.get('checks', [])]}"
    )
    assert label_check["status"] == "fail"
