"""ExperimentPlanSet (v4.0) — multi-run schema expressivity.

v4.0: init_experiment 의 preset 인자 제거. base 는 free-form metadata.
"""
from __future__ import annotations

import json
import subprocess
import sys

from pcq.agent import (
    ChangeOp,
    ExperimentPlan,
    ExperimentPlanSet,
    apply_planset,
    init_experiment,
)
from pcq.agent.yaml_io import read_yaml


def _make_plans(n: int = 3) -> list[ExperimentPlan]:
    """N 개 simple plan — set_config(epochs=...) 만 사용."""
    plans = []
    for i in range(n):
        plans.append(ExperimentPlan(
            id=f"exp-{i:03d}",
            intent=f"sweep epoch {i}",
            base={"baseline": "init"},
            changes=[ChangeOp(op="set_config", key="epochs", value=i + 1)],
        ))
    return plans


def test_planset_schema_roundtrip():
    """to_dict → from_dict 동등성."""
    ps = ExperimentPlanSet(
        id="sweep-001",
        intent="lr sweep",
        base={"baseline": "init"},
        plans=_make_plans(2),
    )
    d = ps.to_dict()
    ps2 = ExperimentPlanSet.from_dict(d)
    assert ps2.id == "sweep-001"
    assert ps2.intent == "lr sweep"
    assert len(ps2.plans) == 2
    assert ps2.plans[0].id == "exp-000"


def test_planset_to_dict_omits_empty_intent():
    """intent 빈 값은 직렬화에서 제거 — clean shape."""
    ps = ExperimentPlanSet(id="s", plans=_make_plans(1))
    d = ps.to_dict()
    assert "intent" not in d
    assert "base" not in d
    assert "parent_run_id" not in d


def test_planset_validate_pass():
    """모든 멤버 valid + id unique → empty errors."""
    ps = ExperimentPlanSet(id="sweep", plans=_make_plans(3))
    errors = ps.validate()
    assert errors == []


def test_planset_validate_empty_plans():
    """plans 비어있으면 fail."""
    ps = ExperimentPlanSet(id="empty", plans=[])
    errors = ps.validate()
    assert any("empty" in e for e in errors)


def test_planset_validate_missing_id():
    """id 빈 문자열 → fail."""
    ps = ExperimentPlanSet(id="", plans=_make_plans(1))
    errors = ps.validate()
    assert any("id" in e for e in errors)


def test_planset_validate_member_invalid_propagates():
    """멤버 plan 중 하나 invalid → set 도 fail (sub-error 누적)."""
    plans = _make_plans(2)
    plans[1].id = ""   # 멤버 id 누락 → 자체 plan validate fail
    ps = ExperimentPlanSet(id="sw", plans=plans)
    errors = ps.validate()
    # 멤버 plan 별 prefix 가 포함되어야 함.
    assert any("plans[1]" in e for e in errors)


def test_planset_unique_plan_ids():
    """중복 plan id 시 명시적 fail."""
    plans = _make_plans(2)
    plans[1].id = plans[0].id   # 같은 id
    ps = ExperimentPlanSet(id="dup-test", plans=plans)
    errors = ps.validate()
    assert any("duplicate" in e.lower() for e in errors)


def test_apply_planset_expands_to_n_dirs(tmp_path):
    """3 plans → 3 output directories."""
    init_experiment(tmp_path, force=True)
    ps = ExperimentPlanSet(id="sw-001", plans=_make_plans(3))

    result = apply_planset(tmp_path, ps, output_pattern="runs/exp{i}")
    assert result.status == "applied"
    assert len(result.expanded) == 3
    # 각 dir 에 cq.yaml 이 생성되어 있다.
    for i in range(3):
        member_dir = tmp_path / "runs" / f"exp{i}"
        assert (member_dir / "cq.yaml").exists()
        # 멤버 plan 의 set_config 가 적용되었는지 (epochs 값).
        data = read_yaml(member_dir / "cq.yaml")
        assert data["configs"]["epochs"] == i + 1


def test_apply_planset_propagates_parent_run_id(tmp_path):
    """set 의 parent_run_id 가 멤버 plan 에 propagate."""
    init_experiment(tmp_path, force=True)
    ps = ExperimentPlanSet(
        id="sw-002",
        parent_run_id="run_baseline_abc",
        parent_run_path="runs/baseline",
        plans=_make_plans(2),
    )
    result = apply_planset(tmp_path, ps, output_pattern="runs/exp{i}")
    assert result.status == "applied"
    # 각 멤버 cq.yaml 의 configs._parent_run_id 가 set 의 값.
    for i in range(2):
        data = read_yaml(tmp_path / "runs" / f"exp{i}" / "cq.yaml")
        assert data["configs"].get("_parent_run_id") == "run_baseline_abc"
        assert data["configs"].get("_parent_run_path") == "runs/baseline"


def test_apply_planset_member_specific_parent_takes_precedence(tmp_path):
    """멤버 plan 이 자체 parent_run_id 를 명시했다면 set 값으로 덮어쓰지 않음."""
    init_experiment(tmp_path, force=True)
    plans = _make_plans(2)
    plans[0].parent_run_id = "explicit_parent_001"
    ps = ExperimentPlanSet(
        id="sw-003",
        parent_run_id="set_parent",
        plans=plans,
    )
    apply_planset(tmp_path, ps, output_pattern="runs/exp{i}")
    data0 = read_yaml(tmp_path / "runs" / "exp0" / "cq.yaml")
    data1 = read_yaml(tmp_path / "runs" / "exp1" / "cq.yaml")
    assert data0["configs"].get("_parent_run_id") == "explicit_parent_001"
    assert data1["configs"].get("_parent_run_id") == "set_parent"


def test_apply_planset_skips_existing_dir_without_force(tmp_path):
    """기존 dir + force=false 면 멤버 skip 으로 기록."""
    init_experiment(tmp_path, force=True)
    target = tmp_path / "runs" / "exp0"
    target.mkdir(parents=True)
    (target / "marker").write_text("already here")

    ps = ExperimentPlanSet(id="sw-004", plans=_make_plans(2))
    result = apply_planset(tmp_path, ps, output_pattern="runs/exp{i}", force=False)
    # exp0 은 skipped, exp1 은 applied.
    statuses = {e["plan_id"]: e["status"] for e in result.expanded}
    assert statuses["exp-000"] == "skipped"
    assert statuses["exp-001"] == "applied"
    # 기존 marker 가 보존되었다.
    assert (target / "marker").exists()


def test_apply_planset_force_overwrites(tmp_path):
    """force=true 면 기존 dir 도 cq.yaml 새로 작성."""
    init_experiment(tmp_path, force=True)
    target = tmp_path / "runs" / "exp0"
    target.mkdir(parents=True)
    (target / "marker").write_text("pre-existing")

    plans = _make_plans(1)
    plans[0].changes = [ChangeOp(op="set_config", key="epochs", value=999)]
    ps = ExperimentPlanSet(id="sw-005", plans=plans)
    result = apply_planset(tmp_path, ps, output_pattern="runs/exp{i}", force=True)
    assert result.status == "applied"
    # 멤버 cq.yaml 이 새로 작성되어 epoch=999 로 patch 되었다.
    data = read_yaml(target / "cq.yaml")
    assert data["configs"]["epochs"] == 999
    # 기존 파일은 그대로 (force 는 cq.yaml 만 새로 씀, 디렉토리 자체는 유지).
    assert (target / "marker").exists()


def test_apply_planset_rejects_when_no_base_cq_yaml(tmp_path):
    """base project 에 cq.yaml 없으면 rejected."""
    ps = ExperimentPlanSet(id="sw-006", plans=_make_plans(1))
    result = apply_planset(tmp_path, ps)
    assert result.status == "rejected"
    assert any("cq.yaml" in r for r in result.rejected_reasons)


def test_apply_planset_rejects_invalid_set(tmp_path):
    """validate 실패 → 어떤 dir 도 만들지 않음."""
    init_experiment(tmp_path, force=True)
    ps = ExperimentPlanSet(id="", plans=_make_plans(1))   # id 빈 → invalid
    result = apply_planset(tmp_path, ps)
    assert result.status == "rejected"
    assert not (tmp_path / "runs").exists()


def test_apply_planset_dict_input_accepted(tmp_path):
    """dict 도 받는다 — JSON 로드 후 직접 전달 가능."""
    init_experiment(tmp_path, force=True)
    ps = ExperimentPlanSet(id="sw-007", plans=_make_plans(2))
    result = apply_planset(tmp_path, ps.to_dict(), output_pattern="runs/exp{i}")
    assert result.status == "applied"
    assert len(result.expanded) == 2


def test_validate_planset_cli(tmp_path):
    """`pcq validate --planset` subprocess 동작."""
    # base project + planset JSON 작성
    init_experiment(tmp_path, force=True)
    ps = ExperimentPlanSet(id="sw-cli", plans=_make_plans(2))
    ps_path = tmp_path / "planset.json"
    ps_path.write_text(json.dumps(ps.to_dict()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable, "-m", "pcq.cli", "validate", str(tmp_path),
            "--planset", str(ps_path), "--json",
        ],
        capture_output=True, text=True, timeout=60,
    )
    # validate 는 non-zero exit on fail — planset 자체 valid 이면 0 또는 정상 종료.
    assert proc.returncode in (0, 1), proc.stderr
    payload = json.loads(proc.stdout)
    ids = {c["id"] for c in payload["checks"]}
    assert "planset_validation" in ids


def test_apply_planset_cli(tmp_path):
    """`pcq apply-planset` subprocess 동작."""
    init_experiment(tmp_path, force=True)
    ps = ExperimentPlanSet(id="sw-cli-apply", plans=_make_plans(2))
    ps_path = tmp_path / "planset.json"
    ps_path.write_text(json.dumps(ps.to_dict()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable, "-m", "pcq.cli", "apply-planset", str(ps_path),
            "--path", str(tmp_path), "--output-pattern", "runs/exp{i}",
            "--json",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["set_id"] == "sw-cli-apply"
    assert payload["status"] == "applied"
    assert len(payload["expanded"]) == 2
