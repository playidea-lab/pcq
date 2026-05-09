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


# ── v4.2 GM-5: 멤버 dir 자족화 (workspace symlink) ───────────────────


def test_apply_planset_symlinks_train_py_to_member_dirs(tmp_path):
    """root 의 train.py / pyproject.toml 을 각 멤버 dir 에 link.

    research/mcp-dogfood GM-5: 이전엔 멤버 dir 에 train.py 가 없어
    `pcq run --path member/dir` 가 ScriptNotFoundError.
    """
    init_experiment(tmp_path, force=True, with_pyproject=True)
    # init 이 train.py 와 pyproject.toml 모두 작성.
    assert (tmp_path / "train.py").exists()
    assert (tmp_path / "pyproject.toml").exists()

    ps = ExperimentPlanSet(id="sw-link", plans=_make_plans(2))
    result = apply_planset(tmp_path, ps, output_pattern="runs/exp{i}")
    assert result.status == "applied"

    for i in range(2):
        member_dir = tmp_path / "runs" / f"exp{i}"
        # train.py 가 멤버 dir 에서도 보임 (symlink 또는 copy).
        assert (member_dir / "train.py").exists(), (
            f"train.py missing in {member_dir}"
        )
        assert (member_dir / "pyproject.toml").exists(), (
            f"pyproject.toml missing in {member_dir}"
        )
        # expanded entry 의 linked_files 에도 기록.
        entry = result.expanded[i]
        assert "linked_files" in entry
        assert "train.py" in entry["linked_files"]
        assert "pyproject.toml" in entry["linked_files"]


def test_apply_planset_link_idempotent_when_file_exists(tmp_path):
    """기존 파일이 있으면 덮어쓰지 않고 skip — 사용자 파일 보존."""
    init_experiment(tmp_path, force=True)
    member_pre = tmp_path / "runs" / "exp0"
    member_pre.mkdir(parents=True)
    custom_train = member_pre / "train.py"
    custom_train.write_text("# custom member train\n", encoding="utf-8")

    ps = ExperimentPlanSet(id="sw-link-idem", plans=_make_plans(1))
    apply_planset(tmp_path, ps, output_pattern="runs/exp{i}", force=True)

    # 기존 custom train.py 보존.
    assert custom_train.read_text(encoding="utf-8") == "# custom member train\n"


# ── v4.2 GM-6: output_dir 자동 fan-out ────────────────────────────────


def test_apply_planset_injects_unique_output_dir_per_member(tmp_path):
    """멤버 plan 에 output_dir 가 없으면 'output' 자동 주입.

    이전엔 멤버가 root cq.yaml 의 output_dir 를 그대로 상속해 N 멤버가
    같은 dir 에 artifact 작성 → 충돌. 이제 멤버 dir 기준 'output' 으로 격리.
    """
    init_experiment(tmp_path, force=True)
    # base cq.yaml 에 output_dir 명시 — root 가 'shared_output' 사용.
    from pcq.agent.yaml_io import read_yaml, write_yaml

    base = read_yaml(tmp_path / "cq.yaml")
    base.setdefault("configs", {})["output_dir"] = "shared_output"
    write_yaml(base, tmp_path / "cq.yaml")

    # 멤버는 output_dir 를 명시 안 함 — auto-injection 으로 'output' 강제.
    ps = ExperimentPlanSet(id="sw-out", plans=_make_plans(3))
    result = apply_planset(tmp_path, ps, output_pattern="runs/exp{i}")
    assert result.status == "applied"

    for i in range(3):
        member_cq = read_yaml(tmp_path / "runs" / f"exp{i}" / "cq.yaml")
        out_val = member_cq.get("configs", {}).get("output_dir")
        # 기본 'output' (멤버 dir 기준).
        assert out_val == "output", (
            f"member exp{i} output_dir not auto-injected: got {out_val!r}"
        )


def test_apply_planset_respects_user_set_config_output_dir(tmp_path):
    """멤버 plan 이 output_dir 명시 → 그대로 보존 (자동 주입하지 않음).

    GM-6 의 핵심 invariant — 사용자 의도 우선. '_normalize_member_output_dir'
    가 relative path 를 'output' 으로 normalize 하는 기존 동작은 유효.
    이 테스트는 absolute path 를 명시했을 때 그대로 보존되는지 확인.
    """
    init_experiment(tmp_path, force=True)
    abs_out = str((tmp_path / "elsewhere" / "exp0_out").resolve())
    plans = [
        ExperimentPlan(
            id="exp-explicit-out",
            base={},
            changes=[
                ChangeOp(op="set_config", key="output_dir", value=abs_out),
                ChangeOp(op="set_config", key="lr", value=0.01),
            ],
        )
    ]
    ps = ExperimentPlanSet(id="sw-user-out", plans=plans)
    result = apply_planset(tmp_path, ps, output_pattern="runs/exp{i}")
    assert result.status == "applied"

    from pcq.agent.yaml_io import read_yaml

    member_cq = read_yaml(tmp_path / "runs" / "exp0" / "cq.yaml")
    # 사용자 absolute output_dir 가 그대로 — 자동 주입의 'output' 이 아님.
    assert member_cq["configs"]["output_dir"] == abs_out


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
