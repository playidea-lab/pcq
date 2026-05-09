"""tests/test_planset_output_dir.py — Fix 5 (G7-1).

apply-planset 가 plan.changes 에 relative output_dir (예: "runs/gen7-X") 가 있을
때 그대로 cq.yaml 에 작성하면, train.py 가 expanded dir 에서 실행될 때
pcq.output_dir() 이 `<expanded>/runs/gen7-X/` 로 이중 nesting 됨.

fix: PlanSet expand 시 멤버 plan 의 set_config(output_dir=...) op 가
  - relative path → expanded dir 기준 'output' 으로 normalize (기본 강제)
  - absolute path → 그대로 보존
"""
from __future__ import annotations

from pathlib import Path

from pcq.agent.apply import apply_planset
from pcq.agent.plan import ChangeOp, ExperimentPlan, ExperimentPlanSet
from pcq.agent.yaml_io import read_yaml, write_yaml


def _bootstrap_project(tmp_path: Path) -> Path:
    """간단한 base project — cq.yaml + 빈 train.py."""
    project = tmp_path / "proj"
    project.mkdir()
    write_yaml(
        {
            "name": "demo",
            "cmd": "python train.py",
            "configs": {"epochs": 1},
        },
        project / "cq.yaml",
    )
    (project / "train.py").write_text(
        "import pcq\ncfg = pcq.config()\nprint(cfg)\n", encoding="utf-8"
    )
    return project


def _planset_with_member_output_dir(output_dir_value: str) -> ExperimentPlanSet:
    """단일 멤버 plan 이 output_dir 을 set_config 으로 지정하는 set."""
    plan = ExperimentPlan(
        id="plan-a",
        intent="test relative output_dir",
        base={},
        changes=[
            ChangeOp(op="set_config", key="output_dir", value=output_dir_value),
            ChangeOp(op="set_config", key="lr", value=0.01),
        ],
    )
    return ExperimentPlanSet(
        id="set-1",
        intent="check output_dir normalization",
        plans=[plan],
    )


def test_apply_planset_relative_output_dir_normalized(tmp_path: Path):
    """relative output_dir → expanded dir 기준 'output' 으로 normalize.

    plan 이 output_dir="runs/gen7-X" 로 지정해도, expanded cq.yaml 의
    configs.output_dir 은 expanded dir 안의 'output' 또는 absolute 변환된 값이어야
    한다 (이중 nesting 방지).
    """
    project = _bootstrap_project(tmp_path)
    ps = _planset_with_member_output_dir("runs/gen7-X")

    result = apply_planset(
        project, ps, output_pattern="runs/exp{i}", force=True
    )
    assert result.status == "applied", result.rejected_reasons

    # expanded 멤버 dir 의 cq.yaml 확인.
    expanded_dir = project / "runs" / "exp0"
    expanded_cq = read_yaml(expanded_dir / "cq.yaml")
    out_val = expanded_cq.get("configs", {}).get("output_dir")
    assert out_val is not None

    # ── 핵심 invariant: expanded_dir 안에서 (또는 absolute 로 expanded_dir.* 안에)
    # 위치해야 한다. 즉 train.py 가 expanded_dir 에서 실행돼도 pcq.output_dir() 이
    # `expanded_dir/runs/gen7-X` 같은 이중 경로로 nesting 되지 않아야 한다.
    out_path = Path(out_val)
    if out_path.is_absolute():
        # 절대 경로면 expanded_dir 안에 있어야 함.
        out_resolved = out_path.resolve()
        expanded_resolved = expanded_dir.resolve()
        assert (
            out_resolved == expanded_resolved
            or expanded_resolved in out_resolved.parents
        ), (
            f"output_dir absolute path must live under expanded dir.\n"
            f"  output_dir = {out_resolved}\n"
            f"  expanded   = {expanded_resolved}"
        )
    else:
        # relative 경로면 expanded_dir 기준 — 'runs/gen7-X' 를 그대로 두면
        # `expanded_dir/runs/gen7-X` 가 됨 → 이중 nesting. 그래서 relative 인 경우
        # 'output' 같은 expanded-dir-local path 여야 한다.
        assert "runs/gen7-X" not in out_val and "runs\\gen7-X" not in out_val, (
            f"member relative output_dir 'runs/gen7-X' leaked into expanded "
            f"cq.yaml as {out_val!r} — would double-nest under "
            f"{expanded_dir}/{out_val}"
        )


def test_apply_planset_absolute_output_dir_preserved(tmp_path: Path):
    """plan 이 absolute output_dir 을 명시하면 그대로 보존."""
    project = _bootstrap_project(tmp_path)
    abs_path = str(tmp_path / "elsewhere" / "exp1")
    ps = _planset_with_member_output_dir(abs_path)

    result = apply_planset(
        project, ps, output_pattern="runs/exp{i}", force=True
    )
    assert result.status == "applied", result.rejected_reasons

    expanded_cq = read_yaml(project / "runs" / "exp0" / "cq.yaml")
    out_val = expanded_cq.get("configs", {}).get("output_dir")
    assert out_val == abs_path, (
        f"absolute output_dir not preserved: got {out_val!r}, "
        f"expected {abs_path!r}"
    )


def test_apply_planset_no_double_nesting_when_resolved(tmp_path: Path):
    """expanded cq.yaml 을 resolve_project 로 해석해도 이중 nesting 없음.

    실전 시나리오: plan.changes 에 output_dir='runs/genX' 같은 relative 가
    있을 때 pcq.output_dir() 호출이 `<expanded>/runs/genX/` 로 빗나가지 않아야
    한다.
    """
    from pcq.agent.resolver import resolve_project

    project = _bootstrap_project(tmp_path)
    ps = _planset_with_member_output_dir("runs/garbage_path")
    result = apply_planset(
        project, ps, output_pattern="runs/exp{i}", force=True
    )
    assert result.status == "applied", result.rejected_reasons

    expanded_dir = project / "runs" / "exp0"
    rc = resolve_project(path=expanded_dir)
    # output_dir 은 absolute 로 expanded_dir 또는 그 sub-dir 안에 있어야 함.
    assert rc.output_dir is not None
    out_resolved = rc.output_dir.resolve()
    expanded_resolved = expanded_dir.resolve()
    # 즉, expanded_dir 와 동일하거나 그 하위.
    assert (
        out_resolved == expanded_resolved
        or expanded_resolved in out_resolved.parents
    ), (
        f"resolve_project produced output_dir outside expanded dir: "
        f"{out_resolved} vs expanded={expanded_resolved}"
    )
    # 그리고 `.../runs/exp0/runs/garbage_path` 같은 이중 nesting 형태는 아님.
    assert "garbage_path" not in str(out_resolved), (
        f"member's relative output_dir leaked into resolved path: "
        f"{out_resolved}"
    )
