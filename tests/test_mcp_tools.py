"""각 MCP tool 의 input/output 검증.

pcq.agent Python API 직접 호출이라 subprocess 없이 빠름.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("mcp", reason="pcq[mcp] extras required")


def _run(coro):
    return asyncio.run(coro)


def _find_tool(name: str):
    from pcq.mcp.tools import build_tools

    for t in build_tools():
        if t.name == name:
            return t
    raise AssertionError(f"tool not found: {name}")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """기본 cq.yaml + train.py 프로젝트."""
    from pcq.agent.init import init_experiment

    init_experiment(output_dir=tmp_path, name="mcp-test", force=True)
    return tmp_path


# ── resolve_project ───────────────────────────────────────────────


def test_resolve_project_tool_returns_resolved_config_dict(project: Path):
    tool = _find_tool("resolve_project")
    result = _run(tool.handler({"path": str(project)}))
    assert isinstance(result, dict)
    assert result["name"] == "mcp-test"
    assert result["cmd"] == "uv run python train.py"
    assert "cfg" in result


def test_resolve_project_tool_with_explicit_cq_yaml(project: Path):
    tool = _find_tool("resolve_project")
    result = _run(
        tool.handler(
            {
                "path": str(project),
                "cq_yaml_path": str(project / "cq.yaml"),
            }
        )
    )
    assert result["name"] == "mcp-test"


# ── inspect_project ───────────────────────────────────────────────


def test_inspect_project_tool(project: Path):
    tool = _find_tool("inspect_project")
    result = _run(tool.handler({"path": str(project)}))
    assert isinstance(result, dict)
    assert "entrypoint" in result
    assert "outputs" in result


# ── validate_project ──────────────────────────────────────────────


def test_validate_project_tool(project: Path):
    tool = _find_tool("validate_project")
    result = _run(tool.handler({"path": str(project), "strictness": 2}))
    assert "status" in result
    assert "checks" in result
    assert isinstance(result["checks"], list)


# ── validate_run ───────────────────────────────────────────────────


def test_validate_run_tool(project: Path):
    """train.py 실행 후 validate_run 호출."""
    import subprocess

    out = project / "output"
    env = {**dict(__import__("os").environ)}
    rc = subprocess.run(
        [sys.executable, "train.py"], cwd=str(project), env=env
    )
    assert rc.returncode == 0
    assert out.exists()

    tool = _find_tool("validate_run")
    result = _run(tool.handler({"output_dir": str(out), "strictness": 2}))
    assert "status" in result
    assert "checks" in result


# ── describe_run ───────────────────────────────────────────────────


def test_describe_run_tool(project: Path):
    import subprocess

    out = project / "output"
    rc = subprocess.run([sys.executable, "train.py"], cwd=str(project))
    assert rc.returncode == 0

    tool = _find_tool("describe_run")
    result = _run(tool.handler({"output_dir": str(out)}))
    assert "decision_facts" in result
    assert isinstance(result["decision_facts"], dict)


# ── compare_runs ───────────────────────────────────────────────────


def test_compare_runs_tool(project: Path, tmp_path: Path):
    import subprocess

    # Run twice into different output dirs by overriding cq.yaml output_dir.
    from pcq.agent.yaml_io import read_yaml, write_yaml

    out_a = project / "out_a"
    out_b = project / "out_b"

    cq_data = read_yaml(project / "cq.yaml")
    cq_data["configs"]["output_dir"] = "out_a"
    write_yaml(cq_data, project / "cq.yaml")
    rc = subprocess.run([sys.executable, "train.py"], cwd=str(project))
    assert rc.returncode == 0

    cq_data["configs"]["output_dir"] = "out_b"
    write_yaml(cq_data, project / "cq.yaml")
    rc = subprocess.run([sys.executable, "train.py"], cwd=str(project))
    assert rc.returncode == 0

    tool = _find_tool("compare_runs")
    result = _run(tool.handler({"a": str(out_a), "b": str(out_b)}))
    assert "decision_facts" in result
    assert "best" in result


# ── lineage_chain ──────────────────────────────────────────────────


def test_lineage_chain_tool(project: Path):
    import subprocess

    rc = subprocess.run([sys.executable, "train.py"], cwd=str(project))
    assert rc.returncode == 0

    tool = _find_tool("lineage_chain")
    result = _run(
        tool.handler({"output_dir": str(project / "output"), "max_depth": 5})
    )
    assert "chain" in result
    assert isinstance(result["chain"], list)


# ── apply_plan ─────────────────────────────────────────────────────


def test_apply_plan_tool_inline_plan(project: Path):
    """plan dict 직접 전달."""
    plan = {
        "id": "p1",
        "intent": "bump epochs",
        "changes": [
            {"op": "set_config", "key": "epochs", "value": 5},
        ],
    }
    tool = _find_tool("apply_plan")
    result = _run(tool.handler({"path": str(project), "plan": plan}))
    assert result["status"] in ("applied", "no_changes")
    assert result["plan_id"] == "p1"


def test_apply_plan_tool_plan_file(project: Path, tmp_path: Path):
    """plan_file path 전달."""
    plan = {
        "id": "p2",
        "intent": "bump lr",
        "changes": [
            {"op": "set_config", "key": "lr", "value": 0.01},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    tool = _find_tool("apply_plan")
    result = _run(
        tool.handler({"path": str(project), "plan_file": str(plan_path)})
    )
    assert result["status"] in ("applied", "no_changes")
    assert result["plan_id"] == "p2"


# ── apply_planset ──────────────────────────────────────────────────


def test_apply_planset_tool_inline(project: Path):
    planset = {
        "id": "ps1",
        "intent": "lr sweep",
        "plans": [
            {
                "id": "m_lr_1e3",
                "intent": "lr=1e-3",
                "changes": [
                    {"op": "set_config", "key": "lr", "value": 0.001},
                ],
            },
            {
                "id": "m_lr_1e4",
                "intent": "lr=1e-4",
                "changes": [
                    {"op": "set_config", "key": "lr", "value": 0.0001},
                ],
            },
        ],
    }
    tool = _find_tool("apply_planset")
    result = _run(
        tool.handler(
            {
                "path": str(project),
                "planset": planset,
                "output_pattern": "runs/exp{i}",
            }
        )
    )
    assert result["status"] in ("applied", "no_changes")


# ── init_experiment ────────────────────────────────────────────────


def test_init_experiment_tool(tmp_path: Path):
    out = tmp_path / "fresh_proj"
    tool = _find_tool("init_experiment")
    result = _run(
        tool.handler({"output": str(out), "name": "from-mcp", "force": True})
    )
    assert result["name"] == "from-mcp"
    assert (out / "cq.yaml").exists()
    assert (out / "train.py").exists()


# ── finalize_run ───────────────────────────────────────────────────


def test_finalize_run_tool(project: Path):
    import subprocess

    rc = subprocess.run([sys.executable, "train.py"], cwd=str(project))
    assert rc.returncode == 0

    tool = _find_tool("finalize_run")
    result = _run(
        tool.handler(
            {
                "output_dir": str(project / "output"),
                "project_root": str(project),
                "status": "completed",
            }
        )
    )
    assert "run_record_path" in result
    assert Path(result["run_record_path"]).exists()


# ── agent_install / agent_status ───────────────────────────────────


def test_agent_install_tool(tmp_path: Path):
    from pcq.agent.init import init_experiment

    init_experiment(output_dir=tmp_path, name="ai-test", force=True)
    tool = _find_tool("agent_install")
    result = _run(
        tool.handler(
            {"path": str(tmp_path), "target": "claude", "dry_run": False}
        )
    )
    assert result["target"] == "claude"
    assert "operations" in result


def test_agent_install_with_mcp_flag(tmp_path: Path):
    """--mcp flag 가 .mcp.json 에 entry 추가."""
    from pcq.agent.init import init_experiment

    init_experiment(output_dir=tmp_path, name="ai-mcp", force=True)
    tool = _find_tool("agent_install")
    result = _run(
        tool.handler(
            {
                "path": str(tmp_path),
                "target": "claude",
                "mcp": True,
            }
        )
    )
    assert result["target"] == "claude"
    mcp_json = tmp_path / ".mcp.json"
    assert mcp_json.exists()
    data = json.loads(mcp_json.read_text(encoding="utf-8"))
    assert "pcq" in data["mcpServers"]


def test_agent_status_tool(tmp_path: Path):
    from pcq.agent.init import init_experiment

    init_experiment(output_dir=tmp_path, name="ai-stat", force=True)
    tool = _find_tool("agent_status")
    result = _run(
        tool.handler({"path": str(tmp_path), "target": "claude"})
    )
    assert result["target"] == "claude"
    assert "assets" in result


# ── run_experiment ─────────────────────────────────────────────────


def test_run_experiment_tool_config_only(project: Path):
    """config_only=True 면 cmd 실행 안 함."""
    tool = _find_tool("run_experiment")
    result = _run(
        tool.handler({"path": str(project), "config_only": True})
    )
    assert result["status"] == "config_only"
    assert "runtime_cfg_path" in result


def test_run_experiment_tool_executes_cmd(project: Path):
    """config_only=False 면 cq.yaml.cmd 실행."""
    # Override cmd with python (more portable than uv run in test env)
    from pcq.agent.yaml_io import read_yaml, write_yaml

    cq_data = read_yaml(project / "cq.yaml")
    cq_data["cmd"] = f"{sys.executable} train.py"
    write_yaml(cq_data, project / "cq.yaml")

    tool = _find_tool("run_experiment")
    result = _run(tool.handler({"path": str(project)}))
    assert result["status"] in ("completed", "failed")
    assert "exit_code" in result
