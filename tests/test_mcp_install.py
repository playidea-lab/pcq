"""agent install --mcp flag — .mcp.json 작성."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_install_agent_assets_with_mcp_writes_mcp_json(tmp_path: Path):
    """mcp=True → .mcp.json 새로 작성, pcq server entry 포함."""
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="mcp-init", force=True)
    result = install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=False, mcp=True
    )
    assert result.target == "claude"

    mcp_json = tmp_path / ".mcp.json"
    assert mcp_json.exists()
    data = json.loads(mcp_json.read_text(encoding="utf-8"))
    assert "mcpServers" in data
    assert "pcq" in data["mcpServers"]
    pcq_entry = data["mcpServers"]["pcq"]
    assert pcq_entry["command"] == "pcq"
    assert pcq_entry["args"] == ["mcp", "serve"]


def test_install_mcp_merges_with_existing_mcp_json(tmp_path: Path):
    """기존 .mcp.json 보존하고 pcq entry 만 추가."""
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="mcp-merge", force=True)
    existing = {
        "mcpServers": {
            "other": {
                "command": "other-cmd",
                "args": ["serve"],
            }
        }
    }
    mcp_json = tmp_path / ".mcp.json"
    mcp_json.write_text(json.dumps(existing), encoding="utf-8")

    install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=False, mcp=True
    )

    data = json.loads(mcp_json.read_text(encoding="utf-8"))
    assert "other" in data["mcpServers"]
    assert "pcq" in data["mcpServers"]


def test_install_mcp_skip_when_pcq_already_present_no_force(tmp_path: Path):
    """이미 pcq entry 있고 force=False 면 변경 없음."""
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="mcp-skip", force=True)
    custom_pcq = {
        "mcpServers": {
            "pcq": {
                "command": "/custom/path/pcq",
                "args": ["mcp", "serve", "--custom"],
            }
        }
    }
    mcp_json = tmp_path / ".mcp.json"
    mcp_json.write_text(json.dumps(custom_pcq), encoding="utf-8")

    install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=False, mcp=True
    )

    # 사용자 정의 그대로 유지
    after = mcp_json.read_text(encoding="utf-8")
    assert json.loads(after)["mcpServers"]["pcq"]["command"] == "/custom/path/pcq"


def test_install_mcp_force_overwrites_pcq_entry(tmp_path: Path):
    """force=True 면 기존 pcq entry 를 덮어씀."""
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="mcp-force", force=True)
    custom_pcq = {
        "mcpServers": {
            "pcq": {
                "command": "/old/pcq",
                "args": [],
            }
        }
    }
    mcp_json = tmp_path / ".mcp.json"
    mcp_json.write_text(json.dumps(custom_pcq), encoding="utf-8")

    install_agent_assets(
        tmp_path, target="claude", force=True, dry_run=False, mcp=True
    )
    data = json.loads(mcp_json.read_text(encoding="utf-8"))
    assert data["mcpServers"]["pcq"]["command"] == "pcq"
    assert data["mcpServers"]["pcq"]["args"] == ["mcp", "serve"]


def test_install_mcp_dry_run_does_not_write(tmp_path: Path):
    """dry_run=True 면 파일 변경 없음."""
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="mcp-dry", force=True)
    install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=True, mcp=True
    )
    assert not (tmp_path / ".mcp.json").exists()


def test_cli_agent_install_mcp_flag(tmp_path: Path, capsys):
    """`pcq agent install --mcp` flag 작동."""
    from pcq.agent.init import init_experiment
    from pcq.cli import main

    init_experiment(output_dir=tmp_path, name="cli-mcp", force=True)
    rc = main(
        [
            "agent",
            "install",
            "--target",
            "claude",
            "--path",
            str(tmp_path),
            "--mcp",
            "--json",
        ]
    )
    assert rc == 0
    assert (tmp_path / ".mcp.json").exists()
    data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert "pcq" in data["mcpServers"]


def test_cli_mcp_serve_help_works():
    """`pcq mcp --help` 작동 (extras 미설치 환경에서도)."""
    from pcq.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["mcp", "--help"])
    assert exc.value.code == 0


def test_cli_mcp_serve_subcommand_registered():
    """`pcq mcp serve --help` 작동."""
    from pcq.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["mcp", "serve", "--help"])
    assert exc.value.code == 0


# ── v4.2 GM-1: uv venv 감지 ────────────────────────────────────────────


def test_install_mcp_uses_global_pcq_when_no_venv(tmp_path: Path):
    """프로젝트에 .venv 가 없으면 기존 동작 — global pcq command."""
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="no-venv", force=True)
    install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=False, mcp=True
    )

    data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    pcq_entry = data["mcpServers"]["pcq"]
    assert pcq_entry["command"] == "pcq"
    assert pcq_entry["args"] == ["mcp", "serve"]


def test_install_mcp_uses_uv_wrapper_when_venv_exists(tmp_path: Path):
    """.venv/bin/pcq 가 있으면 uv run --directory 래퍼로 작성.

    fresh Claude Code 세션이 글로벌 PATH 에서 pcq 를 못 찾는 dogfood 회귀
    (research/mcp-dogfood GM-1) 을 방지.
    """
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="with-venv", force=True)
    # 가짜 venv 구조 생성 — 실제 pcq 바이너리 내용은 무관, 존재 여부만 본다.
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    (venv_bin / "pcq").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=False, mcp=True
    )

    data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    pcq_entry = data["mcpServers"]["pcq"]
    assert pcq_entry["command"] == "uv"
    # args 첫 번째는 'run', 그 다음 '--directory <abs>', 마지막은 'pcq mcp serve'
    args = pcq_entry["args"]
    assert args[0] == "run"
    assert args[1] == "--directory"
    assert Path(args[2]).resolve() == tmp_path.resolve()
    assert args[-3:] == ["pcq", "mcp", "serve"]


def test_install_mcp_uv_wrapper_skip_idempotent(tmp_path: Path):
    """venv 있는 상태에서 두 번 install — 두 번째는 skip (idempotent)."""
    from pcq.agent.init import init_experiment
    from pcq.agent.install import install_agent_assets

    init_experiment(output_dir=tmp_path, name="idem-uv", force=True)
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    (venv_bin / "pcq").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=False, mcp=True
    )
    result2 = install_agent_assets(
        tmp_path, target="claude", force=False, dry_run=False, mcp=True
    )
    # 두 번째 install — pcq entry 가 이미 expected 와 일치하므로 mcp_config skip
    mcp_ops = [
        op for op in result2.operations if op.kind == "mcp_config"
    ]
    assert mcp_ops, "expected mcp_config op in result"
    assert all(op.action == "skip" for op in mcp_ops)
