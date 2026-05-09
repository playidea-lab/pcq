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
