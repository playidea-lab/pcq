"""Agent runtime asset installation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pcq.agent import validate_json_contract
from pcq.agent.install import agent_assets_status, install_agent_assets


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args: str) -> tuple[int, dict | str, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pcq.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    try:
        out: dict | str = json.loads(result.stdout)
    except json.JSONDecodeError:
        out = result.stdout
    return result.returncode, out, result.stderr


def test_install_codex_assets(tmp_path):
    result = install_agent_assets(tmp_path, target="codex")

    assert "AGENTS.md" in result.files_created
    assert ".agents/skills/pcq/SKILL.md" in result.files_created
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / ".agents" / "skills" / "pcq" / "SKILL.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()
    assert "PCQ Agent Rules" in (tmp_path / "AGENTS.md").read_text()
    assert "name: pcq" in (
        tmp_path / ".agents" / "skills" / "pcq" / "SKILL.md"
    ).read_text()


def test_status_missing_for_fresh_project(tmp_path):
    result = agent_assets_status(tmp_path, target="codex")

    assert result.status == "missing"
    assert result.target == "codex"
    assert {asset.status for asset in result.assets} == {"missing"}
    assert result.repair_command.startswith("pcq agent install")


def test_status_installed_after_codex_install(tmp_path):
    install_agent_assets(tmp_path, target="codex")

    result = agent_assets_status(tmp_path, target="codex")

    assert result.status == "installed"
    assert [asset.status for asset in result.assets] == [
        "installed",
        "installed",
    ]


def test_status_partial_when_skill_missing(tmp_path):
    install_agent_assets(tmp_path, target="codex")
    (tmp_path / ".agents" / "skills" / "pcq" / "SKILL.md").unlink()

    result = agent_assets_status(tmp_path, target="codex")

    assert result.status == "partial"
    assert {asset.path: asset.status for asset in result.assets} == {
        "AGENTS.md": "installed",
        ".agents/skills/pcq/SKILL.md": "missing",
    }


def test_status_stale_for_modified_managed_block(tmp_path):
    install_agent_assets(tmp_path, target="codex")
    agents = tmp_path / "AGENTS.md"
    agents.write_text(
        agents.read_text(encoding="utf-8").replace(
            "PCQ Agent Rules",
            "PCQ Agent Rules (edited)",
            1,
        ),
        encoding="utf-8",
    )

    result = agent_assets_status(tmp_path, target="codex")

    assert result.status == "stale"
    assert {asset.path: asset.status for asset in result.assets}[
        "AGENTS.md"
    ] == "stale"


def test_status_unmanaged_for_manual_instruction_rules(tmp_path):
    (tmp_path / "AGENTS.md").write_text(
        "# Project\n\n## PCQ Agent Rules\n\n- manual copy\n",
        encoding="utf-8",
    )

    result = agent_assets_status(tmp_path, target="codex")

    assert result.status == "unmanaged"
    assert {asset.path: asset.status for asset in result.assets}[
        "AGENTS.md"
    ] == "unmanaged"


def test_status_divergent_for_existing_custom_skill(tmp_path):
    skill = tmp_path / ".agents" / "skills" / "pcq" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("custom skill\n", encoding="utf-8")

    result = agent_assets_status(tmp_path, target="codex")

    assert result.status == "divergent"
    assert {asset.path: asset.status for asset in result.assets}[
        ".agents/skills/pcq/SKILL.md"
    ] == "divergent"


def test_status_claude_import_installed_after_both_install(tmp_path):
    install_agent_assets(tmp_path, target="both")

    result = agent_assets_status(tmp_path, target="claude")

    assert result.status == "installed"
    assert {asset.path: asset.status for asset in result.assets} == {
        "CLAUDE.md": "installed",
        ".claude/skills/pcq/SKILL.md": "installed",
    }


def test_packaged_assets_match_repo_mirrors():
    from importlib.resources import files

    asset_root = files("pcq.agent_assets")
    agents_asset = asset_root.joinpath("AGENTS.pcq.md").read_text(
        encoding="utf-8"
    )
    skill_asset = (
        asset_root
        .joinpath("skills")
        .joinpath("pcq")
        .joinpath("SKILL.md")
        .read_text(encoding="utf-8")
    )

    assert agents_asset == (REPO_ROOT / "templates" / "AGENTS.pcq.md").read_text(
        encoding="utf-8"
    )
    assert skill_asset == (REPO_ROOT / "skills" / "pcq" / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_install_is_idempotent(tmp_path):
    install_agent_assets(tmp_path, target="codex")

    result = install_agent_assets(tmp_path, target="codex")

    assert result.files_created == []
    assert "AGENTS.md" in result.files_skipped
    assert ".agents/skills/pcq/SKILL.md" in result.files_skipped


def test_install_appends_marker_block_to_existing_agents(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing Rules\n\n- keep me\n", encoding="utf-8")

    result = install_agent_assets(tmp_path, target="codex")

    assert "AGENTS.md" in result.files_updated
    text = agents.read_text(encoding="utf-8")
    assert "# Existing Rules" in text
    assert "- keep me" in text
    assert "<!-- BEGIN PCQ AGENT RULES -->" in text
    assert "PCQ Agent Rules" in text


def test_install_dry_run_does_not_write(tmp_path):
    result = install_agent_assets(tmp_path, target="both", dry_run=True)

    assert "AGENTS.md" in result.files_created
    assert ".agents/skills/pcq/SKILL.md" in result.files_created
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / ".agents").exists()
    assert result.dry_run is True


def test_install_both_assets_and_claude_import(tmp_path):
    result = install_agent_assets(tmp_path, target="both")

    assert "AGENTS.md" in result.files_created
    assert "CLAUDE.md" in result.files_created
    assert ".agents/skills/pcq/SKILL.md" in result.files_created
    assert ".claude/skills/pcq/SKILL.md" in result.files_created
    assert "@AGENTS.md" in (tmp_path / "CLAUDE.md").read_text()


def test_existing_skill_is_not_overwritten_without_force(tmp_path):
    skill = tmp_path / ".agents" / "skills" / "pcq" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("custom skill\n", encoding="utf-8")

    result = install_agent_assets(tmp_path, target="codex")

    assert ".agents/skills/pcq/SKILL.md" in result.files_skipped
    assert skill.read_text(encoding="utf-8") == "custom skill\n"
    assert result.warnings


def test_existing_skill_is_overwritten_with_force(tmp_path):
    skill = tmp_path / ".agents" / "skills" / "pcq" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("custom skill\n", encoding="utf-8")

    result = install_agent_assets(tmp_path, target="codex", force=True)

    assert ".agents/skills/pcq/SKILL.md" in result.files_updated
    assert "name: pcq" in skill.read_text(encoding="utf-8")


def test_cli_agent_install_codex(tmp_path):
    rc, out, _ = _run_cli(
        "agent", "install",
        "--target", "codex",
        "--path", str(tmp_path),
        "--json",
    )

    assert rc == 0
    assert isinstance(out, dict)
    assert validate_json_contract("pcq.agent_install.result", out) == []
    assert out["target"] == "codex"
    assert "AGENTS.md" in out["files_created"]
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / ".agents" / "skills" / "pcq" / "SKILL.md").exists()


def test_cli_agent_install_dry_run(tmp_path):
    rc, out, _ = _run_cli(
        "agent", "install",
        "--target", "both",
        "--path", str(tmp_path),
        "--dry-run",
        "--json",
    )

    assert rc == 0
    assert isinstance(out, dict)
    assert validate_json_contract("pcq.agent_install.result", out) == []
    assert out["dry_run"] is True
    assert "AGENTS.md" in out["files_created"]
    assert not (tmp_path / "AGENTS.md").exists()


def test_cli_agent_status_json_installed(tmp_path):
    install_agent_assets(tmp_path, target="codex")

    rc, out, _ = _run_cli(
        "agent", "status",
        "--target", "codex",
        "--path", str(tmp_path),
        "--json",
    )

    assert rc == 0
    assert isinstance(out, dict)
    assert validate_json_contract("pcq.agent_status.result", out) == []
    assert out["status"] == "installed"
    assert out["assets"][0]["path"] == "AGENTS.md"


def test_cli_agent_status_json_missing_returns_zero(tmp_path):
    rc, out, _ = _run_cli(
        "agent", "status",
        "--target", "codex",
        "--path", str(tmp_path),
        "--json",
    )

    assert rc == 0
    assert isinstance(out, dict)
    assert validate_json_contract("pcq.agent_status.result", out) == []
    assert out["status"] == "missing"


def test_cli_init_experiment_can_install_agent_assets(tmp_path):
    rc, out, _ = _run_cli(
        "init-experiment",
        "--output", str(tmp_path),
        "--agent", "claude",
        "--json",
    )

    assert rc == 0
    assert isinstance(out, dict)
    assert out["agent_install"]["target"] == "claude"
    assert "CLAUDE.md" in out["agent_install"]["files_created"]
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / ".claude" / "skills" / "pcq" / "SKILL.md").exists()
