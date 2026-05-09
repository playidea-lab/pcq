"""Install pcq agent runtime assets into project discovery paths."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path


_VALID_TARGETS = ("codex", "claude", "both")
_BLOCK_START = "<!-- BEGIN PCQ AGENT RULES -->"
_BLOCK_END = "<!-- END PCQ AGENT RULES -->"
# .mcp.json entry — agent runtime (Claude Code / Codex) 가 자동으로 pcq MCP
# 서버에 attach 할 때 사용하는 명령. extras 미설치 환경에서는 명령이 실패해도
# 경고만 보고 다른 도구는 계속 작동한다.
_PCQ_MCP_SERVER_ENTRY: dict[str, object] = {
    "command": "pcq",
    "args": ["mcp", "serve"],
}


@dataclass
class AgentInstallOperation:
    """One planned or applied file operation."""

    path: str
    action: str
    kind: str
    agent: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "action": self.action,
            "kind": self.kind,
            "agent": self.agent,
            "reason": self.reason,
        }


@dataclass
class AgentInstallResult:
    """JSON-safe result for `pcq agent install`."""

    schema_version: int = 1
    project_root: str = ""
    target: str = "codex"
    dry_run: bool = False
    force: bool = False
    files_created: list[str] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    operations: list[AgentInstallOperation] = field(default_factory=list)

    def add_op(
        self,
        *,
        path: str,
        action: str,
        kind: str,
        agent: str,
        reason: str = "",
    ) -> None:
        self.operations.append(
            AgentInstallOperation(
                path=path,
                action=action,
                kind=kind,
                agent=agent,
                reason=reason,
            )
        )
        if action == "create":
            self.files_created.append(path)
        elif action == "update":
            self.files_updated.append(path)
        elif action == "skip":
            self.files_skipped.append(path)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "project_root": self.project_root,
            "target": self.target,
            "dry_run": self.dry_run,
            "force": self.force,
            "files_created": self.files_created,
            "files_updated": self.files_updated,
            "files_skipped": self.files_skipped,
            "warnings": self.warnings,
            "operations": [op.to_dict() for op in self.operations],
        }


@dataclass
class AgentAssetStatus:
    """Status for one expected agent runtime asset."""

    path: str
    agent: str
    kind: str
    status: str
    detail: str = ""
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "agent": self.agent,
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class AgentStatusResult:
    """Read-only project agent runtime status."""

    schema_version: int = 1
    project_root: str = ""
    target: str = "codex"
    status: str = "missing"
    assets: list[AgentAssetStatus] = field(default_factory=list)
    repair_command: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "project_root": self.project_root,
            "target": self.target,
            "status": self.status,
            "assets": [asset.to_dict() for asset in self.assets],
            "repair_command": self.repair_command,
        }


def install_agent_assets(
    project_root: str | Path = ".",
    *,
    target: str = "codex",
    force: bool = False,
    dry_run: bool = False,
    mcp: bool = False,
) -> AgentInstallResult:
    """Install pcq agent instructions/skills into a project.

    The package stores canonical assets internally. This function copies or
    merges them into runtime-specific discovery paths:

    - Codex: `AGENTS.md`, `.agents/skills/pcq/SKILL.md`
    - Claude Code: `CLAUDE.md`, `.claude/skills/pcq/SKILL.md`

    When ``mcp=True``, also writes/merges ``.mcp.json`` so the agent runtime
    auto-attaches to the pcq MCP server (``pcq mcp serve``). Existing
    ``mcpServers`` entries are preserved; only the ``pcq`` key is added.
    Existing ``pcq`` entries are kept unless ``force=True``.

    Existing instruction files are never destructively overwritten without
    `force=True`; pcq rules are appended in a marked block when possible.
    """
    target = target.lower()
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"unknown target {target!r}; supported: {', '.join(_VALID_TARGETS)}"
        )

    root = Path(project_root).resolve()
    result = AgentInstallResult(
        project_root=str(root),
        target=target,
        dry_run=dry_run,
        force=force,
    )
    rules = _asset_text("AGENTS.pcq.md")
    skill = _asset_text("skills/pcq/SKILL.md")

    install_codex = target in ("codex", "both")
    install_claude = target in ("claude", "both")

    if install_codex:
        _merge_marked_file(
            root,
            Path("AGENTS.md"),
            rules,
            force=force,
            dry_run=dry_run,
            result=result,
            agent="codex",
            kind="instructions",
        )
        _write_file(
            root,
            Path(".agents/skills/pcq/SKILL.md"),
            skill,
            force=force,
            dry_run=dry_run,
            result=result,
            agent="codex",
            kind="skill",
        )

    if install_claude:
        claude_rules = _claude_rules(root, codex_installed=install_codex)
        _merge_marked_file(
            root,
            Path("CLAUDE.md"),
            claude_rules,
            force=force,
            dry_run=dry_run,
            result=result,
            agent="claude",
            kind="instructions",
        )
        _write_file(
            root,
            Path(".claude/skills/pcq/SKILL.md"),
            skill,
            force=force,
            dry_run=dry_run,
            result=result,
            agent="claude",
            kind="skill",
        )

    if mcp:
        _install_mcp_config(
            root,
            target=target,
            force=force,
            dry_run=dry_run,
            result=result,
        )

    return result


def _install_mcp_config(
    root: Path,
    *,
    target: str,
    force: bool,
    dry_run: bool,
    result: AgentInstallResult,
) -> None:
    """Merge a ``pcq`` MCP server entry into ``.mcp.json``.

    Behavior:
      - file missing → create with ``{"mcpServers": {"pcq": {...}}}``
      - file present, no ``pcq`` key → add it
      - file present, has ``pcq`` key matching expected → skip (idempotent)
      - file present, has ``pcq`` key differing → skip unless ``force=True``
    """
    mcp_path = root / ".mcp.json"
    rel = ".mcp.json"
    expected = {"command": _PCQ_MCP_SERVER_ENTRY["command"],
                "args": list(_PCQ_MCP_SERVER_ENTRY["args"])}  # type: ignore[arg-type]

    existing: dict | None = None
    if mcp_path.exists():
        try:
            existing = json.loads(mcp_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                result.warnings.append(
                    f"{rel}: not a JSON object; skipped (rewrite manually)"
                )
                result.add_op(
                    path=rel,
                    action="skip",
                    kind="mcp_config",
                    agent=target,
                    reason="existing .mcp.json is not a JSON object",
                )
                return
        except json.JSONDecodeError as e:
            result.warnings.append(f"{rel}: invalid JSON ({e}); skipped")
            result.add_op(
                path=rel,
                action="skip",
                kind="mcp_config",
                agent=target,
                reason=f"existing .mcp.json invalid: {e}",
            )
            return

    if existing is None:
        merged = {"mcpServers": {"pcq": expected}}
        action = "create"
        reason = "new .mcp.json with pcq entry"
    else:
        servers = existing.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            result.warnings.append(
                f"{rel}: mcpServers is not an object; skipped"
            )
            result.add_op(
                path=rel,
                action="skip",
                kind="mcp_config",
                agent=target,
                reason="mcpServers is not an object",
            )
            return
        current = servers.get("pcq")
        if current == expected:
            result.add_op(
                path=rel,
                action="skip",
                kind="mcp_config",
                agent=target,
                reason="pcq entry already matches expected configuration",
            )
            return
        if current is not None and not force:
            result.add_op(
                path=rel,
                action="skip",
                kind="mcp_config",
                agent=target,
                reason="pcq entry exists; use --force to overwrite",
            )
            result.warnings.append(
                f"{rel}: existing pcq entry differs; skipped without --force"
            )
            return
        servers["pcq"] = expected
        merged = existing
        if current is None:
            action = "update"
            reason = "added pcq entry to existing .mcp.json"
        else:
            action = "update"
            reason = "overwrote pcq entry by --force"

    if not dry_run:
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_path.write_text(
            json.dumps(merged, indent=2) + "\n", encoding="utf-8"
        )
    result.add_op(
        path=rel,
        action=action,
        kind="mcp_config",
        agent=target,
        reason=reason,
    )


def agent_assets_status(
    project_root: str | Path = ".",
    *,
    target: str = "codex",
) -> AgentStatusResult:
    """Inspect installed pcq agent runtime assets without modifying files."""
    target = target.lower()
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"unknown target {target!r}; supported: {', '.join(_VALID_TARGETS)}"
        )

    root = Path(project_root).resolve()
    result = AgentStatusResult(
        project_root=str(root),
        target=target,
        repair_command=(
            f"pcq agent install --target {target} --path {root}"
        ),
    )
    rules = _asset_text("AGENTS.pcq.md")
    skill = _asset_text("skills/pcq/SKILL.md")

    if target in ("codex", "both"):
        result.assets.append(
            _status_instruction(
                root,
                Path("AGENTS.md"),
                rules,
                agent="codex",
                install_target="codex",
            )
        )
        result.assets.append(
            _status_exact_file(
                root,
                Path(".agents/skills/pcq/SKILL.md"),
                skill,
                agent="codex",
                kind="skill",
                install_target="codex",
            )
        )

    if target in ("claude", "both"):
        result.assets.append(_status_claude_instruction(root))
        result.assets.append(
            _status_exact_file(
                root,
                Path(".claude/skills/pcq/SKILL.md"),
                skill,
                agent="claude",
                kind="skill",
                install_target="claude",
            )
        )

    statuses = {asset.status for asset in result.assets}
    if not result.assets or statuses == {"missing"}:
        result.status = "missing"
    elif statuses <= {"installed"}:
        result.status = "installed"
    elif "divergent" in statuses:
        result.status = "divergent"
    elif "unmanaged" in statuses:
        result.status = "unmanaged"
    elif "stale" in statuses:
        result.status = "stale"
    elif "missing" in statuses:
        result.status = "partial"
    else:
        result.status = "partial"
    return result


def _asset_text(relative: str) -> str:
    root = files("pcq.agent_assets")
    for part in relative.split("/"):
        root = root.joinpath(part)
    return root.read_text(encoding="utf-8").strip() + "\n"


def _claude_rules(root: Path, *, codex_installed: bool) -> str:
    agents_path = root / "AGENTS.md"
    agents_has_pcq = (
        agents_path.exists()
        and _has_pcq_rules(agents_path.read_text(encoding="utf-8"))
    )
    if codex_installed or agents_has_pcq:
        return (
            "@AGENTS.md\n\n"
            "## Claude Code\n\n"
            "Use the imported pcq agent rules for pcq experiments. "
            "Claude-specific project rules may be added below this block.\n"
        )
    return _asset_text("AGENTS.pcq.md")


def _status_instruction(
    root: Path,
    rel_path: Path,
    content: str,
    *,
    agent: str,
    install_target: str,
) -> AgentAssetStatus:
    path = root / rel_path
    rel = rel_path.as_posix()
    if not path.exists():
        return AgentAssetStatus(
            path=rel,
            agent=agent,
            kind="instructions",
            status="missing",
            detail="instruction file missing",
            suggested_fix=(
                f"pcq agent install --target {install_target} --path {root}"
            ),
        )

    current = path.read_text(encoding="utf-8")
    block = _marked_block(content)
    if block.strip() in current:
        return AgentAssetStatus(
            path=rel,
            agent=agent,
            kind="instructions",
            status="installed",
            detail="managed pcq block is up to date",
        )
    if _has_marked_block(current):
        return AgentAssetStatus(
            path=rel,
            agent=agent,
            kind="instructions",
            status="stale",
            detail="managed pcq block exists but differs from packaged asset",
            suggested_fix=(
                f"pcq agent install --target {install_target} "
                f"--path {root} --force"
            ),
        )
    if _has_pcq_rules(current):
        return AgentAssetStatus(
            path=rel,
            agent=agent,
            kind="instructions",
            status="unmanaged",
            detail="pcq rules found without managed marker",
            suggested_fix=(
                f"pcq agent install --target {install_target} --path {root}"
            ),
        )
    return AgentAssetStatus(
        path=rel,
        agent=agent,
        kind="instructions",
        status="missing",
        detail="instruction file exists but has no pcq rules",
        suggested_fix=(
            f"pcq agent install --target {install_target} --path {root}"
        ),
    )


def _status_claude_instruction(root: Path) -> AgentAssetStatus:
    rel_path = Path("CLAUDE.md")
    path = root / rel_path
    rel = rel_path.as_posix()
    if not path.exists():
        return AgentAssetStatus(
            path=rel,
            agent="claude",
            kind="instructions",
            status="missing",
            detail="instruction file missing",
            suggested_fix=f"pcq agent install --target claude --path {root}",
        )

    current = path.read_text(encoding="utf-8")
    if "@AGENTS.md" in current:
        agents_path = root / "AGENTS.md"
        if agents_path.exists() and _has_pcq_rules(
            agents_path.read_text(encoding="utf-8")
        ):
            return AgentAssetStatus(
                path=rel,
                agent="claude",
                kind="instructions",
                status="installed",
                detail="imports AGENTS.md with pcq rules",
            )
        return AgentAssetStatus(
            path=rel,
            agent="claude",
            kind="instructions",
            status="partial",
            detail="imports AGENTS.md, but AGENTS.md has no pcq rules",
            suggested_fix=f"pcq agent install --target both --path {root}",
        )

    return _status_instruction(
        root,
        rel_path,
        _asset_text("AGENTS.pcq.md"),
        agent="claude",
        install_target="claude",
    )


def _status_exact_file(
    root: Path,
    rel_path: Path,
    content: str,
    *,
    agent: str,
    kind: str,
    install_target: str,
) -> AgentAssetStatus:
    path = root / rel_path
    rel = rel_path.as_posix()
    if not path.exists():
        return AgentAssetStatus(
            path=rel,
            agent=agent,
            kind=kind,
            status="missing",
            detail=f"{kind} file missing",
            suggested_fix=(
                f"pcq agent install --target {install_target} --path {root}"
            ),
        )
    current = path.read_text(encoding="utf-8")
    if current == content:
        return AgentAssetStatus(
            path=rel,
            agent=agent,
            kind=kind,
            status="installed",
            detail=f"{kind} file is up to date",
        )
    if "name: pcq" in current:
        return AgentAssetStatus(
            path=rel,
            agent=agent,
            kind=kind,
            status="stale",
            detail=f"{kind} file exists but differs from packaged asset",
            suggested_fix=(
                f"pcq agent install --target {install_target} "
                f"--path {root} --force"
            ),
        )
    return AgentAssetStatus(
        path=rel,
        agent=agent,
        kind=kind,
        status="divergent",
        detail=f"{kind} file exists but does not look like pcq",
        suggested_fix=(
            f"inspect {rel}; use --force only if it should be replaced"
        ),
    )


def _write_file(
    root: Path,
    rel_path: Path,
    content: str,
    *,
    force: bool,
    dry_run: bool,
    result: AgentInstallResult,
    agent: str,
    kind: str,
) -> None:
    path = root / rel_path
    rel = rel_path.as_posix()
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == content:
            result.add_op(
                path=rel,
                action="skip",
                kind=kind,
                agent=agent,
                reason="already up to date",
            )
            return
        if not force:
            result.add_op(
                path=rel,
                action="skip",
                kind=kind,
                agent=agent,
                reason="exists; use --force to overwrite",
            )
            result.warnings.append(f"{rel} exists; skipped without --force")
            return
        action = "update"
        reason = "overwritten by --force"
    else:
        action = "create"
        reason = "missing"

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    result.add_op(
        path=rel,
        action=action,
        kind=kind,
        agent=agent,
        reason=reason,
    )


def _merge_marked_file(
    root: Path,
    rel_path: Path,
    content: str,
    *,
    force: bool,
    dry_run: bool,
    result: AgentInstallResult,
    agent: str,
    kind: str,
) -> None:
    path = root / rel_path
    rel = rel_path.as_posix()
    block = _marked_block(content)

    if not path.exists():
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(block, encoding="utf-8")
        result.add_op(
            path=rel,
            action="create",
            kind=kind,
            agent=agent,
            reason="missing",
        )
        return

    current = path.read_text(encoding="utf-8")
    if block.strip() in current:
        result.add_op(
            path=rel,
            action="skip",
            kind=kind,
            agent=agent,
            reason="already up to date",
        )
        return

    if _has_marked_block(current):
        if not force:
            result.add_op(
                path=rel,
                action="skip",
                kind=kind,
                agent=agent,
                reason="pcq block exists; use --force to replace",
            )
            result.warnings.append(
                f"{rel} has a pcq block; skipped without --force"
            )
            return
        new_text = _replace_marked_block(current, block)
        reason = "pcq block replaced by --force"
    elif _has_pcq_rules(current):
        result.add_op(
            path=rel,
            action="skip",
            kind=kind,
            agent=agent,
            reason="pcq rules already present without managed marker",
        )
        return
    else:
        new_text = current.rstrip() + "\n\n" + block
        reason = "pcq block appended"

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    result.add_op(
        path=rel,
        action="update",
        kind=kind,
        agent=agent,
        reason=reason,
    )


def _marked_block(content: str) -> str:
    return f"{_BLOCK_START}\n{content.strip()}\n{_BLOCK_END}\n"


def _has_marked_block(text: str) -> bool:
    return _BLOCK_START in text and _BLOCK_END in text


def _has_pcq_rules(text: str) -> bool:
    return "PCQ Agent Rules" in text or _BLOCK_START in text


def _replace_marked_block(text: str, block: str) -> str:
    start = text.find(_BLOCK_START)
    end = text.find(_BLOCK_END, start)
    if start < 0 or end < 0:
        return text.rstrip() + "\n\n" + block
    end += len(_BLOCK_END)
    return text[:start] + block.rstrip() + text[end:]
