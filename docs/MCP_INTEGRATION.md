# pcq MCP Integration (v4.1.0)

> Phase 6: agent runtime이 subprocess shell parsing 없이 pcq를 직접 호출.

The pcq MCP server exposes pcq's 14 CLI surfaces as Model Context Protocol
tools. Agent runtimes (Claude Code, Codex, custom LLM clients) can call
pcq with structured JSON instead of shelling out and parsing stdout.

## Why

- **Zero subprocess overhead** — handlers call pcq Python APIs directly.
- **Zero JSON parsing on the agent side** — MCP returns structured dicts.
- **Stable schema** — every tool's input/output is anchored in the
  `pcq.agent.json_contracts.JSON_CONTRACTS` registry frozen in v2.13.

## Install

```bash
uv add 'pcq[mcp]'
# or: pip install 'pcq[mcp]'
```

The `mcp` extras pull in the official Anthropic MCP Python SDK and its
transitive dependencies (`starlette`, `uvicorn`, `pydantic` for SSE).

## Run the server

```bash
# Default — stdio (Claude Code, Codex)
pcq mcp serve

# HTTP/SSE — for web services or remote clients
pcq mcp serve --transport sse --host 127.0.0.1 --port 8765
```

## Wire it into a project

```bash
# Generate cq.yaml + train.py + AGENTS rules + .mcp.json in one shot:
pcq init-experiment --output ./my-experiment --agent claude
pcq agent install --target claude --path ./my-experiment --mcp
```

The `--mcp` flag merges the following into `.mcp.json`:

```json
{
  "mcpServers": {
    "pcq": {
      "command": "pcq",
      "args": ["mcp", "serve"]
    }
  }
}
```

Existing `.mcp.json` files are preserved — only the `pcq` server entry is
added. Existing `pcq` entries are kept unless `--force` is passed.

## Tools exposed (14)

| Tool | Read-only | Maps to |
|------|-----------|---------|
| `resolve_project` | yes | `pcq resolve` |
| `inspect_project` | yes | `pcq inspect` |
| `validate_project` | yes | `pcq validate` |
| `validate_run` | yes | `pcq validate-run` |
| `describe_run` | yes | `pcq describe-run` |
| `compare_runs` | yes | `pcq compare-runs` |
| `lineage_chain` | yes | `pcq lineage` |
| `apply_plan` | no | `pcq apply-plan` |
| `apply_planset` | no | `pcq apply-planset` |
| `init_experiment` | no | `pcq init-experiment` |
| `finalize_run` | no | `pcq finalize` |
| `agent_install` | no | `pcq agent install` |
| `agent_status` | yes | `pcq agent status` |
| `run_experiment` | no | `pcq run` |

Read-only tools never mkdir, never mutate cq.yaml, and never spawn
subprocesses.

## Architecture

```
agent runtime (Claude Code / Codex / custom LLM)
       │
       │  JSON-RPC over stdio (or SSE)
       ▼
pcq mcp serve   ── pcq.mcp.server.create_server()
       │
       │  PcqTool.handler(args dict) -> dict
       ▼
pcq.agent.* Python APIs
       │
       ▼
cq.yaml / RunRecord / etc. (the contract surface)
```

`run_experiment` is the only handler that uses subprocess (it must run the
user's `cmd`). Everything else is in-process.

## Embedding in your own server

```python
from pcq.mcp.server import create_server
from mcp.server.stdio import stdio_server
import asyncio

async def main():
    server = create_server()
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

asyncio.run(main())
```

Or use only the tool registry without the MCP server wrapper:

```python
from pcq.mcp.tools import build_tools
import asyncio

tools = build_tools()
resolve = next(t for t in tools if t.name == "resolve_project")
result = asyncio.run(resolve.handler({"path": "."}))
```

## Trade-offs / open questions

- **MCP SDK API stability** — `mcp` is pre-1.0. Pin via `pcq[mcp]` extras
  rather than core deps so churn is contained.
- **Long-running training** — `run_experiment` blocks on the cmd. For
  multi-hour GPU training, prefer the CQ service queue
  (`cq_run_experiment` over remote MCP) instead of in-process.
- **`.mcp.json` location** — pcq writes to the project root. Per-user
  Claude Code config (`~/Library/Application Support/Claude/claude_desktop_config.json`
  on macOS) is intentionally not touched; users who want it global must
  edit it themselves.

## Validation

```bash
# Server boots, tools enumerable
uv run python -c "from pcq.mcp.server import create_server; s = create_server(); print(s.name)"

# CLI subcommand works
pcq mcp --help
pcq mcp serve --help

# Wire-up dry-run
pcq agent install --target claude --path . --mcp --dry-run --json
```
