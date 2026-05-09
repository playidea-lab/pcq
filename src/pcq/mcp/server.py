"""pcq MCP server — expose 14 pcq CLI surfaces as MCP tools.

Transports:
- ``stdio`` (default): Claude Code / Codex 표준
- ``sse`` (HTTP Server-Sent Events): web service / 원격 호출 시 옵션

Entry points:
- ``pcq mcp serve`` — CLI subcommand (src/pcq/cli.py)
- ``pcq.mcp.server.create_server()`` — Python embedding
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError as e:  # pragma: no cover - extras-gated
    raise ImportError(
        "pcq MCP server requires the `mcp` extras. "
        "Install with: uv add 'pcq[mcp]'"
    ) from e

from pcq.mcp.tools import PcqTool, build_tools


logger = logging.getLogger("pcq.mcp.server")


def _serialize_result(value: Any) -> str:
    """JSON-serialize a tool's dict return as compact, indented text."""
    return json.dumps(value, indent=2, default=str)


def create_server() -> Server:
    """Construct an MCP Server with all pcq tools registered.

    The Server is returned unbound to any transport; callers wire it via
    ``stdio_server`` or an SSE/Starlette adapter as needed.
    """
    server: Server = Server("pcq")
    tools: list[PcqTool] = build_tools()
    by_name: dict[str, PcqTool] = {t.name: t for t in tools}

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [t.descriptor for t in tools]

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        tool = by_name.get(name)
        if tool is None:
            raise ValueError(f"unknown pcq MCP tool: {name}")
        try:
            result = await tool.handler(arguments or {})
        except Exception as e:  # noqa: BLE001
            # Surface as structured error rather than propagating the
            # exception — agents need a stable JSON envelope.
            logger.exception("pcq MCP tool %s failed", name)
            result = {
                "schema_version": 1,
                "status": "error",
                "tool": name,
                "error": f"{type(e).__name__}: {e}",
            }
        return [TextContent(type="text", text=_serialize_result(result))]

    return server


async def serve_stdio() -> None:
    """Run the pcq MCP server over stdio (Claude Code / Codex default)."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def serve_sse(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the pcq MCP server over SSE (HTTP).

    Lazily imports starlette/uvicorn (both pulled in by the ``mcp`` extras).
    """
    try:
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
    except ImportError as e:  # pragma: no cover - extras-gated
        raise ImportError(
            "pcq MCP SSE transport requires `starlette` and `uvicorn` "
            "(installed via `pcq[mcp]`)."
        ) from e

    server = create_server()
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    app = Starlette(
        debug=False,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()


def main_stdio() -> None:
    """Entry point for the stdio transport."""
    asyncio.run(serve_stdio())


def main_sse(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Entry point for the SSE transport."""
    asyncio.run(serve_sse(host=host, port=port))
