"""pcq.mcp — MCP server exposing pcq CLI as agent-callable tools.

v4.1: agent runtime (Claude Code, Codex, 임의 LLM) 이 subprocess shell parsing
없이 pcq Python API 를 JSON dict 로 직접 호출하도록 하는 통합 레이어.

이 패키지는 `pcq[mcp]` extras 를 통해 들어오는 `mcp` (Anthropic 공식 SDK) 에
의존한다. extras 미설치 시 ``pcq.mcp.server`` import 만 실패하며, 다른 pcq
서피스는 영향을 받지 않는다.
"""
from __future__ import annotations


__all__ = ["create_server", "build_tools"]


def create_server():  # pragma: no cover - thin wrapper for lazy import
    """Lazy re-export of ``pcq.mcp.server.create_server``.

    Imported lazily so package import does not require ``mcp`` extras until the
    server is actually constructed.
    """
    from pcq.mcp.server import create_server as _impl

    return _impl()


def build_tools():  # pragma: no cover - thin wrapper for lazy import
    """Lazy re-export of ``pcq.mcp.tools.build_tools``."""
    from pcq.mcp.tools import build_tools as _impl

    return _impl()
