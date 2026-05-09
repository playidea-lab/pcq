"""MCP server initialization + tool listing tests.

`mcp` extras 가 설치되어야 모든 테스트가 의미있다. 없으면 skip.
"""
from __future__ import annotations

import pytest


pytest.importorskip("mcp", reason="pcq[mcp] extras required")


def test_create_server_returns_server_with_14_tools():
    """create_server() 가 14 tool 모두 등록한 Server 인스턴스를 반환."""
    from pcq.mcp.server import create_server
    from pcq.mcp.tools import build_tools

    server = create_server()
    assert server is not None
    assert server.name == "pcq"

    tools = build_tools()
    assert len(tools) == 14
    names = {t.name for t in tools}
    expected = {
        "resolve_project",
        "inspect_project",
        "validate_project",
        "validate_run",
        "describe_run",
        "compare_runs",
        "lineage_chain",
        "apply_plan",
        "apply_planset",
        "init_experiment",
        "finalize_run",
        "agent_install",
        "agent_status",
        "run_experiment",
    }
    assert names == expected


def test_each_tool_has_descriptor_and_handler():
    """각 tool 은 MCP Tool descriptor 와 async handler 를 가진다."""
    from pcq.mcp.tools import build_tools

    tools = build_tools()
    for t in tools:
        assert t.name
        assert t.descriptor is not None
        assert t.descriptor.name == t.name
        assert t.descriptor.description
        assert isinstance(t.descriptor.inputSchema, dict)
        assert t.descriptor.inputSchema.get("type") == "object"
        assert callable(t.handler)


def test_module_import_does_not_require_mcp_extras_at_attribute_level():
    """pcq.mcp 패키지 import 는 mcp extras 가 있을 때만 동작.

    extras 없이 server 모듈을 import 하면 ImportError 가 명확해야 한다.
    이 테스트는 extras 가 있는 환경에서는 import 가 성공함을 검증.
    """
    import pcq.mcp  # noqa: F401
    import pcq.mcp.server  # noqa: F401
    import pcq.mcp.tools  # noqa: F401


def test_tool_descriptors_match_expected_input_schema_shape():
    """입력 스키마는 JSON Schema (object + properties) 형태."""
    from pcq.mcp.tools import build_tools

    for t in build_tools():
        schema = t.descriptor.inputSchema
        assert "type" in schema
        assert "properties" in schema
        assert isinstance(schema["properties"], dict)
