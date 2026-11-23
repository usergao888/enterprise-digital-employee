"""Optional MCP server exposing the agent's read-only tools.

Requires the official Python SDK::

    pip install mcp

Run::

    python -m de_agent.mcp_server

Only read-only tools are exposed. ``create_ticket`` is deliberately not an MCP
tool: a remote client must not be able to write without the human-approval path.
"""

from __future__ import annotations

import json
from typing import Any

READ_ONLY_TOOLS = ("knowledge_search", "sql_query", "get_employee_profile")


def read_only_tools(agent: Any) -> dict[str, dict[str, Any]]:
    """JSON-Schema function specs for the read-only tools."""
    specs: dict[str, dict[str, Any]] = {}
    for name in READ_ONLY_TOOLS:
        tool = agent.registry.tools.get(name)
        if tool is not None:
            specs[name] = tool.spec()["function"]
    return specs


def call_read_only(agent: Any, name: str, arguments: dict[str, Any]) -> str:
    """Execute one whitelisted tool and return its text content."""
    if name not in READ_ONLY_TOOLS:
        return json.dumps(
            {"error": f"tool {name!r} is not exposed via MCP", "allowed": list(READ_ONLY_TOOLS)},
            ensure_ascii=False,
        )
    result = agent.registry.call(name, arguments)
    return result.content


def build_server(agent: Any = None) -> Any:
    """Build an MCP server. Supports SDK v1 (FastMCP) and v2 (MCPServer)."""
    try:  # mcp >= 2 renamed FastMCP to MCPServer
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # pragma: no cover - v1 fallback
        from mcp.server.fastmcp import FastMCP as Server

    if agent is None:
        from .agents import build_digital_employee

        agent = build_digital_employee()

    server = Server("enterprise-digital-employee")

    @server.tool()
    def knowledge_search(query: str, top_k: int = 5) -> str:
        """Search the enterprise knowledge base and return cited chunks."""
        return call_read_only(agent, "knowledge_search", {"query": query, "top_k": top_k})

    @server.tool()
    def sql_query(sql: str) -> str:
        """Run one read-only SELECT against the demo employee database."""
        return call_read_only(agent, "sql_query", {"sql": sql})

    @server.tool()
    def employee_profile(employee_id: str) -> str:
        """Look up a demo employee profile by employee id (e.g. E1003)."""
        return call_read_only(agent, "get_employee_profile", {"employee_id": employee_id})

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
