"""MCP server tests.

The pure payload/whitelist tests run without the optional `mcp` package; the
server-construction test is skipped until `pip install mcp` has been run.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.agents import DigitalEmployee
from de_agent.config import Settings
from de_agent.mcp_server import READ_ONLY_TOOLS, call_read_only, read_only_tools
from tests.support import clean_tmp, make_tmp

try:
    from mcp.server.mcpserver import MCPServer  # noqa: F401

    HAS_MCP = True
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401

        HAS_MCP = True
    except ImportError:
        HAS_MCP = False


def make_agent() -> tuple[DigitalEmployee, Path]:
    index_dir = make_tmp()
    agent = DigitalEmployee(
        Settings(
            llm_backend="fake",
            embed_backend="hashing",
            kb_dir=ROOT / "data" / "kb",
            index_dir=index_dir,
            graph_runtime="custom",
            enable_skills=False,
        )
    )
    agent.ingest()
    return agent, index_dir


class TestMcpPayloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent, cls._tmp = make_agent()

    @classmethod
    def tearDownClass(cls):
        clean_tmp(cls._tmp)

    def test_only_read_only_tools_are_exposed(self):
        specs = read_only_tools(self.agent)
        self.assertEqual(set(specs), set(READ_ONLY_TOOLS))
        self.assertNotIn("create_ticket", specs)
        for spec in specs.values():
            self.assertIn("name", spec)
            self.assertIn("parameters", spec)

    def test_mutating_tool_is_rejected(self):
        payload = json.loads(
            call_read_only(self.agent, "create_ticket", {"title": "x"})
        )
        self.assertIn("error", payload)
        self.assertIn("not exposed", payload["error"])


@unittest.skipUnless(HAS_MCP, "mcp not installed; pip install mcp")
class TestMcpServer(unittest.TestCase):
    def test_server_builds(self):
        from de_agent.mcp_server import build_server

        agent, tmp = make_agent()
        try:
            server = build_server(agent)
            self.assertIsNotNone(server)
        finally:
            clean_tmp(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
