"""Permission policy and registry enforcement tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.permissions import PermissionPolicy
from de_agent.tools import Tool, ToolRegistry


def ticket_tool() -> Tool:
    return Tool(
        name="create_ticket",
        description="创建工单",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
        handler=lambda **kwargs: {"ticket_id": "TCK-1", "title": kwargs.get("title")},
        mutating=True,
    )


class TestPermissionPolicy(unittest.TestCase):
    def test_modes(self):
        self.assertFalse(
            PermissionPolicy("read_only").decide("create_ticket", True).allow
        )
        ask = PermissionPolicy("ask").decide("create_ticket", True)
        self.assertTrue(ask.allow)
        self.assertTrue(ask.requires_confirmation)
        auto = PermissionPolicy("auto").decide("create_ticket", True)
        self.assertTrue(auto.allow)
        self.assertFalse(auto.requires_confirmation)
        self.assertTrue(
            PermissionPolicy("read_only").decide("knowledge_search", False).allow
        )

    def test_allow_and_deny_lists(self):
        self.assertFalse(
            PermissionPolicy("auto", deny_tools={"create_ticket"})
            .decide("create_ticket", True)
            .allow
        )
        self.assertFalse(
            PermissionPolicy("auto", allow_tools={"knowledge_search"})
            .decide("create_ticket", True)
            .allow
        )


class TestRegistryEnforcement(unittest.TestCase):
    def test_read_only_denies_write(self):
        registry = ToolRegistry(
            tools={"create_ticket": ticket_tool()},
            policy=PermissionPolicy("read_only"),
        )
        result = registry.call("create_ticket", {"title": "x"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "permission_denied")
        self.assertEqual(registry.pending, {})

    def test_ask_parks_write_even_without_confirm_flag(self):
        registry = ToolRegistry(
            tools={"create_ticket": ticket_tool()},
            confirm_mutating=False,
            policy=PermissionPolicy("ask"),
        )
        result = registry.call("create_ticket", {"title": "x"})
        self.assertTrue(result.pending)
        self.assertEqual(len(registry.pending), 1)

    def test_auto_executes_write(self):
        registry = ToolRegistry(
            tools={"create_ticket": ticket_tool()},
            confirm_mutating=False,
            policy=PermissionPolicy("auto"),
        )
        result = registry.call("create_ticket", {"title": "x"})
        self.assertTrue(result.ok)
        self.assertIn("TCK-1", result.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
