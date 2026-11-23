"""End-to-end tests for the real LangGraph runtime (Phase 0).

Skipped when langgraph / langgraph-checkpoint-sqlite are not installed, so the
zero-dependency custom-engine suite still runs everywhere.
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
from tests.support import clean_tmp, make_tmp

try:
    from de_agent.langgraph_adapter import langgraph_available

    HAS_LANGGRAPH = langgraph_available()
except Exception:  # noqa: BLE001 - optional dependency
    HAS_LANGGRAPH = False


def make_agent(index_dir: Path, *, ingest: bool = True) -> DigitalEmployee:
    settings = Settings(
        llm_backend="fake",
        embed_backend="hashing",
        kb_dir=ROOT / "data" / "kb",
        index_dir=index_dir,
        enable_reviewer=True,
        max_revisions=1,
        require_confirmation=True,
        graph_runtime="langgraph",
    )
    agent = DigitalEmployee(settings)
    if ingest:
        agent.ingest()
    return agent


@unittest.skipUnless(HAS_LANGGRAPH, "langgraph not installed")
class TestLangGraphRuntime(unittest.TestCase):
    def _tmp(self) -> Path:
        path = make_tmp()
        self.addCleanup(clean_tmp, path)
        return path

    def test_runtime_is_langgraph(self):
        agent = make_agent(self._tmp())
        self.assertEqual(agent.runtime, "langgraph")
        self.assertIsNotNone(agent.langgraph_app)
        self.assertTrue(agent.stats()["langgraph"])

    def test_interrupt_then_confirm(self):
        tmp = self._tmp()
        agent = make_agent(tmp)
        result = agent.ask("电脑蓝屏了，帮我报修一下", fresh_memory=True)

        self.assertTrue(result["interrupted"])
        self.assertEqual(len(result["pending_actions"]), 1)
        self.assertIn("待确认", result["answer"])
        self.assertFalse((tmp / "tickets.json").exists())

        config = {"configurable": {"thread_id": result["thread_id"]}}
        snapshot = agent.langgraph_app.get_state(config)
        self.assertEqual(tuple(snapshot.next), ("approval",))

        action_id = result["pending_actions"][0]["action_id"]
        done = agent.confirm(action_id)
        self.assertTrue(done["ok"], done)
        self.assertIn("TCK-", done["result"])

        tickets = json.loads((tmp / "tickets.json").read_text(encoding="utf-8"))
        self.assertEqual(len(tickets), 1)

    def test_cancel_does_not_write(self):
        tmp = self._tmp()
        agent = make_agent(tmp)
        result = agent.ask("电脑蓝屏了，帮我报修一下", fresh_memory=True)
        action_id = result["pending_actions"][0]["action_id"]

        cancelled = agent.cancel(action_id)
        self.assertTrue(cancelled["cancelled"])
        self.assertFalse((tmp / "tickets.json").exists())

    def test_resume_after_restart(self):
        tmp = self._tmp()
        first = make_agent(tmp)
        result = first.ask("电脑蓝屏了，帮我报修一下", fresh_memory=True)
        thread_id = result["thread_id"]

        # New process / new DigitalEmployee, same index dir and checkpoint DB.
        second = make_agent(tmp, ingest=False)
        resumed = second.resume(thread_id, approved=True)

        self.assertIn("TCK-", resumed["answer"])
        tickets = json.loads((tmp / "tickets.json").read_text(encoding="utf-8"))
        self.assertEqual(len(tickets), 1)

    def test_confirm_survives_restart(self):
        tmp = self._tmp()
        first = make_agent(tmp)
        result = first.ask("电脑蓝屏了，帮我报修一下", fresh_memory=True)
        action_id = result["pending_actions"][0]["action_id"]

        # action_id -> thread_id is persisted next to the checkpoint, so the
        # CLI confirm path still works after a process restart.
        second = make_agent(tmp, ingest=False)
        done = second.confirm(action_id)

        self.assertTrue(done["ok"], done)
        self.assertIn("TCK-", done["result"])
        tickets = json.loads((tmp / "tickets.json").read_text(encoding="utf-8"))
        self.assertEqual(len(tickets), 1)

    def test_unknown_action_is_rejected(self):
        agent = make_agent(self._tmp())
        outcome = agent.confirm("act-doesnotexist")
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["error"], "unknown_action")


if __name__ == "__main__":
    unittest.main(verbosity=2)
