"""Harness tests: budget stop, durable progress, HITL resume."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.agents import DigitalEmployee
from de_agent.config import Settings
from de_agent.harness import AgentHarness, RunBudget, TaskStep
from tests.support import clean_tmp, make_tmp


def make_agent(
    index_dir: Path, *, require_confirmation: bool = False
) -> DigitalEmployee:
    agent = DigitalEmployee(
        Settings(
            llm_backend="fake",
            embed_backend="hashing",
            kb_dir=ROOT / "data" / "kb",
            index_dir=index_dir,
            graph_runtime="custom",
            enable_skills=False,
            require_confirmation=require_confirmation,
        )
    )
    agent.ingest()
    return agent


class TestAgentHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()
        self.addCleanup(clean_tmp, self.tmp)

    def test_budget_stops_then_resumes(self):
        agent = make_agent(self.tmp)
        harness = AgentHarness(agent)
        steps = [
            TaskStep("leave", "年假有几天？"),
            TaskStep("sick", "病假需要什么材料？"),
            TaskStep("travel", "一线城市出差住宿每晚能报多少？"),
        ]

        first = harness.run(steps, budget=RunBudget(max_steps=1, max_cost_cny=1.0))
        self.assertEqual(first.status, "budget_exceeded")
        self.assertEqual(first.steps_done, 1)
        self.assertIn("max_steps=1", first.error)

        state = json.loads(
            (harness.runs_dir / first.run_id / "state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(state["next_index"], 1)
        self.assertTrue(
            (harness.runs_dir / first.run_id / "progress.md").exists()
        )

        resumed = AgentHarness(agent).resume(
            first.run_id, budget=RunBudget(max_steps=10, max_cost_cny=1.0)
        )
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.steps_done, 3)
        self.assertEqual(len(resumed.results), 3)

    def test_zero_cost_budget_stops_before_first_step(self):
        agent = make_agent(self.tmp)
        result = AgentHarness(agent).run(
            [TaskStep("q", "年假有几天？")],
            budget=RunBudget(max_steps=5, max_cost_cny=0.0),
        )
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.steps_done, 0)
        self.assertIn("max_cost_cny", result.error)

    def test_approval_gate_resumes_and_writes_ticket(self):
        agent = make_agent(self.tmp, require_confirmation=True)
        harness = AgentHarness(agent)
        ticket_path = self.tmp / "tickets.json"

        first = harness.run(
            [TaskStep("ticket", "电脑蓝屏了，帮我报修一下")],
            budget=RunBudget(max_steps=5, max_cost_cny=1.0),
        )
        self.assertEqual(first.status, "awaiting_approval")
        self.assertIsNotNone(first.pending_action)
        self.assertFalse(ticket_path.exists())

        resumed = harness.resume(first.run_id, approved=True)
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.steps_done, 1)
        self.assertEqual(resumed.results[0]["status"], "approved")
        self.assertIn("TCK-", resumed.results[0]["answer"])
        self.assertTrue(ticket_path.exists())

    def test_cancel_does_not_write(self):
        agent = make_agent(self.tmp, require_confirmation=True)
        harness = AgentHarness(agent)
        first = harness.run(
            [TaskStep("ticket", "电脑蓝屏了，帮我报修一下")],
            budget=RunBudget(max_steps=5, max_cost_cny=1.0),
        )
        resumed = harness.resume(first.run_id, approved=False)
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.results[0]["status"], "cancelled")
        self.assertFalse((self.tmp / "tickets.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
