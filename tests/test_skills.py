"""Skill loader tests: metadata vs body (progressive disclosure)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.skills import SkillLibrary


class TestSkillLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = SkillLibrary(ROOT / "skills")

    def test_names_and_frontmatter(self):
        self.assertEqual(
            self.lib.names(), ["expense-qa", "sql-report", "ticket-triage"]
        )
        expense = self.lib.get("expense-qa")
        self.assertIsNotNone(expense)
        self.assertIn("knowledge_search", expense.allowed_tools)

    def test_select_expense_and_ticket(self):
        expense = [s.name for s in self.lib.select("一线城市出差住宿每晚能报多少？")]
        self.assertIn("expense-qa", expense)

        ticket = [s.name for s in self.lib.select("电脑蓝屏了，帮我报修一下")]
        self.assertEqual(ticket, ["ticket-triage"])

        sql = [s.name for s in self.lib.select("上个月各部门报销总额排名前三的是哪些？")]
        self.assertIn("sql-report", sql)

    def test_body_is_loaded_only_on_demand(self):
        meta = self.lib.describe_for_prompt()
        self.assertIn("expense-qa", meta)
        # A body heading must not leak into the always-on metadata.
        self.assertNotIn("差旅报销问答", meta)

        selected = self.lib.select("出差住宿标准")
        body = self.lib.instructions_for(selected)
        self.assertIn("差旅报销问答", body)

    def test_missing_dir_is_empty(self):
        self.assertEqual(SkillLibrary(ROOT / "var" / "no-such-skills").names(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
