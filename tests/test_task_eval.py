"""Task-level evaluation tests: 30 cases, summary, badcase attribution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.agents import DigitalEmployee
from de_agent.config import Settings
from de_agent.task_eval import TASKS, evaluate_tasks, render_task_report
from tests.support import clean_tmp, make_tmp


class TestTaskEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = make_tmp()
        agent = DigitalEmployee(
            Settings(
                llm_backend="fake",
                embed_backend="hashing",
                kb_dir=ROOT / "data" / "kb",
                index_dir=cls._tmp,
                graph_runtime="custom",
                enable_skills=False,
            )
        )
        agent.ingest()
        cls.report = evaluate_tasks(agent)

    @classmethod
    def tearDownClass(cls):
        clean_tmp(cls._tmp)

    def test_has_30_cases_and_rows(self):
        self.assertEqual(len(TASKS), 30)
        self.assertEqual(self.report["summary"]["cases"], 30)
        self.assertEqual(len(self.report["rows"]), 30)

    def test_summary_and_badcase_attribution(self):
        summary = self.report["summary"]
        self.assertGreaterEqual(summary["pass_rate"], 0.6)
        failed = summary["cases"] - summary["passed"]
        self.assertEqual(sum(summary["badcases"].values()), failed)
        self.assertIn("tool_missing", summary["badcases"])

    def test_report_renders(self):
        markdown = render_task_report(
            self.report["summary"], self.report["rows"]
        )
        self.assertIn("任务级评测报告", markdown)
        self.assertIn("Badcase 归因", markdown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
