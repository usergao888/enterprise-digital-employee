"""Context budget, tool-output pruning, and scratchpad tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.agents import DigitalEmployee
from de_agent.config import Settings
from de_agent.context import ContextBudget, ContextManager, prune_tool_output
from tests.support import clean_tmp, make_tmp


class TestContextBudget(unittest.TestCase):
    def test_fit_respects_section_budget(self):
        budget = ContextBudget(total_tokens=80, shares={"system": 1.0})
        text = "制度条款。" * 500
        fitted = budget.fit(text, "system")
        self.assertLess(len(fitted), len(text))
        self.assertIn("已按上下文预算截断", fitted)


class TestToolPruning(unittest.TestCase):
    def test_long_json_stays_valid(self):
        payload = {
            "found": 20,
            "results": [
                {"index": i, "content": "很长的检索原文" * 80, "score": 0.5}
                for i in range(20)
            ],
        }
        pruned = prune_tool_output(json.dumps(payload, ensure_ascii=False), max_chars=600)
        self.assertLessEqual(len(pruned), 600)
        decoded = json.loads(pruned)  # must not be a truncated JSON fragment
        self.assertIsInstance(decoded, dict)

    def test_plain_text_keeps_head_and_marker(self):
        text = "A" * 5000
        pruned = prune_tool_output(text, max_chars=300)
        self.assertLessEqual(len(pruned), 300)
        self.assertIn("已裁剪", pruned)


class TestScratchpad(unittest.TestCase):
    def test_record_turn_writes_per_thread_note(self):
        tmp = make_tmp()
        self.addCleanup(clean_tmp, tmp)
        manager = ContextManager(tmp, total_tokens=1000)
        manager.record_turn("thread-abc", "年假有几天？", "5 天。", [{"name": "knowledge_search"}])

        path = tmp / "scratchpad" / "thread-abc.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("年假有几天", text)
        self.assertIn("knowledge_search", text)

    def test_agent_ask_records_scratchpad(self):
        tmp = make_tmp()
        self.addCleanup(clean_tmp, tmp)
        agent = DigitalEmployee(
            Settings(
                llm_backend="fake",
                embed_backend="hashing",
                kb_dir=ROOT / "data" / "kb",
                index_dir=tmp,
                graph_runtime="custom",
                enable_skills=False,
            )
        )
        agent.ingest()
        result = agent.ask("年假有几天？", fresh_memory=True)
        path = tmp / "scratchpad" / f"{result['thread_id']}.md"
        self.assertTrue(path.exists())
        self.assertIn("年假", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
