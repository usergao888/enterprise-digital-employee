"""Trace recorder tests: one JSON trace per ask, OTel-shaped spans."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.agents import DigitalEmployee
from de_agent.config import Settings
from de_agent.tracing import load_trace
from tests.support import clean_tmp, make_tmp


def make_agent(tmp: Path, *, tracing: bool = True) -> DigitalEmployee:
    agent = DigitalEmployee(
        Settings(
            llm_backend="fake",
            embed_backend="hashing",
            kb_dir=ROOT / "data" / "kb",
            index_dir=tmp,
            graph_runtime="custom",
            enable_skills=False,
            enable_tracing=tracing,
        )
    )
    agent.ingest()
    return agent


class TestTracing(unittest.TestCase):
    def test_ask_writes_trace_with_spans(self):
        tmp = make_tmp()
        self.addCleanup(clean_tmp, tmp)
        agent = make_agent(tmp)
        result = agent.ask("年假有几天？", fresh_memory=True)

        trace_id = result["trace_id"]
        self.assertTrue(trace_id)
        record = load_trace(tmp / "traces", trace_id)
        self.assertIsNotNone(record)
        names = {span["name"] for span in record["spans"]}
        self.assertIn("agent.ask", names)
        self.assertIn("node.plan", names)
        self.assertIn("llm.plan", names)
        self.assertIn("tool.knowledge_search", names)
        self.assertGreater(record["total_tokens"], 0)
        self.assertGreater(record["duration_ms"], 0)

    def test_tracing_can_be_disabled(self):
        tmp = make_tmp()
        self.addCleanup(clean_tmp, tmp)
        agent = make_agent(tmp, tracing=False)
        result = agent.ask("年假有几天？", fresh_memory=True)
        self.assertTrue(result["trace_id"])
        self.assertFalse((tmp / "traces").exists() and any((tmp / "traces").glob("*/*.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
