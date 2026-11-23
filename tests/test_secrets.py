"""Secret-handling regression: Settings must not print the API key."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.config import Settings


class TestSecrets(unittest.TestCase):
    def test_settings_repr_hides_api_key(self):
        settings = Settings(llm_api_key="sk-very-secret-value")
        self.assertNotIn("sk-very-secret-value", repr(settings))
        self.assertNotIn("sk-very-secret-value", str(settings))

    def test_env_example_has_no_real_key(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        # Only placeholder comments are allowed; no long secret-looking value.
        self.assertNotRegex(text, r"DE_LLM_API_KEY=[A-Za-z0-9_-]{20,}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
