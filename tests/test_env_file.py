"""`.env` loading tests: file values load, process env wins."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.config import Settings
from tests.support import clean_tmp, make_tmp


class TestDotEnv(unittest.TestCase):
    def test_env_file_is_loaded(self):
        tmp = make_tmp()
        self.addCleanup(clean_tmp, tmp)
        env_file = tmp / ".env"
        env_file.write_text(
            "DE_LLM_BACKEND=openai_compat\n"
            "DE_LLM_API_KEY=test-key\n"
            "DE_LLM_MODEL=test-model\n"
            "DE_SMALL_MODEL=test-small\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"DE_ENV_FILE": str(env_file)}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_backend, "openai_compat")
        self.assertEqual(settings.llm_api_key, "test-key")
        self.assertEqual(settings.llm_model, "test-model")

    def test_process_env_wins_over_file(self):
        tmp = make_tmp()
        self.addCleanup(clean_tmp, tmp)
        env_file = tmp / ".env"
        env_file.write_text(
            "DE_LLM_BACKEND=openai_compat\nDE_LLM_API_KEY=from-file\n",
            encoding="utf-8",
        )
        with patch.dict(
            os.environ,
            {"DE_ENV_FILE": str(env_file), "DE_LLM_BACKEND": "fake"},
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_backend, "fake")

    def test_missing_env_file_is_noop(self):
        with patch.dict(
            os.environ, {"DE_ENV_FILE": str(ROOT / "var" / "no-such.env")}, clear=True
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_backend, "fake")


if __name__ == "__main__":
    unittest.main(verbosity=2)
