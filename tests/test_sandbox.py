"""Sandbox helper tests (no Docker daemon required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.sandbox import SandboxSpec, docker_command, run_python


class TestSandbox(unittest.TestCase):
    def test_docker_command_is_hardened(self):
        command = docker_command(
            SandboxSpec(), ["node", "--check", "app.js"], workdir=Path.cwd()
        )
        joined = " ".join(command)
        self.assertIn("--network none", joined)
        self.assertIn("--read-only", joined)
        self.assertIn("--cap-drop ALL", joined)
        self.assertIn("--pids-limit 128", joined)
        self.assertIn("--user 1000:1000", joined)
        self.assertIn(":/work:ro", joined)
        self.assertEqual(command[-3:], ["node", "--check", "app.js"])

    def test_run_python_isolated_mode(self):
        result = run_python("print(6 * 7)", timeout_s=10)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stdout"].strip(), "42")

    def test_timeout_is_reported(self):
        result = run_python("import time; time.sleep(2)", timeout_s=0.5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
