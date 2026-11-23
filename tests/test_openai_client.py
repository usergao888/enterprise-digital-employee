"""OpenAI-compatible client tests against a local stub server (no API key)."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from de_agent.config import Settings
from de_agent.llm import LLMError, Message, OpenAICompatClient


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.last_payload = json.loads(body.decode("utf-8"))  # type: ignore[attr-defined]
        if getattr(self.server, "fail", False):  # type: ignore[attr-defined]
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"invalid key"}')
            return
        payload = {
            "model": "stub-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "knowledge_search",
                                    "arguments": '{"query": "年假"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args, **kwargs):  # keep test output clean
        return


class TestOpenAICompatClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        base = f"http://127.0.0.1:{cls.httpd.server_address[1]}/v1"
        cls.settings = Settings(
            llm_backend="openai_compat",
            llm_base_url=base,
            llm_api_key="test-key",
            llm_model="stub-model",
            small_model="stub-model",
        )

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_chat_parses_content_tool_calls_and_usage(self):
        client = OpenAICompatClient(self.settings)
        completion = client.chat(
            [Message.user("年假有几天？")],
            stage="execute",
            tools=[{"type": "function", "function": {"name": "knowledge_search"}}],
        )
        self.assertEqual(completion.message.content, "ok")
        self.assertEqual(completion.message.tool_calls[0].name, "knowledge_search")
        self.assertIn("年假", completion.message.tool_calls[0].arguments)
        self.assertEqual(completion.prompt_tokens, 12)
        self.assertEqual(completion.completion_tokens, 7)
        payload = self.httpd.last_payload  # type: ignore[attr-defined]
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_http_error_becomes_llm_error(self):
        self.httpd.fail = True  # type: ignore[attr-defined]
        try:
            client = OpenAICompatClient(self.settings)
            with self.assertRaises(LLMError):
                client.chat([Message.user("hi")], stage="plan")
        finally:
            self.httpd.fail = False  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main(verbosity=2)
