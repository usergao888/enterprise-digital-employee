"""Zero-dependency HTTP server (stdlib only) with JSON + SSE endpoints.

    python -m de_agent serve --port 8710

Endpoints:
    GET  /              minimal chat UI
    GET  /health        liveness
    GET  /stats         knowledge base / tools / graph description
    POST /ask           {"question": "..."} -> structured answer
    POST /chat/stream   {"question": "..."} -> text/event-stream of tokens
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agents import DigitalEmployee

INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>企业数字员工</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:840px;margin:40px auto;
      padding:0 16px;color:#1a1a1a;background:#fafafa}
 h1{font-size:20px;margin-bottom:4px} .sub{color:#666;font-size:13px;margin-bottom:24px}
 #log{border:1px solid #e3e3e3;border-radius:10px;background:#fff;padding:16px;min-height:320px;
      white-space:pre-wrap;line-height:1.65;font-size:14px}
 .q{color:#0b5fff;font-weight:600;margin-top:12px} .a{margin-top:4px}
 .src{color:#888;font-size:12px;margin-top:6px}
 form{display:flex;gap:8px;margin-top:14px}
 input{flex:1;padding:10px 12px;border:1px solid #d5d5d5;border-radius:8px;font-size:14px}
 button{padding:10px 18px;border:0;border-radius:8px;background:#0b5fff;color:#fff;font-size:14px}
 button:disabled{background:#9dbcf7}
</style></head><body>
<h1>企业数字员工 Agent</h1>
<div class="sub">RAG + Tool Calling + 多智能体编排 · 回答带原文出处</div>
<div id="log">你好，我可以回答制度问题、查数据、建工单。</div>
<form id="f"><input id="q" placeholder="例如：一线城市出差住宿能报多少？" autocomplete="off">
<button id="b">发送</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),b=document.getElementById('b');
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault(); if(!q.value.trim())return;
  const question=q.value.trim(); q.value=''; b.disabled=true;
  log.innerHTML += '\\n<span class="q">你：'+question+'</span>\\n<span class="a" id="live">思考中…</span>';
  const live=document.getElementById('live');
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question})});
    const d=await r.json();
    live.textContent='助手：'+(d.answer||d.error||'（无回答）');
    let src=''; (d.sources||[]).forEach(s=>{src+='\\n  ['+s.index+'] '+s.title+' · '+s.heading});
    if(src) live.innerHTML += '<span class="src">'+src.replace(/\\n/g,'<br>')+'</span>';
    if(d.usage) live.innerHTML += '<span class="src">\\n延迟 '+d.latency_ms+'ms · 成本 ¥'+
      Number(d.usage.total_cost_cny).toFixed(6)+' · 省 '+(100*Number(d.usage.saving_rate)).toFixed(0)+'%</span>';
  }catch(err){ live.textContent='助手：请求失败 '+err; }
  b.disabled=false; q.focus();
};
</script></body></html>
"""


class AgentHandler(BaseHTTPRequestHandler):
    agent: DigitalEmployee = None  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"
    server_version = "DigitalEmployee/0.3"

    # -- helpers -------------------------------------------------------
    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        if self.path and self.path.startswith(("/ask", "/chat")):
            print(f"[http] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    # -- routes --------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in {"/", "/index.html"}:
            self._send_html(INDEX_HTML)
        elif path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/stats":
            self._send_json(self.agent.stats())
        else:
            self._send_json({"error": "not found", "path": path}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/ask":
            body = self._read_json()
            question = str(body.get("question", "")).strip()
            if not question:
                self._send_json({"error": "question 不能为空"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                self._send_json(self.agent.ask(question))
            except Exception as exc:  # noqa: BLE001 - never kill the server
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/chat/stream":
            body = self._read_json()
            question = str(body.get("question", "")).strip()
            if not question:
                self._send_json({"error": "question 不能为空"}, HTTPStatus.BAD_REQUEST)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                for event in self.agent.stream_answer(question):
                    kind = event.get("type")
                    if kind == "token":
                        payload = {"type": "token", "text": event["text"]}
                    elif kind == "sources":
                        payload = {"type": "sources", "sources": event["sources"]}
                    else:
                        result = event.get("result", {})
                        payload = {
                            "type": "done",
                            "citations": result.get("citations", {}),
                            "usage": result.get("usage", {}),
                            "latency_ms": result.get("latency_ms"),
                        }
                    self.wfile.write(
                        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self._send_json({"error": "not found", "path": path}, HTTPStatus.NOT_FOUND)


def serve(agent: DigitalEmployee, host: str = "127.0.0.1", port: int = 8710) -> None:
    handler = type("BoundAgentHandler", (AgentHandler,), {"agent": agent})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"企业数字员工已启动： http://{host}:{port}")
    print("接口：GET /health  GET /stats  POST /ask  POST /chat/stream")
    print("按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭…")
    finally:
        httpd.server_close()


def serve_in_thread(agent: DigitalEmployee, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Used by tests to bind an ephemeral port."""
    handler = type("BoundAgentHandler", (AgentHandler,), {"agent": agent})
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread
