"""Offline trace recorder.

Each `ask()` writes one JSON trace with OTel-shaped spans:

    trace_id, question, runtime, duration_ms, cost_cny, total_tokens, spans[]

A span has ``name / duration_ms / attributes``. Node spans come from the custom
engine's step trace (LangGraph node spans have duration 0 there but carry the
node name), LLM spans come from the metered router's per-stage usage, and tool
spans come from the tool trace. This file is the source of truth; an OTLP or
Langfuse exporter can translate it without changing the agent.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceStore:
    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    def record(
        self,
        *,
        question: str,
        result: dict[str, Any],
        elapsed_ms: float,
        runtime: str,
    ) -> str:
        trace_id = uuid.uuid4().hex
        if not self.enabled:
            return trace_id

        usage = result.get("usage") or {}
        spans: list[dict[str, Any]] = [
            {
                "name": "agent.ask",
                "duration_ms": round(elapsed_ms, 2),
                "attributes": {
                    "runtime": runtime,
                    "question_chars": len(question),
                    "answer_chars": len(result.get("answer", "") or ""),
                    "interrupted": bool(result.get("interrupted")),
                },
            }
        ]
        for step in result.get("trace") or []:
            spans.append(
                {
                    "name": f"node.{step.get('node', '?')}",
                    "duration_ms": float(step.get("elapsed_ms", 0.0) or 0.0),
                    "attributes": {
                        "attempts": int(step.get("attempts", 1) or 1),
                        "updates": list(step.get("updates") or []),
                    },
                }
            )
        for stage, bucket in (usage.get("by_stage") or {}).items():
            spans.append(
                {
                    "name": f"llm.{stage}",
                    "duration_ms": 0.0,
                    "attributes": {
                        "calls": int(bucket.get("calls", 0) or 0),
                        "prompt_tokens": int(bucket.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(
                            bucket.get("completion_tokens", 0) or 0
                        ),
                        "cost_cny": float(bucket.get("cost_cny", 0.0) or 0.0),
                    },
                }
            )
        for tool in result.get("tool_trace") or []:
            spans.append(
                {
                    "name": f"tool.{tool.get('name', '?')}",
                    "duration_ms": float(tool.get("latency_ms", 0.0) or 0.0),
                    "attributes": {"ok": bool(tool.get("ok"))},
                }
            )

        record = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "runtime": runtime,
            "thread_id": result.get("thread_id"),
            "duration_ms": round(elapsed_ms, 2),
            "cost_cny": float(usage.get("total_cost_cny", 0.0) or 0.0),
            "total_tokens": sum(
                int(bucket.get("prompt_tokens", 0) or 0)
                + int(bucket.get("completion_tokens", 0) or 0)
                for bucket in (usage.get("by_stage") or {}).values()
            ),
            "skills": result.get("skills", []),
            "spans": spans,
        }
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f"{trace_id}.json.tmp"
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(directory / f"{trace_id}.json")
        return trace_id

    def stats(self) -> dict[str, Any]:
        try:
            count = len(list(self.root.glob("*/*.json"))) if self.root.exists() else 0
        except OSError:
            count = 0
        return {"enabled": self.enabled, "count": count, "root": str(self.root)}


def load_trace(root: Path, trace_id: str) -> dict[str, Any] | None:
    for path in Path(root).glob(f"*/{trace_id}.json"):
        return json.loads(path.read_text(encoding="utf-8"))
    return None
