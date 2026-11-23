"""Context engineering: budget, tool-output pruning, and a per-thread scratchpad.

The goal is not "fit more text"; it is to keep the working context predictable:
each section has a token budget, large tool results are reduced before they enter
the next model call, and durable notes live outside the window.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import estimate_tokens

SECTION_SHARES: dict[str, float] = {
    "system": 0.15,
    "skills": 0.10,
    "memory": 0.20,
    "retrieval": 0.30,
    "tool": 0.20,
    "history": 0.05,
}

_TRUNCATION_MARK = "\n…[已按上下文预算截断]"


@dataclass
class ContextBudget:
    """Token budget split by section; shares are renormalised on demand."""

    total_tokens: int = 6000
    shares: dict[str, float] = field(default_factory=lambda: dict(SECTION_SHARES))

    def limit(self, section: str) -> int:
        share = self.shares.get(section, 0.10)
        return max(64, int(self.total_tokens * share))

    def fit(self, text: str, section: str) -> str:
        if not text:
            return text
        limit = self.limit(section)
        if estimate_tokens(text) <= limit:
            return text
        # Char/token ratio is stable per string; allocate chars proportionally.
        chars = max(1, int(len(text) * limit / max(1, estimate_tokens(text))))
        return text[:chars].rstrip() + _TRUNCATION_MARK


def _shrink_json(value: Any, str_limit: int, list_limit: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _shrink_json(item, str_limit, list_limit)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _shrink_json(item, str_limit, list_limit) for item in value[:list_limit]
        ]
    if isinstance(value, str):
        return value[:str_limit] + ("…" if len(value) > str_limit else "")
    return value


def _prune_json(text: str, parsed: Any, max_chars: int) -> str:
    """Always return valid JSON, never a truncated JSON fragment."""
    for str_limit, list_limit in ((400, 5), (200, 2), (80, 1)):
        encoded = json.dumps(
            _shrink_json(parsed, str_limit, list_limit),
            ensure_ascii=False,
            default=str,
        )
        if len(encoded) <= max_chars:
            return encoded
    summary = {
        "truncated": True,
        "original_chars": len(text),
        "keys": list(parsed) if isinstance(parsed, dict) else None,
        "preview": text[: max(40, max_chars // 2)],
    }
    return json.dumps(summary, ensure_ascii=False)


def prune_tool_output(text: str, max_chars: int = 4000) -> str:
    """Keep schema/head of a tool result; never let one result eat the window."""
    text = text or ""
    if len(text) <= max_chars:
        return text

    loaded = True
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        loaded = False
        parsed = None
    if loaded:
        return _prune_json(text, parsed, max_chars)

    head = text[: max(1, max_chars - 140)]
    tail = text[-80:]
    result = head + f"\n…[原始工具输出 {len(text)} 字符，已裁剪]…\n" + tail
    return result[:max_chars]


class Scratchpad:
    """Durable per-thread notes outside the model window."""

    def __init__(self, path: Path, max_chars: int = 4000) -> None:
        self.path = Path(path)
        self.max_chars = max(400, max_chars)

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".md.tmp")
        tmp.write_text(text[-self.max_chars :], encoding="utf-8")
        tmp.replace(self.path)

    def append_turn(
        self,
        question: str,
        answer: str,
        tool_names: list[str] | None = None,
    ) -> None:
        block = (
            f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- Q: {question[:300]}\n"
            f"- A: {(answer or '')[:900]}\n"
            f"- tools: {', '.join(tool_names or []) or 'none'}\n"
        )
        self.write(self.read() + block)


class ContextManager:
    """Budget + pruning + scratchpad facade used by the agent."""

    def __init__(
        self,
        index_dir: Path,
        *,
        total_tokens: int = 6000,
        tool_output_max_chars: int = 4000,
        scratchpad_max_chars: int = 4000,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.budget = ContextBudget(total_tokens=total_tokens)
        self.tool_output_max_chars = tool_output_max_chars
        self.scratchpad_max_chars = scratchpad_max_chars
        self.root = Path(index_dir) / "scratchpad"

    def fit(self, text: str, section: str) -> str:
        return self.budget.fit(text, section) if self.enabled else text

    def prune_tool_output(self, text: str) -> str:
        if not self.enabled:
            return text
        return prune_tool_output(text, self.tool_output_max_chars)

    def scratchpad(self, thread_id: str) -> Scratchpad:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id or "default")[:80]
        return Scratchpad(
            self.root / f"{safe}.md", max_chars=self.scratchpad_max_chars
        )

    def record_turn(
        self,
        thread_id: str,
        question: str,
        answer: str,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.enabled or not thread_id:
            return
        names = [str(item.get("name", "")) for item in (tool_trace or [])]
        try:
            self.scratchpad(thread_id).append_turn(question, answer, names)
        except OSError:
            pass

    def stats(self) -> dict[str, Any]:
        try:
            count = len(list(self.root.glob("*.md"))) if self.root.exists() else 0
        except OSError:
            count = 0
        return {
            "enabled": self.enabled,
            "total_tokens": self.budget.total_tokens,
            "tool_output_max_chars": self.tool_output_max_chars,
            "scratchpads": count,
        }
