"""Permission modes for tool calls: read_only / ask / auto.

`ask` is the useful default for production: mutating tools are parked for human
approval. `auto` preserves the demo behaviour, `read_only` hard-denies writes.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_MODES = {"read_only", "ask", "auto"}


@dataclass(frozen=True)
class PermissionDecision:
    allow: bool
    requires_confirmation: bool = False
    reason: str = ""


class PermissionPolicy:
    def __init__(
        self,
        mode: str = "auto",
        *,
        allow_tools: set[str] | None = None,
        deny_tools: set[str] | None = None,
    ) -> None:
        mode = (mode or "auto").lower()
        if mode not in VALID_MODES:
            raise ValueError(f"permission mode must be one of {sorted(VALID_MODES)}")
        self.mode = mode
        self.allow_tools = allow_tools
        self.deny_tools = deny_tools or set()

    def decide(self, tool_name: str, mutating: bool) -> PermissionDecision:
        if tool_name in self.deny_tools:
            return PermissionDecision(False, False, "tool_denied")
        if self.allow_tools is not None and tool_name not in self.allow_tools:
            return PermissionDecision(False, False, "tool_not_allowlisted")
        if self.mode == "read_only" and mutating:
            return PermissionDecision(False, False, "read_only_mode")
        if self.mode == "ask" and mutating:
            return PermissionDecision(True, True, "ask_mode")
        return PermissionDecision(True, False, "auto_mode")
