"""Persistent task harness: step budget, checkpointed progress, HITL resume.

This is the runtime *around* the agent graph: it drives a list of steps, records
progress after each step, and stops on a cost/token/wall-clock budget. It does
not replace LangGraph's checkpointer; the two persist different things:

* LangGraph checkpoint: the state of one interrupted graph turn;
* harness state.json: which task step we are on, and what it cost so far.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class RunBudget:
    max_steps: int = 20
    max_cost_cny: float = 0.05
    max_tokens: int = 20000
    max_wall_clock_s: float = 120.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunBudget":
        return cls(
            max_steps=int(data.get("max_steps", 20)),
            max_cost_cny=float(data.get("max_cost_cny", 0.05)),
            max_tokens=int(data.get("max_tokens", 20000)),
            max_wall_clock_s=float(data.get("max_wall_clock_s", 120.0)),
        )


@dataclass
class TaskStep:
    name: str
    question: str


@dataclass
class HarnessResult:
    run_id: str
    status: str
    steps_done: int
    total_steps: int
    results: list[dict[str, Any]] = field(default_factory=list)
    cost_cny: float = 0.0
    tokens: int = 0
    error: str = ""
    pending_action: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _usage_tokens(usage: dict[str, Any]) -> int:
    total = 0
    for bucket in (usage.get("by_stage") or {}).values():
        total += int(bucket.get("prompt_tokens", 0))
        total += int(bucket.get("completion_tokens", 0))
    return total


class AgentHarness:
    """Run a task list with a budget and durable progress."""

    def __init__(self, agent: Any, runs_dir: Path | None = None) -> None:
        self.agent = agent
        self.runs_dir = Path(runs_dir or (agent.settings.index_dir / "harness"))
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ----------------------------------------------------
    def run(
        self,
        steps: Sequence[TaskStep],
        *,
        run_id: str | None = None,
        budget: RunBudget | None = None,
    ) -> HarnessResult:
        if not steps:
            raise ValueError("steps 不能为空")
        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        budget = budget or RunBudget()
        state: dict[str, Any] = {
            "run_id": run_id,
            "thread_id": f"harness-{run_id}",
            "steps": [asdict(step) for step in steps],
            "next_index": 0,
            "results": [],
            "status": "running",
            "budget": asdict(budget),
            "cost_cny": 0.0,
            "tokens": 0,
            "started_at": time.time(),
            "elapsed_s": 0.0,
            "error": "",
            "pending_action": None,
            "pending_step_index": None,
        }
        self._save(state)
        return self._continue(state)

    def resume(
        self,
        run_id: str,
        *,
        approved: bool = True,
        budget: RunBudget | None = None,
    ) -> HarnessResult:
        state = self._load(run_id)
        if budget is not None:
            state["budget"] = asdict(budget)
        if state.get("status") == "awaiting_approval":
            self._resolve_approval(state, approved=approved)
        return self._continue(state)

    def status(self, run_id: str) -> dict[str, Any]:
        return self._load(run_id)

    # -- internals -----------------------------------------------------
    def _resolve_approval(self, state: dict[str, Any], *, approved: bool) -> None:
        action = state.get("pending_action") or {}
        action_id = str(action.get("action_id") or "")
        if approved:
            outcome = self.agent.confirm(action_id)
            answer = outcome.get("result", "")
            ok = bool(outcome.get("ok"))
        else:
            self.agent.cancel(action_id)
            answer = "用户取消，未写入任何数据。"
            ok = True
        index = int(state.get("pending_step_index") or state["next_index"])
        step = state["steps"][index]
        record = {
            "index": index,
            "name": step["name"],
            "question": step["question"],
            "answer": answer,
            "status": "approved" if approved else "cancelled",
            "ok": ok,
            "usage": {},
        }
        for item in state["results"]:
            if item.get("index") == index and item.get("status") == "awaiting_approval":
                item.update(record)
                break
        else:
            state["results"].append(record)
        state["next_index"] = index + 1
        state["pending_action"] = None
        state["pending_step_index"] = None
        state["status"] = "running"
        self._save(state)

    def _continue(self, state: dict[str, Any]) -> HarnessResult:
        budget = RunBudget.from_dict(state.get("budget") or {})
        started = time.time()
        base_elapsed = float(state.get("elapsed_s", 0.0))
        steps = state["steps"]

        while state["next_index"] < len(steps):
            elapsed = base_elapsed + (time.time() - started)
            reason = self._budget_reason(state, budget, elapsed)
            if reason:
                state["status"] = "budget_exceeded"
                state["error"] = reason
                break

            index = int(state["next_index"])
            step = steps[index]
            state["status"] = "running"
            try:
                result = self.agent.ask(
                    step["question"], thread_id=state["thread_id"]
                )
            except Exception as exc:  # noqa: BLE001 - harness reports, never hides
                state["status"] = "failed"
                state["error"] = f"{type(exc).__name__}: {exc}"
                break
            usage = result.get("usage") or {}
            state["cost_cny"] = round(
                float(state["cost_cny"]) + float(usage.get("total_cost_cny", 0.0)), 6
            )
            state["tokens"] = int(state["tokens"]) + _usage_tokens(usage)
            state["results"].append(
                {
                    "index": index,
                    "name": step["name"],
                    "question": step["question"],
                    "answer": result.get("answer", ""),
                    "skills": result.get("skills", []),
                    "tool_trace": result.get("tool_trace", []),
                    "status": "awaiting_approval"
                    if result.get("pending_actions")
                    else "done",
                    "usage": {
                        "cost_cny": usage.get("total_cost_cny", 0.0),
                        "tokens": _usage_tokens(usage),
                    },
                }
            )

            if result.get("pending_actions"):
                state["status"] = "awaiting_approval"
                state["pending_action"] = result["pending_actions"][0]
                state["pending_step_index"] = index
                state["elapsed_s"] = round(
                    base_elapsed + (time.time() - started), 2
                )
                break

            state["next_index"] = index + 1
            state["elapsed_s"] = round(base_elapsed + (time.time() - started), 2)
            self._save(state)

        else:
            state["status"] = "completed"

        state["elapsed_s"] = round(base_elapsed + (time.time() - started), 2)
        self._save(state)
        return self._result(state)

    def _budget_reason(
        self, state: dict[str, Any], budget: RunBudget, elapsed: float
    ) -> str | None:
        if int(state["next_index"]) >= budget.max_steps:
            return f"max_steps={budget.max_steps}"
        if float(state["cost_cny"]) >= budget.max_cost_cny:
            return f"max_cost_cny={budget.max_cost_cny}"
        if int(state["tokens"]) >= budget.max_tokens:
            return f"max_tokens={budget.max_tokens}"
        if elapsed >= budget.max_wall_clock_s:
            return f"max_wall_clock_s={budget.max_wall_clock_s}"
        return None

    def _result(self, state: dict[str, Any]) -> HarnessResult:
        return HarnessResult(
            run_id=state["run_id"],
            status=state["status"],
            steps_done=len(state["results"]),
            total_steps=len(state["steps"]),
            results=state["results"],
            cost_cny=float(state["cost_cny"]),
            tokens=int(state["tokens"]),
            error=state.get("error", ""),
            pending_action=state.get("pending_action"),
        )

    # -- persistence ---------------------------------------------------
    def _dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _load(self, run_id: str) -> dict[str, Any]:
        path = self._dir(run_id) / "state.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, state: dict[str, Any]) -> None:
        directory = self._dir(state["run_id"])
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "state.json.tmp"
        tmp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(directory / "state.json")
        (directory / "progress.md").write_text(
            self._progress(state), encoding="utf-8"
        )

    def _progress(self, state: dict[str, Any]) -> str:
        lines = [
            f"# Harness run {state['run_id']}",
            "",
            f"- status: **{state['status']}**",
            f"- steps: {state['next_index']}/{len(state['steps'])}",
            f"- cost_cny: {state['cost_cny']}",
            f"- tokens: {state['tokens']}",
        ]
        if state.get("error"):
            lines.append(f"- stop_reason: {state['error']}")
        lines.append("")
        done = {
            item.get("index")
            for item in state.get("results", [])
            if item.get("status") != "awaiting_approval"
        }
        pending = {
            item.get("index")
            for item in state.get("results", [])
            if item.get("status") == "awaiting_approval"
        }
        for index, step in enumerate(state["steps"]):
            mark = "x" if index in done else ("?" if index in pending else " ")
            lines.append(f"- [{mark}] {index + 1}. {step['name']} — {step['question']}")
        return "\n".join(lines) + "\n"
