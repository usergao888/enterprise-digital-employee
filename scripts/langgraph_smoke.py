"""Verify the real LangGraph API before migrating de_agent.

Run with the venv created by scripts/bootstrap-langgraph.ps1:

    .venv\\Scripts\\python.exe scripts\\langgraph_smoke.py

What it checks:
  1. imports used by the migration (StateGraph / MemorySaver / interrupt / Command)
  2. interrupt -> get_state().next -> Command(resume=...) round trip
  3. SqliteSaver durability: interrupt, resume, close, reopen, get_state

Exit code 0 means every required check passed.
"""

from __future__ import annotations

import importlib.metadata as metadata
import sqlite3
import tempfile
from pathlib import Path
from typing import TypedDict

FAILURES: list[str] = []


def ok(label: str, detail: str = "") -> None:
    print(f"[OK]   {label}" + (f" - {detail}" if detail else ""))


def check(label: str, fn) -> None:
    try:
        fn()
        ok(label)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the smoke run
        FAILURES.append(label)
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")


def print_versions() -> None:
    print("installed versions:")
    for pkg in ("langgraph", "langgraph-checkpoint-sqlite", "numpy"):
        try:
            print(f"  {pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"  {pkg}: NOT INSTALLED")


class State(TypedDict, total=False):
    value: int
    approved: bool


def build_graph(checkpointer):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def start_node(state: State) -> dict:
        return {"value": 1}

    def approval_node(state: State) -> dict:
        decision = interrupt({"question": "approve?", "value": state.get("value")})
        return {"approved": bool(decision)}

    builder = StateGraph(State)
    builder.add_node("start", start_node)
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "start")
    builder.add_edge("start", "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=checkpointer)


def check_imports() -> None:
    from langgraph.checkpoint.memory import MemorySaver  # noqa: F401
    from langgraph.graph import END, START, StateGraph  # noqa: F401
    from langgraph.types import Command, interrupt  # noqa: F401


def check_interrupt_roundtrip() -> None:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    app = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "smoke-memory"}}

    first = app.invoke({"value": 0}, config=config)
    if "__interrupt__" not in first:
        raise AssertionError(f"invoke did not stop at interrupt: {first!r}")

    snapshot = app.get_state(config)
    if tuple(snapshot.next) != ("approval",):
        raise AssertionError(f"unexpected next={snapshot.next!r}")

    second = app.invoke(Command(resume=True), config=config)
    if second.get("approved") is not True:
        raise AssertionError(f"resume did not set approved=True: {second!r}")

    print("       interrupt -> get_state.next -> Command(resume) works")


def check_sqlite_durability() -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    with tempfile.TemporaryDirectory(prefix="lg-smoke-") as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        config = {"configurable": {"thread_id": "smoke-sqlite"}}

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            app = build_graph(SqliteSaver(conn))
            first = app.invoke({"value": 0}, config=config)
            if "__interrupt__" not in first:
                raise AssertionError(f"sqlite saver did not interrupt: {first!r}")
            app.invoke(Command(resume=True), config=config)
        finally:
            conn.close()

        # Reopen: the checkpoint must survive process/connection restart.
        conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            app2 = build_graph(SqliteSaver(conn2))
            snapshot = app2.get_state(config)
            if snapshot.values.get("approved") is not True:
                raise AssertionError(f"state not durable after reopen: {snapshot.values!r}")
        finally:
            conn2.close()

        print("       SqliteSaver reopen + get_state keeps the approved state")


def main() -> int:
    print("== LangGraph smoke test ==")
    print_versions()
    print()

    check("imports (StateGraph, MemorySaver, interrupt, Command)", check_imports)
    check("interrupt / get_state / Command(resume)", check_interrupt_roundtrip)
    check("SqliteSaver durability", check_sqlite_durability)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} check(s)): {', '.join(FAILURES)}")
        print("Send this output back; the migration patch will target the installed versions.")
        return 1

    print("RESULT: LANGGRAPH_SMOKE_OK")
    print("Send the version lines + this result back, then write the Phase 0 adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
