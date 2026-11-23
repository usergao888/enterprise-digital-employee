"""Run the existing de_agent nodes on the real LangGraph runtime.

Phase 0: the custom `graph.py` stays as the zero-dependency fallback; this
module is only imported when `DE_GRAPH_RUNTIME=langgraph` (or `auto` with
`langgraph` installed).

Design notes
------------
* State keys are replaced wholesale, matching the custom engine's
  ``state = {**state, **update}``. Nodes currently return full lists
  (``tool_trace`` / ``sources``), so do **not** add list reducers here unless
  the node contract changes.
* ``interrupt()`` is called in ``DigitalEmployee.node_approval``, never inside
  ``execute``: a resumed node re-executes from the top, and the approval node
  has no side effects before ``interrupt()``.
* The checkpointer is SQLite (``langgraph-checkpoint-sqlite``) so an interrupted
  thread survives a process restart.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any, TypedDict

CHECKPOINT_FILENAME = "langgraph_checkpoints.sqlite"


class AgentState(TypedDict, total=False):
    """Shared state. ``total=False`` mirrors nodes that write only a slice."""

    question: str
    question_safe: str
    guard: dict[str, Any]
    plan: dict[str, Any]
    plan_raw: str
    skills: list[str]
    skill_instructions: str
    draft: str
    review: dict[str, Any]
    revisions: int
    sources: list[dict[str, Any]]
    hits: list[dict[str, Any]]
    retrieval: dict[str, Any]
    pending_actions: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    final_answer: str
    citations: dict[str, Any]
    iterations: int
    finished_at: str
    trace: list[dict[str, Any]]
    thread_id: str
    turn_id: str
    finalized_turn: str
    approval: dict[str, Any]


def langgraph_available() -> bool:
    """True only if the runtime + SQLite checkpointer are both importable."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        from langgraph.graph import StateGraph  # noqa: F401
        from langgraph.types import Command, interrupt  # noqa: F401
    except ImportError:
        return False
    return True


def _config(agent: Any, thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": int(agent.settings.recursion_limit),
    }


def _checkpoint_path(agent: Any) -> Path:
    return Path(agent.settings.index_dir) / CHECKPOINT_FILENAME


def build_langgraph_agent(agent: Any) -> Any:
    """Compile ``DigitalEmployee``'s nodes on LangGraph.

    The returned app is also cached on ``agent.langgraph_app``; the SQLite
    connection is kept on ``agent._langgraph_conn`` for the agent's lifetime.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    path = _checkpoint_path(agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    # WAL + busy_timeout: tests open a second agent on the same DB to verify
    # checkpoint survival across a restart.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(AgentState)
    for name, func in agent.node_map().items():
        graph.add_node(name, func)
    # HITL is a real graph node in the LangGraph runtime; the custom engine
    # keeps its application-level confirmation path.
    graph.add_node("approval", agent.node_approval)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges(
        "guard",
        agent.route_after_guard,
        {"continue": "plan", "blocked": "finalize"},
    )
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        agent.route_after_execute,
        {"approval": "approval", "review": "review"},
    )
    graph.add_edge("approval", "finalize")
    graph.add_conditional_edges(
        "review",
        agent.route_after_review,
        {"revise": "revise", "finalize": "finalize"},
    )
    graph.add_edge("revise", "execute")
    graph.add_edge("finalize", END)

    app = graph.compile(checkpointer=checkpointer)
    agent.langgraph_app = app
    agent._langgraph_conn = conn
    return app


def _values(app: Any, config: dict[str, Any], thread_id: str,
            trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    snapshot = app.get_state(config)
    state = dict(snapshot.values or {})
    state["thread_id"] = state.get("thread_id") or thread_id
    state["__next__"] = tuple(snapshot.next or ())
    if trace is not None:
        state["trace"] = trace
    return state


def invoke_langgraph(
    agent: Any,
    app: Any,
    question: str,
    thread_id: str,
    turn_id: str | None = None,
) -> dict[str, Any]:
    """Run one turn and return the final/interrupted state.

    ``state["__next__"]`` is non-empty when the graph stopped at a breakpoint
    (currently only the approval node).
    """
    turn_id = turn_id or uuid.uuid4().hex
    config = _config(agent, thread_id)
    trace: list[dict[str, Any]] = []

    for chunk in app.stream(
        {"question": question, "thread_id": thread_id, "turn_id": turn_id},
        config=config,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node == "__interrupt__":
                trace.append(
                    {
                        "node": "__interrupt__",
                        "updates": ["pending_actions"],
                        "elapsed_ms": 0.0,
                        "attempts": 1,
                    }
                )
            else:
                trace.append(
                    {
                        "node": node,
                        "updates": sorted((update or {}).keys()),
                        "elapsed_ms": 0.0,
                        "attempts": 1,
                    }
                )

    return _values(app, config, thread_id, trace)


def resume_langgraph(
    agent: Any,
    app: Any,
    thread_id: str,
    decision: bool | dict[str, Any],
) -> dict[str, Any]:
    """Resume an interrupted thread with the user's decision."""
    from langgraph.types import Command

    config = _config(agent, thread_id)
    app.invoke(Command(resume=decision), config=config)
    return _values(app, config, thread_id)


def get_langgraph_values(agent: Any, app: Any, thread_id: str) -> dict[str, Any]:
    return _values(app, _config(agent, thread_id), thread_id)


def ask_with_langgraph(agent: Any, question: str, thread_id: str = "default") -> dict[str, Any]:
    """Compatibility wrapper with the same result shape as ``DigitalEmployee.ask``."""
    app = agent.langgraph_app or build_langgraph_agent(agent)
    state = invoke_langgraph(agent, app, question, thread_id)
    return {
        "question": question,
        "answer": state.get("final_answer") or state.get("draft", ""),
        "sources": state.get("sources", []),
        "citations": state.get("citations", {}),
        "plan": state.get("plan", {}),
        "review": state.get("review", {}),
        "tool_trace": state.get("tool_trace", []),
        "revisions": state.get("revisions", 0),
        "pending_actions": state.get("pending_actions", []),
        "interrupted": bool(state.get("__next__")),
        "usage": agent.router.usage_summary(),
        "thread_id": thread_id,
    }
