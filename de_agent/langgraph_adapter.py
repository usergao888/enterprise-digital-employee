"""Run the *same* node functions on the real LangGraph runtime.

The package ships its own `StateGraph` so it can run and be tested with zero
third-party dependencies. That engine deliberately mirrors LangGraph's call
shape, and `DigitalEmployee.node_map()` exposes the nodes as plain callables, so
swapping runtimes is a wiring change rather than a rewrite:

    from de_agent.langgraph_adapter import build_langgraph_agent
    app = build_langgraph_agent(agent)              # needs `pip install langgraph`
    result = app.invoke({"question": "年假有几天？"},
                        config={"configurable": {"thread_id": "u1"}})

What you gain from the real library: durable checkpointers (SQLite/Postgres),
`interrupt()` for human-in-the-loop approval on the mutating `create_ticket`
tool, and time-travel replay of any step. What you do not need to relearn: the
nodes, routing, prompts, retrieval and tools are identical.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the graph.

    `total=False` mirrors the fact that each node writes only part of the state;
    LangGraph merges the returned dict into the running state.
    """

    question: str
    question_safe: str
    guard: dict[str, Any]
    plan: dict[str, Any]
    plan_raw: str
    draft: str
    review: dict[str, Any]
    revisions: int
    sources: list[dict[str, Any]]
    hits: list[Any]
    tool_trace: list[dict[str, Any]]
    final_answer: str
    citations: dict[str, Any]
    iterations: int
    finished_at: str
    trace: list[dict[str, Any]]
    thread_id: str


def build_langgraph_agent(agent: Any) -> Any:
    """Compile this agent's nodes onto a real `langgraph.graph.StateGraph`."""
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END as LG_END
        from langgraph.graph import StateGraph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "需要先安装 langgraph：pip install 'langgraph>=0.2.28'"
        ) from exc

    nodes = agent.node_map()
    graph = StateGraph(AgentState)
    for name, func in nodes.items():
        graph.add_node(name, func)

    graph.set_entry_point("guard")
    graph.add_conditional_edges(
        "guard",
        agent.route_after_guard,
        {"continue": "plan", "blocked": "finalize"},
    )
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges(
        "review",
        agent.route_after_review,
        {"revise": "revise", "finalize": "finalize"},
    )
    graph.add_edge("revise", "execute")
    graph.add_edge("finalize", LG_END)
    return graph.compile(checkpointer=MemorySaver())


def ask_with_langgraph(agent: Any, question: str, thread_id: str = "default") -> dict[str, Any]:
    """Convenience wrapper: same output shape as `DigitalEmployee.ask`."""
    app = agent.langgraph_app or build_langgraph_agent(agent)
    agent.router.reset_usage()
    state = app.invoke(
        {"question": question}, config={"configurable": {"thread_id": thread_id}}
    )
    return {
        "question": question,
        "answer": state.get("final_answer", ""),
        "sources": state.get("sources", []),
        "citations": state.get("citations", {}),
        "plan": state.get("plan", {}),
        "review": state.get("review", {}),
        "tool_trace": state.get("tool_trace", []),
        "revisions": state.get("revisions", 0),
        "usage": agent.router.usage_summary(),
        "thread_id": thread_id,
    }
