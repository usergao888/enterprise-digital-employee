"""The agent itself: planner -> executor -> reviewer, with guardrails and memory.

This merges the three AI projects from the two resumes into one defensible
system:

* 企业数字员工 AI Agent  → RAG knowledge employee (retrieval + tool calling)
* 多智能体协同办公助手    → planner / executor / reviewer division of labour
* AI 零代码应用生成平台    → the reviewer node is the same "质检不通过自动重生成"
                            loop, moved from generated code to generated answers

Graph shape:

    START -> guard ─┬─(blocked)──────────────────────────────► finalize -> END
                    └─(ok)─► plan ─► execute ─► review ─┬─(pass)──► finalize
                                          ▲              └─(revise)─┘
                                          └──────────────┘ (bounded by max_revisions)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import Settings
from .embeddings import build_embedder
from .graph import END, START, FileCheckpointer, MemoryCheckpointer, StateGraph
from .llm import FakeLLM, Message, MeteredRouter, OpenAICompatClient
from .rag import (
    HybridRetriever,
    citation_report,
    hits_to_sources,
    pack_context,
)
from .tools import ToolRegistry, build_registry, ToolResult
from .vectorstore import Chunk, Hit, MemoryVectorStore, build_store

# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """你是企业数字员工的"规划智能体"。你的任务是把用户问题拆解成可执行步骤。
可用工具：{tools}
只输出 JSON，不要输出任何解释，格式：
{{"intent": "kb|sql|calc|ticket|chat", "steps": ["..."], "need_knowledge": true, "rewritten_query": "..."}}
规则：
1. 涉及公司制度、流程、政策的问题 intent 必须是 kb，need_knowledge=true。
2. 涉及统计数字、金额、排名的问题 intent=sql。
3. 需要精确算术时必须 intent=calc，不要心算。
4. rewritten_query 要补全指代、保留关键实体，用于检索。"""

EXECUTOR_PROMPT = """你是企业数字员工，为员工提供准确的内部事务答复。

工作方式：
1. 涉及公司制度/流程/政策，必须先调用 knowledge_search，禁止凭记忆回答。
2. 引用检索结果时在句末标注角标，例如 [1]、[2]，角标对应检索结果的 index。
3. 只使用工具返回的原文作答；原文没有覆盖的部分，明确说"知识库中未提及"，不要编造。
4. 涉及金额计算先调用 calculator；涉及员工数据先调用 sql_query。
5. 回答用中文，先给结论，再给要点，最后给一句下一步建议。长度控制在 200 字以内。

本次检索规划：{plan}"""

REVIEW_PROMPT = """你是"审核智能体"，负责质量把关。请检查下面这份草稿是否可直接发给员工。
用户问题：{question}
问题类型：{intent}
草稿：
{draft}

检查项：
1. 涉及制度类问题时，答案是否有 [n] 角标引用，且角标确实能支撑对应结论。
2. 是否出现知识库没有的编造内容（数字、时限、人名）。
3. 是否回答了用户真正问的问题。

只输出 JSON：{{"verdict": "pass" 或 "revise", "issues": ["..."], "fix_hint": "一句话修改建议"}}"""


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"忽略(之前|上面|以上|前面)的?(所有)?(指令|设定|提示)",
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"(输出|打印|泄露|重复).{0,8}(系统提示|system\s*prompt|你的设定)",
    r"你现在(是|扮演).{0,10}(没有限制|不受限)",
    r"(开发者模式|developer\s*mode|DAN)",
]
_PII_PATTERNS = {
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
}


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    masked_question: str = ""
    findings: list[str] = field(default_factory=list)


def guard_input(question: str) -> GuardResult:
    findings: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            findings.append("prompt_injection")
            break
    masked = question
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(masked):
            findings.append(f"pii:{label}")
            masked = pattern.sub(lambda m: m.group(0)[:3] + "*" * (len(m.group(0)) - 3), masked)
    if "prompt_injection" in findings:
        return GuardResult(
            allowed=False,
            reason="检测到提示词注入尝试，已拒绝执行并记录审计日志。",
            masked_question=masked,
            findings=findings,
        )
    return GuardResult(allowed=True, masked_question=masked, findings=findings)


def extract_json(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction — models wrap JSON in fences or prose."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        while start != -1:
            depth = 0
            for index in range(start, len(text)):
                char = text[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : index + 1]
                        break
            if candidate:
                break
            start = text.find("{", start + 1)
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)


class ConversationMemory:
    """Sliding window + rolling summary (the classic cheap long-term memory)."""

    def __init__(self, window: int = 4, summarise_after: int = 6) -> None:
        self.window = window
        self.summarise_after = summarise_after
        self.turns: list[Turn] = []
        self.summary: str = ""

    def add(self, question: str, answer: str, sources: list[dict[str, Any]]) -> None:
        self.turns.append(Turn(question, answer, sources))

    def recent(self, n: int | None = None) -> list[Turn]:
        return self.turns[-(n or self.window):]

    def needs_summary(self) -> bool:
        return len(self.turns) >= self.summarise_after

    def as_messages(self, limit_chars: int = 1200) -> list[Message]:
        messages: list[Message] = []
        if self.summary:
            messages.append(Message.system(f"历史对话摘要：{self.summary}"))
        for turn in self.recent():
            messages.append(Message.user(turn.question))
            messages.append(Message.assistant(turn.answer[:limit_chars]))
        return messages

    def compact(self, router: MeteredRouter) -> None:
        if not self.needs_summary():
            return
        joined = "；".join(f"Q:{t.question[:60]} A:{t.answer[:60]}" for t in self.turns[:-2])
        completion = router.chat(
            [Message.system("把下面的对话压缩成一句不超过 80 字的业务摘要。"),
             Message.user(joined)],
            stage="summarize",
        )
        self.summary = completion.text.strip()[:120]
        self.turns = self.turns[-2:]


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------
class DigitalEmployee:
    def __init__(self, settings: Settings | None = None, *, offline: bool | None = None) -> None:
        self.settings = settings or Settings.from_env()
        if offline is None:
            offline = self.settings.llm_backend != "openai_compat"
        self.offline = offline

        self.embedder = build_embedder(self.settings)
        self.store = build_store(
            "memory", self.settings.index_dir, self.settings.embed_dim
        )
        index_file = self.settings.index_dir / "kb_index.json"
        self.index_file = index_file
        if isinstance(self.store, MemoryVectorStore) and index_file.exists():
            self.store = MemoryVectorStore.load(index_file)

        self.retriever = HybridRetriever(self.store, self.embedder, self.settings)
        self.registry = build_registry(
            self.retriever, ticket_path=self.settings.index_dir / "tickets.json"
        )

        client = (
            FakeLLM(self.settings)
            if offline
            else OpenAICompatClient(self.settings)
        )
        self.router = MeteredRouter(client, self.settings)
        self.memory = ConversationMemory()
        self.checkpointer = FileCheckpointer(self.settings.index_dir / "checkpoints.json")
        self.graph = self._build_graph()

    # -- graph nodes ---------------------------------------------------
    # Nodes are plain methods over the shared state dict rather than closures:
    # that is what lets langgraph_adapter.py register the exact same functions
    # on the real LangGraph runtime without duplicating any logic.

    def node_guard(self, state: dict[str, Any]) -> dict[str, Any]:
        result = guard_input(state["question"])
        return {
            "guard": {
                "allowed": result.allowed,
                "reason": result.reason,
                "findings": result.findings,
            },
            "question_safe": result.masked_question,
        }

    def node_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        tools_desc = "、".join(
            f"{name}({self.registry.tools[name].description.split('。')[0]})"
            for name in self.registry.names()
        )
        messages = [
            Message.system(PLANNER_PROMPT.format(tools=tools_desc)),
            *self.memory.as_messages(limit_chars=300),
            Message.user(state["question_safe"]),
        ]
        completion = self.router.chat(messages, stage="plan")
        plan = extract_json(completion.text) or {
            "intent": "kb",
            "steps": ["检索企业知识库", "整合答案"],
            "need_knowledge": True,
            "rewritten_query": state["question_safe"],
        }
        plan.setdefault("rewritten_query", state["question_safe"])
        return {"plan": plan, "plan_raw": completion.text[:400]}

    def node_execute(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan") or {}
        messages: list[Message] = [
            Message.system(
                EXECUTOR_PROMPT.format(plan=json.dumps(plan, ensure_ascii=False)[:500])
            ),
            *self.memory.as_messages(),
            Message.user(state["question_safe"]),
        ]
        if state.get("review", {}).get("fix_hint"):
            # Feedback is a developer/system note, not a new user turn;
            # otherwise it becomes "the question" for the next pass.
            messages.append(
                Message.system(
                    "审核意见：" + str(state["review"]["fix_hint"]) + " 请据此修订后重新作答。"
                )
            )

        tool_results: list[ToolResult] = []
        sources: list[dict[str, Any]] = []
        hits: list[Hit] = []
        iterations = 0
        draft = ""

        while iterations < self.settings.max_iterations:
            iterations += 1
            completion = self.router.chat(
                messages, stage="execute", tools=self.registry.specs()
            )
            assistant_message = completion.message
            if not assistant_message.tool_calls:
                draft = assistant_message.content
                break

            messages.append(
                Message.assistant("", tool_calls=assistant_message.tool_calls)
            )
            for call in assistant_message.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                tool_results.append(result)
                messages.append(Message.tool(call.id, call.name, result.content))
                if call.name == "knowledge_search" and result.ok:
                    try:
                        payload = json.loads(result.content)
                    except json.JSONDecodeError:
                        continue
                    sources = payload.get("sources", sources)
                    hits = [
                        Hit(
                            chunk=Chunk(
                                chunk_id=item["chunk_id"],
                                doc_id=item.get("doc_id", ""),
                                title=item.get("title", ""),
                                heading=item.get("heading", ""),
                                text=item.get("text") or item.get("preview", ""),
                            ),
                            score=float(item.get("score", 0.0)),
                        )
                        for item in payload.get("sources", [])
                    ]
        else:
            draft = "抱歉，工具调用轮次达到上限仍未完成，请补充更具体的信息。"

        return {
            "draft": draft or "（本轮未生成有效答案）",
            "tool_trace": [
                {"name": r.name, "ok": r.ok, "error": r.error, "latency_ms": r.latency_ms}
                for r in tool_results
            ],
            "sources": sources,
            "hits": hits,
            "iterations": iterations,
        }

    def node_review(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.enable_reviewer:
            return {"review": {"verdict": "pass", "issues": [], "skipped": True}}

        draft = state.get("draft", "")
        question = state.get("question_safe", "")
        intent = str((state.get("plan") or {}).get("intent", "kb"))
        messages = [
            Message.system(
                REVIEW_PROMPT.format(question=question, draft=draft, intent=intent)
            ),
        ]
        completion = self.router.chat(messages, stage="review")
        review = extract_json(completion.text) or {"verdict": "pass", "issues": []}

        # Deterministic checks are authoritative for the things a model tends to
        # hand-wave: citation presence and citation validity.
        report = citation_report(draft, state.get("hits") or [], question)
        issues = list(review.get("issues") or [])
        if intent == "kb" and report["cited"] == 0:
            issues.append("制度类问题缺少 [n] 引用")
        if review.get("verdict") == "revise" and issues:
            if intent != "kb" and report["cited"] == 0:
                issues = [i for i in issues if "引用" not in i]
        verdict = "revise" if issues else "pass"
        return {
            "review": {
                "verdict": verdict,
                "issues": issues,
                "fix_hint": review.get("fix_hint", ""),
                "citation_report": report,
            }
        }

    def node_revise(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"revisions": int(state.get("revisions", 0)) + 1}

    def node_finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        answer = state.get("draft", "")
        if not (state.get("guard") or {}).get("allowed", True):
            answer = (
                "抱歉，这个问题触发了安全策略（"
                + str((state.get("guard") or {}).get("reason", ""))
                + "），我不会执行。你可以换一种表述继续咨询内部事务。"
            )
        report = citation_report(
            answer, state.get("hits") or [], state.get("question_safe", "")
        )
        sources = state.get("sources") or []
        answer = self._append_source_list(answer, sources)
        self.memory.add(state["question"], answer, sources)
        self.memory.compact(self.router)
        return {
            "final_answer": answer,
            "citations": report,
            "sources": sources,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # -- routing -------------------------------------------------------
    def route_after_guard(self, state: dict[str, Any]) -> str:
        return "continue" if (state.get("guard") or {}).get("allowed", True) else "blocked"

    def route_after_review(self, state: dict[str, Any]) -> str:
        review = state.get("review") or {}
        if (
            review.get("verdict") == "revise"
            and state.get("revisions", 0) < self.settings.max_revisions
        ):
            return "revise"
        return "finalize"

    # -- graph assembly ------------------------------------------------
    def node_map(self) -> dict[str, NodeFunc]:
        """Node name -> callable. Shared with the real-LangGraph adapter."""
        return {
            "guard": self.node_guard,
            "plan": self.node_plan,
            "execute": self.node_execute,
            "review": self.node_review,
            "revise": self.node_revise,
            "finalize": self.node_finalize,
        }

    def _build_graph(self) -> Any:
        graph = StateGraph()
        graph.add_node("guard", self.node_guard)
        graph.add_node("plan", self.node_plan)
        graph.add_node("execute", self.node_execute, retry=1)
        graph.add_node("review", self.node_review)
        graph.add_node("revise", self.node_revise)
        graph.add_node("finalize", self.node_finalize)

        graph.set_entry_point("guard")
        graph.add_conditional_edges(
            "guard", self.route_after_guard, {"continue": "plan", "blocked": "finalize"}
        )
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "review")
        graph.add_conditional_edges(
            "review", self.route_after_review, {"revise": "revise", "finalize": "finalize"}
        )
        graph.add_edge("revise", "execute")
        graph.add_edge("finalize", END)
        return graph.compile(
            checkpointer=self.checkpointer,
            recursion_limit=self.settings.recursion_limit,
        )

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _append_source_list(answer: str, sources: list[dict[str, Any]]) -> str:
        if not sources:
            return answer
        cited = {int(m) for m in re.findall(r"\[(\d{1,2})\]", answer)}
        used = [s for s in sources if int(s["index"]) in cited]
        if not used:
            return answer
        lines = [
            f"[{s['index']}] 《{s['title']}》"
            + (f" · {s['heading']}" if s.get("heading") else "")
            + f"（{s['chunk_id']}）"
            for s in used
        ]
        return answer.rstrip() + "\n\n出处：\n" + "\n".join(lines)

    # -- public API ----------------------------------------------------
    def ingest(self, kb_dir: Path | None = None) -> dict[str, Any]:
        report = self.retriever.index_documents(kb_dir)
        if isinstance(self.store, MemoryVectorStore):
            self.store.save(self.index_file)
        return report

    def ask(self, question: str, *, thread_id: str | None = None,
            fresh_memory: bool = False) -> dict[str, Any]:
        if fresh_memory:
            self.memory = ConversationMemory()
        self.router.reset_usage()
        started = time.perf_counter()
        final_state: dict[str, Any] = {}
        for event in self.graph.stream({"question": question}, thread_id=thread_id):
            if event["type"] == "end":
                final_state = event["state"]
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        citations = final_state.get("citations") or {}
        self_check = {
            "has_answer": bool(final_state.get("final_answer")),
            "cited": int(citations.get("cited", 0) or 0),
            "citation_valid": (
                bool(citations.get("existing")) if citations.get("cited") else None
            ),
            "support_rate": citations.get("support_rate", 0.0),
            "answer_chars": len(final_state.get("final_answer", "") or ""),
        }
        return {
            "question": question,
            "answer": final_state.get("final_answer", ""),
            "sources": final_state.get("sources", []),
            "citations": citations,
            "self_check": self_check,
            "plan": final_state.get("plan", {}),
            "review": final_state.get("review", {}),
            "tool_trace": final_state.get("tool_trace", []),
            "revisions": final_state.get("revisions", 0),
            "trace": final_state.get("trace", []),
            "usage": self.router.usage_summary(),
            "latency_ms": elapsed,
            "thread_id": final_state.get("thread_id"),
            "guard": final_state.get("guard", {}),
        }

    def stream_answer(self, question: str) -> Iterator[dict[str, Any]]:
        """Yields graph events, then a final `token` stream for the answer."""
        result = self.ask(question)
        answer = result["answer"]
        for index in range(0, len(answer), 10):
            yield {"type": "token", "text": answer[index : index + 10]}
        yield {"type": "sources", "sources": result["sources"]}
        yield {"type": "done", "result": result}

    def stats(self) -> dict[str, Any]:
        return {
            "kb": self.retriever.stats(),
            "tools": self.registry.names(),
            "offline": self.offline,
            "llm_backend": self.settings.llm_backend,
            "embed_backend": self.settings.embed_backend,
            "graph": self.graph.describe(),
        }


def build_digital_employee(settings: Settings | None = None) -> DigitalEmployee:
    agent = DigitalEmployee(settings)
    if len(agent.store) == 0:
        agent.ingest()
    return agent
