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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import Settings
from .context import ContextManager
from .embeddings import build_embedder
from .graph import END, FileCheckpointer, StateGraph
from .llm import FakeLLM, Message, MeteredRouter, OpenAICompatClient
from .rag import HybridRetriever, citation_report
from .skills import SkillLibrary
from .tools import PendingAction, build_registry, ToolResult
from .tracing import TraceStore
from .vectorstore import Chunk, Hit, MemoryVectorStore, build_store

# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """你是企业数字员工的"规划智能体"。你的任务是把用户问题拆解成可执行步骤。
可用工具：{tools}
可用技能（只按需参考，不代表必须使用）：
{skills}
只输出 JSON，不要输出任何解释，格式：
{{"intent": "kb|sql|calc|ticket|chat", "steps": ["..."], "need_knowledge": true, "rewritten_query": "...", "sub_queries": ["..."]}}
规则：
1. 涉及公司制度、流程、政策的问题 intent 必须是 kb，need_knowledge=true。
2. 涉及统计数字、金额、排名的问题 intent=sql。
3. 需要精确算术时必须 intent=calc，不要心算。
4. rewritten_query 要补全指代、保留关键实体，用于检索。
5. 若问题包含两个及以上可独立检索的子问题（如"年假有几天，怎么申请"），
   拆成 2-3 条 sub_queries，每条都能单独拿去检索；单一问题给空数组 []。"""

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
            self.retriever,
            ticket_path=self.settings.index_dir / "tickets.json",
            settings=self.settings,
        )
        # Skills are procedural knowledge; tools stay in the registry above.
        self.skills = (
            SkillLibrary(self.settings.skills_dir)
            if self.settings.enable_skills
            else None
        )
        self.context = ContextManager(
            self.settings.index_dir,
            total_tokens=self.settings.context_total_tokens,
            tool_output_max_chars=self.settings.tool_output_max_chars,
            scratchpad_max_chars=self.settings.scratchpad_max_chars,
            enabled=self.settings.enable_context_budget,
        )
        self.tracer = TraceStore(
            self.settings.index_dir / "traces",
            enabled=self.settings.enable_tracing,
        )

        client = (
            FakeLLM(self.settings)
            if offline
            else OpenAICompatClient(self.settings)
        )
        self.router = MeteredRouter(client, self.settings)
        self.memory = ConversationMemory()
        self.checkpointer = FileCheckpointer(self.settings.index_dir / "checkpoints.json")
        self.runtime = self._resolve_runtime(self.settings.graph_runtime)
        # The custom graph is always compiled: it is the fallback runtime and
        # the source of `stats()["graph"]`.
        self.graph = self._build_graph()
        # Real LangGraph runtime (only when selected and importable).
        self.langgraph_app: Any = None
        self._langgraph_conn: Any = None
        self._pending_threads_path = self.settings.index_dir / "pending_threads.json"
        self._pending_threads: dict[str, str] = self._load_pending_threads()
        if self.runtime == "langgraph":
            self.langgraph_app = self._build_langgraph()

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
        skills_desc = (
            self.skills.describe_for_prompt() if self.skills else "（未启用技能）"
        )
        skills_desc = self.context.fit(skills_desc, "skills")
        selected = (
            self.skills.select(state.get("question_safe", ""), limit=1)
            if self.skills
            else []
        )
        messages = [
            Message.system(
                PLANNER_PROMPT.format(tools=tools_desc, skills=skills_desc)
            ),
            *self.memory.as_messages(limit_chars=300),
            Message.user(state["question_safe"]),
        ]
        completion = self.router.chat(messages, stage="plan")
        plan = extract_json(completion.text) or {
            "intent": "kb",
            "steps": ["检索企业知识库", "整合答案"],
            "need_knowledge": True,
            "rewritten_query": state["question_safe"],
            "sub_queries": [],
        }
        plan.setdefault("rewritten_query", state["question_safe"])
        plan.setdefault("sub_queries", [])
        if not isinstance(plan.get("sub_queries"), list):
            plan["sub_queries"] = []
        return {
            "plan": plan,
            "plan_raw": completion.text[:400],
            "skills": [skill.name for skill in selected],
            "skill_instructions": (
                self.skills.instructions_for(selected) if self.skills else ""
            ),
        }

    def node_execute(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan") or {}
        plan_text = self.context.fit(
            json.dumps(plan, ensure_ascii=False), "system"
        )
        messages: list[Message] = [
            Message.system(EXECUTOR_PROMPT.format(plan=plan_text[:900])),
            *self.memory.as_messages(),
            Message.user(state["question_safe"]),
        ]
        if state.get("skill_instructions"):
            # Progressive disclosure step 2: body only for the selected skill.
            messages.insert(
                1,
                Message.system(
                    "适用技能说明：\n"
                    + self.context.fit(str(state["skill_instructions"]), "skills")
                ),
            )
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
        retrieval: dict[str, Any] = {}
        pending_actions: list[dict[str, Any]] = []
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
                messages.append(
                    Message.tool(
                        call.id,
                        call.name,
                        self.context.prune_tool_output(result.content),
                    )
                )
                if result.pending_action_id:
                    action = self.registry.pending.get(result.pending_action_id)
                    if action is not None:
                        pending_actions.append(action.as_dict())
                    continue
                if call.name == "knowledge_search" and result.ok:
                    try:
                        payload = json.loads(result.content)
                    except json.JSONDecodeError:
                        continue
                    sources = payload.get("sources", sources)
                    retrieval = {
                        "queries": payload.get("queries", []),
                        "decomposed": payload.get("decomposed", False),
                        "decomposed_by": payload.get("decomposed_by", "none"),
                    }
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

            if pending_actions:
                first = pending_actions[0]
                draft = (
                    f"这个动作会修改数据，我已生成待确认任务 {first['action_id']}"
                    f"（工具 {first['tool']}）。**确认之后才会真正执行**，"
                    "请回复确认，或调用 confirm 接口传入 action_id。"
                )
                break
        else:
            draft = "抱歉，工具调用轮次达到上限仍未完成，请补充更具体的信息。"

        return {
            "draft": draft or "（本轮未生成有效答案）",
            "tool_trace": [
                {"name": r.name, "ok": r.ok, "error": r.error, "latency_ms": r.latency_ms}
                for r in tool_results
            ],
            "sources": sources,
            "hits": [self._hit_dict(hit) for hit in hits],
            "retrieval": retrieval,
            "pending_actions": pending_actions,
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
        report = citation_report(draft, self._state_hits(state), question)
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

    def node_approval(self, state: dict[str, Any]) -> dict[str, Any]:
        """Pause a mutating tool call, then execute or discard it on resume.

        ``interrupt()`` re-runs this node from the top when the graph resumes,
        so nothing before the call may have a side effect.
        """
        from langgraph.types import interrupt

        actions = state.get("pending_actions") or []
        if not actions:
            return {"approval": {"approved": False, "reason": "no_pending_action"}}
        action = actions[0]
        decision = interrupt(
            {"action": action, "question": "该操作会修改数据，是否确认执行？"}
        )
        approved = (
            bool(decision.get("approved"))
            if isinstance(decision, dict)
            else bool(decision)
        )
        trace = list(state.get("tool_trace") or [])
        action_id = str(action.get("action_id") or "")
        if not approved:
            self.registry.cancel(action_id)
            return {
                "draft": f"已取消 `{action.get('tool', '')}`，未写入任何数据。",
                "pending_actions": [],
                "tool_trace": trace,
                "approval": {
                    "approved": False,
                    "action_id": action_id,
                    "tool": action.get("tool", ""),
                },
            }
        result = self.registry.confirm(action_id)
        trace.append(
            {
                "name": result.name,
                "ok": result.ok,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
        )
        return {
            "draft": self._format_approval_result(action, result),
            "pending_actions": [],
            "tool_trace": trace,
            "approval": {
                "approved": True,
                "action_id": action_id,
                "tool": action.get("tool", ""),
                "ok": result.ok,
            },
        }

    def node_finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        turn_key = state.get("turn_id")
        if turn_key and state.get("finalized_turn") == turn_key:
            # Resume/replay: the turn was already written to memory. Do not
            # append it twice (and do not re-append the source list).
            return {
                "final_answer": state.get("final_answer", ""),
                "citations": state.get("citations") or {},
                "sources": state.get("sources") or [],
            }
        answer = state.get("draft", "")
        if not (state.get("guard") or {}).get("allowed", True):
            answer = (
                "抱歉，这个问题触发了安全策略（"
                + str((state.get("guard") or {}).get("reason", ""))
                + "），我不会执行。你可以换一种表述继续咨询内部事务。"
            )
        report = citation_report(
            answer, self._state_hits(state), state.get("question_safe", "")
        )
        sources = state.get("sources") or []
        answer = self._append_source_list(answer, sources)
        self.memory.add(state["question"], answer, sources)
        self.memory.compact(self.router)
        result = {
            "final_answer": answer,
            "citations": report,
            "sources": sources,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if turn_key:
            result["finalized_turn"] = turn_key
        return result

    # -- routing -------------------------------------------------------
    def route_after_guard(self, state: dict[str, Any]) -> str:
        return "continue" if (state.get("guard") or {}).get("allowed", True) else "blocked"

    def route_after_execute(self, state: dict[str, Any]) -> str:
        return "approval" if state.get("pending_actions") else "review"

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

    # -- real LangGraph runtime ---------------------------------------
    @staticmethod
    def _resolve_runtime(requested: str) -> str:
        requested = (requested or "auto").lower()
        if requested not in {"auto", "custom", "langgraph"}:
            raise ValueError(
                f"DE_GRAPH_RUNTIME 只能是 auto/custom/langgraph，收到 {requested!r}"
            )
        if requested == "custom":
            return "custom"
        from .langgraph_adapter import langgraph_available

        available = langgraph_available()
        if requested == "langgraph" and not available:
            raise RuntimeError(
                "DE_GRAPH_RUNTIME=langgraph 但未安装 langgraph / "
                "langgraph-checkpoint-sqlite；先运行 scripts/bootstrap-langgraph.ps1"
            )
        return "langgraph" if available else "custom"

    def _build_langgraph(self) -> Any:
        from .langgraph_adapter import build_langgraph_agent

        return build_langgraph_agent(self)

    def _ask_langgraph(self, question: str, thread_id: str, turn_id: str) -> dict[str, Any]:
        from .langgraph_adapter import invoke_langgraph

        state = invoke_langgraph(self, self.langgraph_app, question, thread_id, turn_id)
        for action in state.get("pending_actions") or []:
            action_id = str(action.get("action_id") or "")
            if action_id:
                self._remember_pending_thread(action_id, thread_id)
        return state

    def _resume_langgraph(self, thread_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        from .langgraph_adapter import resume_langgraph

        # A restarted process no longer has the in-memory PendingAction; rebuild
        # it from the checkpoint before the approval node calls registry.confirm.
        self._restore_pending_action(thread_id)
        return resume_langgraph(self, self.langgraph_app, thread_id, decision)

    def _langgraph_values(self, thread_id: str) -> dict[str, Any]:
        from .langgraph_adapter import get_langgraph_values

        return get_langgraph_values(self, self.langgraph_app, thread_id)

    def _restore_pending_action(self, thread_id: str) -> None:
        state = self._langgraph_values(thread_id)
        for item in state.get("pending_actions") or []:
            action_id = str(item.get("action_id") or "")
            if not action_id or action_id in self.registry.pending:
                continue
            self.registry.pending[action_id] = PendingAction(
                action_id=action_id,
                tool=str(item.get("tool") or ""),
                arguments=dict(item.get("arguments") or {}),
                created_at=time.time(),
            )

    def _load_pending_threads(self) -> dict[str, str]:
        try:
            data = json.loads(self._pending_threads_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, AttributeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _remember_pending_thread(self, action_id: str, thread_id: str) -> None:
        self._pending_threads[action_id] = thread_id
        self._flush_pending_threads()

    def _forget_pending_thread(self, action_id: str) -> None:
        self._pending_threads.pop(action_id, None)
        merged = self._load_pending_threads()
        merged.pop(action_id, None)
        self._pending_threads = merged
        self._flush_pending_threads()

    def _flush_pending_threads(self) -> None:
        try:
            self._pending_threads_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._pending_threads_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._pending_threads, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self._pending_threads_path)
        except OSError:
            pass

    def _pending_thread_for(self, action_id: str) -> str | None:
        return self._pending_threads.get(action_id) or self._load_pending_threads().get(
            action_id
        )

    @staticmethod
    def _format_approval_result(action: dict[str, Any], result: ToolResult) -> str:
        if not result.ok:
            return (
                f"`{action.get('tool', '')}` 执行失败："
                f"{result.error or result.content}"
            )
        try:
            payload = json.loads(result.content)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            ticket = payload.get("ticket_id") or payload.get("id")
            if ticket:
                return f"已创建工单 {ticket}。"
        return f"`{action.get('tool', '')}` 已执行：{str(result.content)[:500]}"

    def resume(self, thread_id: str, *, approved: bool = True) -> dict[str, Any]:
        """Resume an interrupted LangGraph thread (also works after restart)."""
        if self.runtime != "langgraph":
            raise RuntimeError("resume() 仅用于 langgraph 运行时")
        state = self._resume_langgraph(thread_id, {"approved": approved})
        return self._result(state.get("question", ""), state, 0.0, thread_id)

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _hit_dict(hit: Hit) -> dict[str, Any]:
        """Checkpoint-safe form of a Hit (LangGraph state must be serializable)."""
        chunk = hit.chunk
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "heading": chunk.heading,
            "text": chunk.text,
            "score": hit.score,
        }

    @staticmethod
    def _state_hits(state: dict[str, Any]) -> list[Hit]:
        hits: list[Hit] = []
        for item in state.get("hits") or []:
            if isinstance(item, Hit):
                hits.append(item)
                continue
            if not isinstance(item, dict):
                continue
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=str(item.get("chunk_id") or ""),
                        doc_id=str(item.get("doc_id") or ""),
                        title=str(item.get("title") or ""),
                        heading=str(item.get("heading") or ""),
                        text=str(item.get("text") or ""),
                    ),
                    score=float(item.get("score") or 0.0),
                )
            )
        return hits

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
    def confirm(self, action_id: str) -> dict[str, Any]:
        """Execute a parked mutating action (the `interrupt()` resume half)."""
        if self.runtime == "langgraph":
            thread_id = self._pending_thread_for(action_id)
            if not thread_id:
                return {
                    "action_id": action_id,
                    "tool": "",
                    "ok": False,
                    "result": "",
                    "error": "unknown_action",
                }
            state = self._resume_langgraph(
                thread_id, {"approved": True, "action_id": action_id}
            )
            self._forget_pending_thread(action_id)
            approval = state.get("approval") or {}
            ok = bool(approval.get("ok", False))
            return {
                "action_id": action_id,
                "tool": approval.get("tool", ""),
                "ok": ok,
                "result": state.get("final_answer") or state.get("draft", ""),
                "error": None if ok else "tool_failed",
            }
        result = self.registry.confirm(action_id)
        return {
            "action_id": action_id,
            "tool": result.name,
            "ok": result.ok,
            "result": result.content,
            "error": result.error,
        }

    def cancel(self, action_id: str) -> dict[str, Any]:
        if self.runtime == "langgraph":
            thread_id = self._pending_thread_for(action_id)
            if not thread_id:
                return {"cancelled": False, "action_id": action_id}
            self._resume_langgraph(
                thread_id, {"approved": False, "action_id": action_id}
            )
            self._forget_pending_thread(action_id)
            return {"cancelled": True, "action_id": action_id}
        return self.registry.cancel(action_id)

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
        turn_id = uuid.uuid4().hex
        thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        if self.runtime == "langgraph":
            final_state = self._ask_langgraph(question, thread_id, turn_id)
        else:
            final_state = {}
            for event in self.graph.stream(
                {"question": question, "turn_id": turn_id}, thread_id=thread_id
            ):
                if event["type"] == "end":
                    final_state = event["state"]
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        result = self._result(question, final_state, elapsed, thread_id)
        self.context.record_turn(
            result.get("thread_id") or thread_id,
            question,
            result.get("answer", ""),
            result.get("tool_trace") or [],
        )
        result["trace_id"] = self.tracer.record(
            question=question,
            result=result,
            elapsed_ms=elapsed,
            runtime=self.runtime,
        )
        return result

    def _result(self, question: str, final_state: dict[str, Any],
                elapsed_ms: float, thread_id: str | None = None) -> dict[str, Any]:
        citations = final_state.get("citations") or {}
        answer = final_state.get("final_answer") or final_state.get("draft") or ""
        self_check = {
            "has_answer": bool(answer),
            "cited": int(citations.get("cited", 0) or 0),
            "citation_valid": (
                bool(citations.get("existing")) if citations.get("cited") else None
            ),
            "support_rate": citations.get("support_rate", 0.0),
            "answer_chars": len(answer),
        }
        return {
            "question": question,
            "answer": answer,
            "sources": final_state.get("sources", []),
            "retrieval": final_state.get("retrieval", {}),
            "pending_actions": final_state.get("pending_actions", []),
            "citations": citations,
            "self_check": self_check,
            "plan": final_state.get("plan", {}),
            "skills": final_state.get("skills", []),
            "review": final_state.get("review", {}),
            "tool_trace": final_state.get("tool_trace", []),
            "revisions": final_state.get("revisions", 0),
            "trace": final_state.get("trace", []),
            "usage": self.router.usage_summary(),
            "latency_ms": elapsed_ms,
            "thread_id": final_state.get("thread_id") or thread_id,
            "guard": final_state.get("guard", {}),
            "interrupted": bool(final_state.get("__next__")),
            "runtime": self.runtime,
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
            "runtime": self.runtime,
            "langgraph": bool(self.langgraph_app),
            "skills": self.skills.names() if self.skills else [],
            "context": self.context.stats(),
            "tracing": self.tracer.stats(),
            "graph": self.graph.describe(),
        }


def build_digital_employee(settings: Settings | None = None) -> DigitalEmployee:
    agent = DigitalEmployee(settings)
    if len(agent.store) == 0:
        agent.ingest()
    return agent
