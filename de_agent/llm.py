"""LLM access layer: OpenAI-compatible client, deterministic offline fake,
and the per-stage model router.

The router is the piece that makes the resume bullet "根据需求复杂度自动选择模型"
real: every graph stage declares a workload profile, we pick a model for it,
and every call is metered so the cost saving can be reported as a number.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol

from .config import Settings
from .textutil import keyword_overlap, split_sentences


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly like the OpenAI API

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            data = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Iterable[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, tool_call_id: str, name: str, content: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class Completion:
    message: Message
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    stage: str = "chat"

    @property
    def text(self) -> str:
        return self.message.content


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------
class LLMClient(Protocol):
    def chat(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Completion: ...

    def stream(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate that is good enough for metering.

    CJK characters are roughly one token each; latin text roughly 4 chars per
    token. Deliberately conservative (rounds up).
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = max(0, len(text) - cjk)
    return cjk + (other + 3) // 4


# ---------------------------------------------------------------------------
# OpenAI-compatible client (DeepSeek / DashScope / vLLM / Ollama / ...)
# ---------------------------------------------------------------------------
class OpenAICompatClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.llm_base_url.rstrip("/")

    # -- plumbing ------------------------------------------------------
    def _request(self, path: str, payload: dict[str, Any], stream: bool) -> urllib.response.addinfourl:  # type: ignore[name-defined]
        url = f"{self.base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Accept": "text/event-stream" if stream else "application/json",
            },
        )
        try:
            return urllib.request.urlopen(request, timeout=self.settings.llm_timeout_s)
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise LLMError(f"无法连接 {url}: {exc.reason}") from exc

    def _payload(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "temperature": (
                self.settings.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        return payload

    # -- public --------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Completion:
        model = model or self.settings.llm_model
        payload = self._payload(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            stream=False,
        )
        started = time.perf_counter()
        response = self._request("/chat/completions", payload, stream=False)
        raw = response.read().decode("utf-8", "replace")
        latency = (time.perf_counter() - started) * 1000
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - network path
            raise LLMError(f"响应不是合法 JSON: {raw[:300]}") from exc

        choices = data.get("choices") or []
        if not choices:  # pragma: no cover - network path
            raise LLMError(f"响应缺少 choices: {raw[:300]}")
        message = choices[0].get("message") or {}
        tool_calls = [
            ToolCall(
                id=call.get("id") or f"call_{i}",
                name=(call.get("function") or {}).get("name", ""),
                arguments=(call.get("function") or {}).get("arguments", "{}"),
            )
            for i, call in enumerate(message.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        return Completion(
            message=Message(
                role="assistant",
                content=message.get("content") or "",
                tool_calls=tool_calls,
            ),
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens")
            or sum(estimate_tokens(m.content) for m in messages),
            completion_tokens=usage.get("completion_tokens")
            or estimate_tokens(message.get("content") or ""),
            latency_ms=latency,
            stage=stage,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        model = model or self.settings.llm_model
        payload = self._payload(
            messages,
            model=model,
            tools=None,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            stream=True,
        )
        response = self._request("/chat/completions", payload, stream=True)
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece


# ---------------------------------------------------------------------------
# deterministic offline backend (tests + demo without any API key)
# ---------------------------------------------------------------------------
_KB_HINTS = re.compile(
    r"制度|流程|规定|政策|标准|规范|手册|报销|请假|年假|病假|婚假|产假|陪产|考勤|打卡"
    r"|入职|离职|转正|试用期|社保|公积金|薪酬|绩效|调薪|晋升|权限|账号|密码|报修"
    r"|住宿|交通|出差|差旅|补贴|发票|审批|付款|培训|合同|SLA"
)
_SQL_HINTS = re.compile(r"排名|统计|汇总|总额|合计|各部门|哪个部门|环比|同比|趋势|报表|多少笔|笔数")
_SQL_CONTEXT = re.compile(r"部门|金额|总额|数量|笔数|数据|表|上个月|上月|本月|季度|20\d\d年")
_TICKET_ACTION = re.compile(r"(帮我|请|麻烦|我要|申请).{0,8}(报修|提交工单|创建工单|开工单|报个修)|报修一下")
_CALC_ACTION = re.compile(
    r"计算|算一下|算算|等于多少|打\s*\d+(?:\.\d+)?\s*折|折扣|乘以|除以|[×*x]\s*\d"
)
_PROFILE = re.compile(r"E\d{4}")
_TIME = re.compile(r"现在几点|当前时间|今天几号|今天星期|几点了")
_QUESTION_FORM = re.compile(
    r"怎么|如何|什么|哪些|哪种|是否|能否|可以么|能不能|多久|几天|多少天|为什么|去哪|找谁"
)


def detect_intent(text: str) -> str:
    """Best-effort intent classification for the offline planner.

    This is a stand-in for what a real model does, but the ordering matters and
    mirrors production logic: an explicit *action* (创建工单/计算) beats a
    topic keyword, and a topic keyword alone ("报销标准") must never be
    mistaken for a data query just because it contains a quantity word.
    """
    text = text or ""
    if _TICKET_ACTION.search(text):
        return "ticket"
    if _CALC_ACTION.search(text):
        return "calc"
    if _PROFILE.search(text):
        return "profile"
    if _TIME.search(text):
        return "time"
    if _SQL_HINTS.search(text) and _SQL_CONTEXT.search(text):
        return "sql"
    if _KB_HINTS.search(text):
        return "kb"
    # A question-shaped utterance with no actionable intent is a knowledge
    # question, not small talk — "电脑开不了机怎么办" must not fall to chat.
    if _QUESTION_FORM.search(text):
        return "kb"
    return "chat"


INTENT_TOOL = {
    "kb": "knowledge_search",
    "sql": "sql_query",
    "calc": "calculator",
    "ticket": "create_ticket",
    "profile": "get_employee_profile",
    "time": "current_time",
    "chat": "",
}


class FakeLLM:
    """Rule-based stand-in for a real model.

    It speaks the same protocol (including tool_calls) so the whole graph,
    tool loop, reviewer and evaluation harness can be exercised offline in CI.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.calls: list[tuple[str, str]] = []
        self._current_model: str | None = None
        self._prompt_tokens = 0

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _original_question(messages: list[Message]) -> str:
        """The first user turn is the task; later user turns are corrections."""
        for message in messages:
            if message.role == "user":
                return message.content
        return ""

    @staticmethod
    def _system_text(messages: list[Message]) -> str:
        return "\n".join(m.content for m in messages if m.role == "system")

    @classmethod
    def _planned_intent(cls, messages: list[Message]) -> str | None:
        match = re.search(r'"intent"\s*:\s*"([a-z_]+)"', cls._system_text(messages))
        return match.group(1) if match else None

    @classmethod
    def _planned_query(cls, messages: list[Message]) -> str | None:
        match = re.search(r'"rewritten_query"\s*:\s*"([^"]{1,300})"', cls._system_text(messages))
        return match.group(1) if match else None

    @staticmethod
    def _tool_results(messages: list[Message]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for message in messages:
            if message.role == "tool":
                out.append((message.name or "", message.content))
        return out

    def _completion(self, text: str, stage: str, tool_calls: list[ToolCall] | None = None) -> Completion:
        # Model and prompt tokens come from the router, so the cost report
        # reflects the routing decision instead of always billing the strong model.
        model = self._current_model or (
            self.settings.llm_model if self.settings else "fake-model"
        )
        return Completion(
            message=Message(role="assistant", content=text, tool_calls=list(tool_calls or [])),
            model=model,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=estimate_tokens(text),
            latency_ms=1.0,
            stage=stage,
        )

    # -- public --------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Completion:
        question = self._original_question(messages)
        self._current_model = model or (
            self.settings.llm_model if self.settings else "fake-model"
        )
        self._prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        self.calls.append((stage, question))
        if stage == "plan":
            return self._plan(question)
        if stage == "review":
            return self._review(messages, question)
        if stage == "summarize":
            return self._completion(
                "用户关注：" + (question[:60] or "暂无"), "summarize"
            )
        if stage == "execute":
            return self._execute(messages, question, tools or [])
        return self._chat(question)

    def stream(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        completion = self.chat(messages, stage=stage, model=model)
        text = completion.text
        for i in range(0, len(text), 12):
            yield text[i : i + 12]

    # -- stage implementations -----------------------------------------
    def _plan(self, question: str) -> Completion:
        intent = detect_intent(question)
        steps = {
            "kb": ["检索企业知识库", "整合答案并标注出处"],
            "sql": ["查询业务数据库", "汇总指标并解释口径"],
            "calc": ["调用计算工具", "给出计算过程与结论"],
            "ticket": ["调用工单工具创建工单", "回复工单号与处理时限"],
            "profile": ["按工号查询员工档案", "返回部门与上级信息"],
            "time": ["读取系统时间", "按工作日推算时限"],
            "chat": ["理解问题并直接回答"],
        }.get(intent, ["理解问题并直接回答"])
        plan = {
            "intent": intent,
            "steps": steps,
            "need_knowledge": intent == "kb",
            "rewritten_query": re.sub(r"\s+", " ", question).strip()[:300],
        }
        return self._completion(json.dumps(plan, ensure_ascii=False), "plan")

    def _execute(
        self,
        messages: list[Message],
        question: str,
        tools: list[dict[str, Any]],
    ) -> Completion:
        available = {
            (t.get("function") or {}).get("name", "") for t in tools if t.get("type") == "function"
        }
        results = self._tool_results(messages)
        # The executor follows the planner's decision; re-classifying here is
        # how you end up with a plan that says "answer from the knowledge base"
        # and an executor that runs an expense aggregate instead.
        intent = self._planned_intent(messages) or detect_intent(messages[0].content if messages else "")
        query = self._planned_query(messages) or question

        if not results:
            planned_tool = INTENT_TOOL.get(intent, "")
            if planned_tool and planned_tool in available:
                arguments = self._fake_arguments(planned_tool, query, intent)
                call = ToolCall(
                    id=f"call_{planned_tool}_1",
                    name=planned_tool,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
                return self._completion("", "execute", [call])
            return self._completion(
                f"关于「{question[:60]}」，知识库中没有可引用的内部依据，建议联系对应业务负责人确认。",
                "execute",
            )

        return self._compose(question, results, intent)

    def _fake_arguments(self, tool: str, question: str, intent: str) -> dict[str, Any]:
        numbers = re.findall(r"\d+(?:\.\d+)?", question)
        if tool == "sql_query":
            if re.search(r"员工|人数|headcount|工号", question):
                sql = (
                    "SELECT department, COUNT(*) AS headcount FROM employees "
                    "GROUP BY department ORDER BY headcount DESC"
                )
            elif re.search(r"报销|费用|支出|花费|金额", question):
                sql = (
                    "SELECT department, SUM(amount) AS total FROM expenses "
                    "GROUP BY department ORDER BY total DESC LIMIT 3"
                )
            else:
                sql = (
                    "SELECT department, category, SUM(amount) AS total FROM expenses "
                    "GROUP BY department, category ORDER BY total DESC LIMIT 5"
                )
            return {"sql": sql}
        if tool == "calculator":
            if "折" in question and len(numbers) >= 2:
                expression = f"{numbers[0]}*{numbers[1]}/100"
            elif re.search(r"[×*x]\s*\d", question) and len(numbers) >= 2:
                expression = f"{numbers[0]}*{numbers[1]}"
            elif len(numbers) >= 2:
                expression = f"{numbers[0]}+{numbers[1]}"
            elif numbers:
                expression = numbers[0]
            else:
                expression = "0"
            return {"expression": expression}
        if tool == "create_ticket":
            title = re.sub(r"^(帮我|请|麻烦|我要|我想)", "", question).strip()[:40] or "故障报修"
            severity = (
                "high" if re.search(r"开不了机|无法开机|打不开|多人|演示|宕机|断电", question) else "medium"
            )
            return {
                "title": title,
                "category": "IT",
                "severity": severity,
                "detail": question[:1000],
            }
        if tool == "get_employee_profile":
            match = re.search(r"E\d{4}", question)
            return {"employee_id": match.group(0) if match else "E1001"}
        if tool == "current_time":
            return {"utc_offset_hours": 8}
        return {"query": question, "top_k": 5}

    @staticmethod
    def _render_rows(payload: dict[str, Any]) -> str:
        rows = payload.get("rows") or []
        if not rows:
            return "查询没有返回数据。"
        columns = list(rows[0].keys())
        header = " | ".join(columns)
        divider = " | ".join("---" for _ in columns)
        body = "\n".join(
            " | ".join(str(row.get(col)) for col in columns) for row in rows[:10]
        )
        return (
            f"```\n{header}\n{divider}\n{body}\n```\n"
            f"共 {payload.get('row_count', len(rows))} 行。"
        )

    def _compose(
        self, question: str, results: list[tuple[str, str]], intent: str
    ) -> Completion:
        name, content = results[-1]
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            return self._completion(
                f"工具 `{name}` 执行失败（{payload.get('error')}）。建议改走人工：IT 服务台分机 8000。",
                "execute",
            )
        if name == "sql_query" and isinstance(payload, dict):
            return self._completion(
                "查询结果如下：\n"
                + self._render_rows(payload)
                + "\n口径：amount 为当月报销金额合计，未包含预付款与借款。",
                "execute",
            )
        if name == "calculator" and isinstance(payload, dict):
            return self._completion(
                f"计算结果：{payload.get('result')}（表达式 {payload.get('expression')}）。",
                "execute",
            )
        if name == "create_ticket" and isinstance(payload, dict):
            return self._completion(
                f"已为你创建工单 {payload.get('ticket_id')}，严重程度 {payload.get('severity')}，"
                f"按 SLA 约定将在 {payload.get('sla_hours')} 小时内响应，截止时间 {payload.get('due_at')}。",
                "execute",
            )
        if name == "get_employee_profile" and isinstance(payload, dict):
            if not payload.get("found"):
                return self._completion(
                    f"未查到工号 {payload.get('employee_id')} 的档案，请确认工号是否正确。",
                    "execute",
                )
            return self._completion(
                f"工号 {payload.get('employee_id')}：{payload.get('name')}，"
                f"{payload.get('department')} · {payload.get('title')}，"
                f"直属上级 {payload.get('manager')}，办公地 {payload.get('location')}，"
                f"入职日期 {payload.get('hire_date')}。",
                "execute",
            )
        if name == "current_time" and isinstance(payload, dict):
            return self._completion(
                f"当前时间 {payload.get('iso')}（{payload.get('weekday')}）。", "execute"
            )

        hits = (payload or {}).get("results") or []
        if not hits:
            return self._completion(
                "知识库中没有找到与问题直接相关的内容，建议补充关键词或联系对应负责人确认。",
                "execute",
            )
        lines = ["结论："]
        bullets = 0
        for position, hit in enumerate(hits):
            if bullets >= 3:
                break
            sentences = split_sentences(hit.get("content", ""))
            # An FAQ heading is a question, not evidence. Citing "实习生能出差吗？"
            # as a finding is the offline equivalent of a hallucination.
            declarative = [
                s for s in sentences if not s.strip().endswith(("？", "?"))
            ] or sentences
            if not declarative:
                continue
            ranked = sorted(
                declarative, key=lambda s: keyword_overlap(question, s), reverse=True
            )
            wanted = 2 if position == 0 else 1
            # Chunks further down the ranking must clear a higher bar. A 5th-place
            # chunk that shares one word ("申请") is how an IT permission rule
            # ends up quoted in an annual-leave answer. The top chunk is exempt:
            # it already won on title/heading relevance, and its best sentence is
            # the answer even when the wording differs from the question.
            for rank_in_chunk, sentence in enumerate(ranked[:wanted]):
                sentence = sentence.strip()
                if position == 0 and rank_in_chunk == 0:
                    required = 0.0
                else:
                    required = 0.14 if position <= 1 else 0.26
                if keyword_overlap(question, sentence) < required:
                    continue
                # Answer-only fragments ("可以，需导师与HRBP双签…") lose their
                # meaning once the FAQ question above them is stripped.
                if sentence.startswith(("可以，", "不可以", "是的", "不是", "能，", "不能，")):
                    continue
                title = hit.get("title", "制度文档")
                lines.append(f"- {sentence}（《{title}》）[{hit.get('index')}]")
                bullets += 1
                if bullets >= 3:
                    break
        if len(lines) == 1:
            return self._completion(
                "知识库中没有找到能直接支撑结论的条款，建议补充关键词或联系对应负责人确认。",
                "execute",
            )
        lines.append("以上条款摘自企业知识库原文，角标可回溯到原文段落。")
        return self._completion("\n".join(lines), "execute")

    def _review(self, messages: list[Message], question: str) -> Completion:
        # The draft lives inside the reviewer's system prompt, not in an
        # assistant turn — reading the wrong place silently reviewed "".
        system_text = self._system_text(messages)
        draft_match = re.search(r"草稿：\s*(.*?)\s*检查项", system_text, flags=re.DOTALL)
        draft = draft_match.group(1) if draft_match else ""
        if not draft:
            for message in reversed(messages):
                if message.role == "assistant" and message.content:
                    draft = message.content
                    break
        intent_match = re.search(r"问题类型[：:]\s*([a-z_]+)", system_text)
        intent = intent_match.group(1) if intent_match else detect_intent(question)
        issues: list[str] = []
        if intent == "kb" and "[" not in draft:
            issues.append("缺少知识库引用角标")
        if len(draft) < 20:
            issues.append("答案过短，未形成结论")
        verdict = "revise" if issues else "pass"
        payload = {
            "verdict": verdict,
            "issues": issues,
            "fix_hint": "补充引用并给出可执行结论" if issues else "",
        }
        return self._completion(json.dumps(payload, ensure_ascii=False), "review")

    def _chat(self, question: str) -> Completion:
        return self._completion(f"（离线模式）我理解你的问题是：{question[:120]}", "chat")


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
_STAGE_PROFILE = {
    "plan": "strong",
    "review": "strong",
    "execute": "small",
    "chat": "small",
    "summarize": "small",
}


@dataclass
class UsageRecord:
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_cny: float


class MeteredRouter:
    """Wraps a client, chooses a model per stage, and meters every call."""

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.catalog = settings.model_catalog()
        self.records: list[UsageRecord] = []

    # -- routing -------------------------------------------------------
    def choose_model(self, stage: str, question: str = "") -> str:
        if not self.settings.router_enabled:
            return self.settings.llm_model
        profile = _STAGE_PROFILE.get(stage, "small")
        if profile == "strong":
            return self.settings.llm_model
        # Escalate genuinely long/compound questions to the strong model.
        if len(question) >= self.settings.router_long_question_chars:
            return self.settings.llm_model
        return self.settings.small_model

    # -- metering ------------------------------------------------------
    def _meter(self, completion: Completion, question: str) -> None:
        spec = self.catalog.get(completion.model)
        cost = 0.0
        if spec is not None:
            cost = (
                completion.prompt_tokens / 1000 * spec.input_price_per_1k
                + completion.completion_tokens / 1000 * spec.output_price_per_1k
            )
        self.records.append(
            UsageRecord(
                stage=completion.stage,
                model=completion.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                latency_ms=completion.latency_ms,
                cost_cny=round(cost, 6),
            )
        )

    # -- public --------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        question = ""
        for message in reversed(messages):
            if message.role == "user":
                question = message.content
                break
        model = self.choose_model(stage, question)
        completion = self.client.chat(
            messages,
            stage=stage,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
        )
        self._meter(completion, question)
        return completion

    def stream(
        self,
        messages: list[Message],
        *,
        stage: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        question = ""
        for message in reversed(messages):
            if message.role == "user":
                question = message.content
                break
        model = self.choose_model(stage, question)
        yield from self.client.stream(
            messages,
            stage=stage,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

    # -- reporting -----------------------------------------------------
    def usage_summary(self) -> dict[str, Any]:
        by_stage: dict[str, dict[str, float]] = {}
        for record in self.records:
            bucket = by_stage.setdefault(
                record.stage,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_cny": 0.0},
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += record.prompt_tokens
            bucket["completion_tokens"] += record.completion_tokens
            bucket["cost_cny"] = round(bucket["cost_cny"] + record.cost_cny, 6)
        total_calls = len(self.records)
        total_cost = round(sum(r.cost_cny for r in self.records), 6)
        # What the same traffic would have cost on the strong model only.
        strong = self.catalog.get(self.settings.llm_model)
        baseline = 0.0
        if strong is not None:
            for record in self.records:
                baseline += (
                    record.prompt_tokens / 1000 * strong.input_price_per_1k
                    + record.completion_tokens / 1000 * strong.output_price_per_1k
                )
        saving = 0.0
        if baseline > 0:
            saving = (baseline - total_cost) / baseline
        return {
            "calls": total_calls,
            "by_stage": by_stage,
            "total_cost_cny": total_cost,
            "single_model_baseline_cny": round(baseline, 6),
            "saving_rate": round(saving, 4),
        }

    def reset_usage(self) -> None:
        self.records.clear()
