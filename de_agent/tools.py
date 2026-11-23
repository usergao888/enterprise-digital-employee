"""Tool calling layer.

Three things this module takes seriously, because they are what interviewers
actually probe:

1. **Schema quality** — a tool description is prompt engineering. Parameter
   names, enum values and units decide whether the model calls the tool
   correctly; we declare them as real JSON Schema and validate before executing.
2. **Failure containment** — a tool exception must become a structured,
   model-readable error (`{"error": ...}`), never a 500 that kills the turn.
   The graph then gets one chance to recover, which is exactly the retry path.
3. **Write-side safety** — read-only tools run freely; mutating tools are
   flagged `mutating=True` and can be gated behind confirmation.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .permissions import PermissionDecision, PermissionPolicy
from .rag import HybridRetriever, hits_to_sources
from .textutil import decompose_question

MAX_ROWS = 50
MAX_SQL_LENGTH = 600
_BLOCKED_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|pragma|vacuum)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Minimal JSON-Schema validation: required/type/enum/min/max/length."""
    errors: list[str] = []
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in arguments or arguments[name] in (None, ""):
            errors.append(f"缺少必填参数 `{name}`")
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            continue
        expected = spec.get("type")
        if expected in _TYPE_MAP and value is not None:
            python_type = _TYPE_MAP[expected]
            if not isinstance(value, python_type) or (
                expected in {"integer", "number"} and isinstance(value, bool)
            ):
                errors.append(f"参数 `{name}` 类型应为 {expected}")
                continue
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"参数 `{name}` 必须是 {spec['enum']} 之一")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"参数 `{name}` 不得小于 {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"参数 `{name}` 不得大于 {spec['maximum']}")
        if isinstance(value, str):
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                errors.append(f"参数 `{name}` 超过 {spec['maxLength']} 字符")
    return errors


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    name: str
    ok: bool
    content: str
    latency_ms: float = 0.0
    error: str | None = None
    pending_action_id: str | None = None

    def to_message_content(self) -> str:
        return self.content

    @property
    def pending(self) -> bool:
        return self.pending_action_id is not None


@dataclass
class PendingAction:
    action_id: str
    tool: str
    arguments: dict[str, Any]
    created_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "summary": f"{self.tool}({json.dumps(self.arguments, ensure_ascii=False)[:120]})",
        }


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    mutating: bool = False

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    confirm_mutating: bool = False
    pending: dict[str, PendingAction] = field(default_factory=dict)
    policy: PermissionPolicy | None = None

    def register(self, tool: Tool) -> "ToolRegistry":
        self.tools[tool.name] = tool
        return self

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def names(self) -> list[str]:
        return sorted(self.tools)

    # -- human-in-the-loop ---------------------------------------------
    def confirm(self, action_id: str) -> ToolResult:
        """Execute a previously parked mutating call (LangGraph `interrupt` analog)."""
        action = self.pending.pop(action_id, None)
        if action is None:
            return ToolResult(
                name="confirm",
                ok=False,
                content=json.dumps(
                    {"error": f"待确认动作 {action_id} 不存在或已过期"}, ensure_ascii=False
                ),
                error="unknown_action",
            )
        return self._execute(self.tools[action.tool], action.arguments)

    def cancel(self, action_id: str) -> dict[str, Any]:
        removed = self.pending.pop(action_id, None)
        return {"cancelled": removed is not None, "action_id": action_id}

    def _park(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        action = PendingAction(
            action_id=f"act-{uuid.uuid4().hex[:8]}",
            tool=tool.name,
            arguments=arguments,
            created_at=time.time(),
        )
        self.pending[action.action_id] = action
        return ToolResult(
            name=tool.name,
            ok=True,
            content=json.dumps(
                {
                    "status": "pending_confirmation",
                    "action_id": action.action_id,
                    "message": (
                        f"`{tool.name}` 会修改数据，已暂存待人工确认；"
                        "在确认前不要声称操作已完成。"
                    ),
                    "arguments": arguments,
                },
                ensure_ascii=False,
            ),
            pending_action_id=action.action_id,
        )

    # -- dispatch ------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                name=name,
                ok=False,
                content=json.dumps(
                    {"error": f"未知工具 `{name}`", "available": self.names()},
                    ensure_ascii=False,
                ),
                error="unknown_tool",
            )
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return ToolResult(
                    name=name,
                    ok=False,
                    content=json.dumps(
                        {"error": f"参数不是合法 JSON: {exc}", "raw": arguments[:200]},
                        ensure_ascii=False,
                    ),
                    error="bad_json",
                )
            arguments = parsed

        errors = validate_arguments(tool.parameters, arguments)
        if errors:
            return ToolResult(
                name=name,
                ok=False,
                content=json.dumps(
                    {"error": "参数校验失败", "details": errors, "schema": tool.parameters},
                    ensure_ascii=False,
                ),
                error="bad_arguments",
            )
        decision: PermissionDecision | None = (
            self.policy.decide(name, tool.mutating) if self.policy else None
        )
        if decision is not None and not decision.allow:
            return ToolResult(
                name=name,
                ok=False,
                content=json.dumps(
                    {"error": "权限策略拒绝", "reason": decision.reason},
                    ensure_ascii=False,
                ),
                error="permission_denied",
            )
        if tool.mutating and (
            self.confirm_mutating
            or (decision is not None and decision.requires_confirmation)
        ):
            return self._park(tool, arguments)
        return self._execute(tool, arguments)

    def _execute(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = tool.handler(**arguments)
            if isinstance(result, (dict, list)):
                content = json.dumps(result, ensure_ascii=False, default=str)
            else:
                content = str(result)
            ok, error = True, None
        except Exception as exc:  # noqa: BLE001 - tools must never kill the turn
            content = json.dumps(
                {"error": f"工具执行失败: {type(exc).__name__}: {exc}"}, ensure_ascii=False
            )
            ok, error = False, str(exc)

        latency = round((time.perf_counter() - started) * 1000, 2)
        self.call_log.append(
            {"name": tool.name, "ok": ok, "error": error, "latency_ms": latency,
             "arguments": arguments}
        )
        return ToolResult(
            name=tool.name, ok=ok, content=content, latency_ms=latency, error=error
        )


# ---------------------------------------------------------------------------
# safe calculator
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """AST-whitelist arithmetic — no eval(), no attribute access, no calls."""
    node = ast.parse(expression, mode="eval").body

    def walk(current: ast.AST) -> float:
        if isinstance(current, ast.Constant):
            if isinstance(current.value, (int, float)) and not isinstance(current.value, bool):
                return float(current.value)
            raise ValueError("只支持数字常量")
        if isinstance(current, ast.BinOp) and type(current.op) in _BIN_OPS:
            left = walk(current.left)
            right = walk(current.right)
            if isinstance(current.op, ast.Pow) and abs(right) > 12:
                raise ValueError("指数过大")
            return _BIN_OPS[type(current.op)](left, right)
        if isinstance(current, ast.UnaryOp) and type(current.op) in _UNARY_OPS:
            return _UNARY_OPS[type(current.op)](walk(current.operand))
        raise ValueError(f"不支持的表达式节点: {type(current).__name__}")

    value = walk(node)
    if abs(value) > 1e15:
        raise ValueError("结果超出可表示范围")
    return value


# ---------------------------------------------------------------------------
# built-in tools
# ---------------------------------------------------------------------------
def _seed_employee_db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT, department TEXT, title TEXT,
            manager TEXT, location TEXT, hire_date TEXT
        );
        CREATE TABLE expenses (
            id INTEGER PRIMARY KEY, department TEXT, category TEXT,
            amount REAL, month TEXT
        );
        INSERT INTO employees VALUES
          ('E1001','张伟','技术中心','后端工程师','E2001','武汉','2021-03-01'),
          ('E1002','李娜','人力资源部','HRBP','E2002','武汉','2020-07-15'),
          ('E1003','王强','财务部','财务分析师','E2003','上海','2019-11-02'),
          ('E1004','刘洋','技术中心','算法工程师','E2001','武汉','2022-06-20'),
          ('E1005','陈晨','市场部','市场专员','E2004','北京','2023-09-11');
        INSERT INTO expenses VALUES
          (1,'技术中心','云服务',128000,'2026-07'),(2,'技术中心','硬件',56000,'2026-07'),
          (3,'财务部','差旅',32000,'2026-07'),(4,'市场部','广告',210000,'2026-07'),
          (5,'人力资源部','招聘',45000,'2026-07'),(6,'技术中心','云服务',141000,'2026-08'),
          (7,'市场部','广告',198000,'2026-08'),(8,'财务部','差旅',28000,'2026-08');
        """
    )
    return connection


def guard_sql(sql: str) -> str:
    """Read-only enforcement: single SELECT statement, capped row count."""
    statement = (sql or "").strip().rstrip(";")
    if not statement:
        raise ValueError("SQL 为空")
    if len(statement) > MAX_SQL_LENGTH:
        raise ValueError("SQL 过长")
    if ";" in statement:
        raise ValueError("不允许多条语句")
    if not re.match(r"(?is)^\s*(select|with)\b", statement):
        raise ValueError("只允许 SELECT / WITH 查询")
    if _BLOCKED_SQL.search(statement):
        raise ValueError("检测到写操作关键词，已拒绝")
    if "limit" not in statement.lower():
        statement = f"{statement} LIMIT {MAX_ROWS}"
    return statement


def build_registry(
    retriever: HybridRetriever,
    *,
    ticket_path: Path,
    db: sqlite3.Connection | None = None,
    settings: Settings | None = None,
) -> ToolRegistry:
    db = db or _seed_employee_db()
    ticket_path.parent.mkdir(parents=True, exist_ok=True)
    settings = settings or Settings()
    registry = ToolRegistry(
        confirm_mutating=settings.require_confirmation,
        policy=PermissionPolicy(settings.permission_mode),
    )

    # -- knowledge search ---------------------------------------------
    def knowledge_search(
        query: str, top_k: int = 5, sub_queries: list[str] | None = None
    ) -> dict[str, Any]:
        k = min(int(top_k), 10)
        supplied = [str(s).strip() for s in (sub_queries or []) if str(s).strip()]
        origin = "llm" if supplied else "none"
        if not settings.enable_decomposition:
            # Config switch wins over the model: lets the decomposition A/B run
            # on identical inputs.
            supplied, origin = [], "none"
        elif not supplied:
            # The model may forget to decompose; the tool still does it. This is
            # why the feature works with models whose function-calling output is
            # minimal, not just the ones that follow the planner prompt exactly.
            supplied = decompose_question(query, max_parts=settings.max_sub_queries)
            origin = "rule_fallback" if supplied else "none"

        if supplied:
            # Sub-queries are substrings of the original by construction, so only
            # drop exact duplicates — filtering on "is a substring" deletes them all.
            queries = [query] + [s for s in supplied if s != query]
            result = retriever.search_multi(queries, k=k)
        else:
            result = retriever.search(query, k=k)

        return {
            "query": query,
            "queries": result.queries,
            "decomposed": len(result.queries) > 1,
            "decomposed_by": origin,
            "found": len(result.hits),
            "results": [
                {
                    "index": index,
                    "chunk_id": hit.chunk.chunk_id,
                    "title": hit.chunk.title,
                    "heading": hit.chunk.heading,
                    "score": round(hit.score, 4),
                    "content": hit.chunk.text,
                }
                for index, hit in enumerate(result.hits, start=1)
            ],
            "sources": hits_to_sources(result.hits),
            "timings_ms": result.timings_ms,
        }

    registry.register(
        Tool(
            name="knowledge_search",
            description=(
                "检索企业知识库（员工手册/IT 规范/财务报销/产品资料）。"
                "回答任何涉及公司制度、流程、政策的问题前必须先调用。"
                "复合问题（如“年假有几天、怎么申请”）请拆成 sub_queries 一起检索，"
                "只问一件事时留空。返回按相关度排序的原文片段，片段带 index，用于在答案中标注 [index] 出处。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户的原始问题，保留关键实体与限定词"},
                    "sub_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "复合问题拆出的独立子查询，最多 3 条；单问题传空数组。与 query 一起做多路召回融合",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10,
                              "default": 5, "description": "返回片段数"},
                },
                "required": ["query"],
            },
            handler=knowledge_search,
        )
    )

    # -- sql ----------------------------------------------------------
    def sql_query(sql: str) -> dict[str, Any]:
        statement = guard_sql(sql)
        cursor = db.execute(statement)
        rows = [dict(row) for row in cursor.fetchmany(MAX_ROWS)]
        return {"sql": statement, "row_count": len(rows), "rows": rows}

    registry.register(
        Tool(
            name="sql_query",
            description=(
                "对只读业务库执行 SELECT 查询。可用表：employees(employee_id,name,department,"
                "title,manager,location,hire_date)、expenses(id,department,category,amount,month)。"
                "仅支持单条 SELECT，自动追加 LIMIT。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "maxLength": MAX_SQL_LENGTH,
                            "description": "单条 SELECT 语句，字段名必须来自上面列出的表"},
                },
                "required": ["sql"],
            },
            handler=sql_query,
        )
    )

    # -- calculator ---------------------------------------------------
    def calculator(expression: str) -> dict[str, Any]:
        value = safe_eval(expression)
        return {"expression": expression, "result": round(value, 6)}

    registry.register(
        Tool(
            name="calculator",
            description="执行四则运算/乘方/取模，用于金额、比例、报销额度等精确计算。不要心算。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "maxLength": 200,
                                   "description": "仅含数字与 + - * / // % ** 与括号的表达式"},
                },
                "required": ["expression"],
            },
            handler=calculator,
        )
    )

    # -- employee profile --------------------------------------------
    def get_employee_profile(employee_id: str) -> dict[str, Any]:
        cursor = db.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return {"found": False, "employee_id": employee_id}
        return {"found": True, **dict(row)}

    registry.register(
        Tool(
            name="get_employee_profile",
            description="按工号查询员工档案（部门、岗位、直属上级、办公地点、入职日期）。",
            parameters={
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "description": "工号，例如 E1001"},
                },
                "required": ["employee_id"],
            },
            handler=get_employee_profile,
        )
    )

    # -- ticket -------------------------------------------------------
    def create_ticket(title: str, category: str, severity: str, detail: str = "") -> dict[str, Any]:
        tickets: list[dict[str, Any]] = []
        if ticket_path.exists():
            try:
                tickets = json.loads(ticket_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                tickets = []
        sla_hours = {"low": 48, "medium": 24, "high": 4}.get(severity, 24)
        ticket = {
            "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
            "title": title,
            "category": category,
            "severity": severity,
            "detail": detail,
            "status": "open",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "sla_hours": sla_hours,
            "due_at": (
                datetime.now(timezone.utc).astimezone() + timedelta(hours=sla_hours)
            ).isoformat(timespec="seconds"),
        }
        tickets.append(ticket)
        ticket_path.write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")
        return ticket

    registry.register(
        Tool(
            name="create_ticket",
            description=(
                "创建 IT/行政工单。仅在用户明确要求报修、提交故障或申请时调用；"
                "调用前需确认标题与严重程度，严重程度决定 SLA。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "maxLength": 80, "description": "工单标题"},
                    "category": {"type": "string", "enum": ["IT", "行政", "HR", "财务"],
                                 "description": "工单分类"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"],
                                 "description": "严重程度：high=4h、medium=24h、low=48h 响应"},
                    "detail": {"type": "string", "maxLength": 1000, "description": "问题详细描述"},
                },
                "required": ["title", "category", "severity"],
            },
            handler=create_ticket,
            mutating=True,
        )
    )

    # -- time ---------------------------------------------------------
    def current_time(utc_offset_hours: int = 8) -> dict[str, Any]:
        tz = timezone(timedelta(hours=int(utc_offset_hours)))
        now = datetime.now(tz)
        return {"iso": now.isoformat(timespec="seconds"),
                "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]}

    registry.register(
        Tool(
            name="current_time",
            description="获取当前时间与星期，用于判断报销/审批是否在时限内。",
            parameters={
                "type": "object",
                "properties": {
                    "utc_offset_hours": {"type": "integer", "minimum": -12, "maximum": 14,
                                         "default": 8, "description": "时区偏移，北京时间填 8"},
                },
                "required": [],
            },
            handler=current_time,
        )
    )

    return registry
