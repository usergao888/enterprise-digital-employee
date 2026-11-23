"""Task-level evaluation: 30 tasks, checks, and badcase attribution.

This complements `evaluation.py` (retrieval ablation + 15 golden questions) with
a task-oriented runner: each case declares the tool it expects, the documents it
needs, whether citations are mandatory, and the keywords the answer must contain.
Failures are bucketed so a bad answer is actionable:

    guard / tool_missing / tool_error / retrieval / citation / generation
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluation import ABLATION_EXTRA, GOLDEN, GoldenCase


@dataclass
class TaskCase:
    id: str
    question: str
    kind: str
    expected_docs: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_tool: str = ""
    must_cite: bool = False
    expect_blocked: bool = False
    strict_keywords: bool = True


def _from_golden(index: int, case: GoldenCase) -> TaskCase:
    return TaskCase(
        id=f"g{index:02d}",
        question=case.question,
        kind=case.intent,
        expected_docs=list(case.expected_docs),
        expected_keywords=list(case.expected_keywords),
        expected_tool=case.expected_tool,
        must_cite=case.must_cite,
        expect_blocked=case.expect_blocked,
    )


def _from_ablation(index: int, case: GoldenCase) -> TaskCase:
    return TaskCase(
        id=f"a{index:02d}",
        question=case.question,
        kind="kb",
        expected_docs=list(case.expected_docs),
        expected_keywords=list(case.expected_keywords),
        expected_tool="knowledge_search",
        must_cite=True,
    )


_EXTRA: list[TaskCase] = [
    TaskCase("x01", "迟到会扣钱吗？", "kb", ["员工手册/考勤与请假管理制度.md"], ["迟到"], "knowledge_search", True),
    TaskCase("x02", "加班可以调休吗？", "kb", ["员工手册/薪酬结构与绩效管理.md"], ["调休"], "knowledge_search", True),
    TaskCase("x03", "试用期工资怎么算？", "kb", ["员工手册/薪酬结构与绩效管理.md"], ["80%"], "knowledge_search", True),
    TaskCase("x04", "离职要提前多久申请？", "kb", ["员工手册/新员工入职与转正流程.md"], ["30 天"], "knowledge_search", True),
    TaskCase("x05", "报销需要哪些材料？", "kb", ["财务制度/差旅与报销管理制度.md"], ["发票"], "knowledge_search", True),
    TaskCase("x06", "工单提交后怎么查进度？", "kb", ["IT规范/设备报修与工单流程.md"], ["工单"], "knowledge_search", True),
    TaskCase("x07", "忘记打卡可以补卡吗？", "kb", ["员工手册/考勤与请假管理制度.md"], ["补卡"], "knowledge_search", True),
]

TASKS: list[TaskCase] = (
    [_from_golden(i, case) for i, case in enumerate(GOLDEN, start=1)]
    + [_from_ablation(i, case) for i, case in enumerate(ABLATION_EXTRA, start=1)]
    + _EXTRA
)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))], 2)


def evaluate_tasks(
    agent: Any,
    *,
    limit: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    cases = TASKS[:limit] if limit and limit > 0 else TASKS

    for index, case in enumerate(cases, start=1):
        if progress:
            print(
                f"[task] {index}/{len(cases)} {case.question}",
                file=sys.stderr,
                flush=True,
            )
        t0 = time.perf_counter()
        result = agent.ask(case.question, fresh_memory=True)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        answer = result.get("answer", "") or ""
        trace = result.get("tool_trace") or []
        sources = result.get("sources") or []
        citations = result.get("citations") or {}
        docs = {str(item.get("doc_id", "")) for item in sources}

        called = [item for item in trace if item.get("name") == case.expected_tool]
        tool_ok = not case.expected_tool or bool(called)
        tool_error = bool(called) and not all(item.get("ok") for item in called)
        doc_ok = not case.expected_docs or bool(docs & set(case.expected_docs))
        citation_ok = not case.must_cite or (
            int(citations.get("cited", 0) or 0) > 0
            and bool(citations.get("existing"))
        )
        keyword_ok = not (case.strict_keywords and case.expected_keywords) or all(
            keyword in answer for keyword in case.expected_keywords
        )
        guard_ok = (
            not case.expect_blocked
            or not (result.get("guard") or {}).get("allowed", True)
        )

        checks = {
            "guard": guard_ok,
            "tool": tool_ok,
            "tool_error": not tool_error,
            "retrieval": doc_ok,
            "citation": citation_ok,
            "generation": keyword_ok,
        }
        if not guard_ok:
            badcase = "guard"
        elif not tool_ok:
            badcase = "tool_missing"
        elif tool_error:
            badcase = "tool_error"
        elif not doc_ok:
            badcase = "retrieval"
        elif not citation_ok:
            badcase = "citation"
        elif not keyword_ok:
            badcase = "generation"
        else:
            badcase = None

        rows.append(
            {
                "id": case.id,
                "kind": case.kind,
                "question": case.question,
                "passed": all(checks.values()),
                "checks": checks,
                "badcase": badcase,
                "answer": answer[:300],
                "tool_trace": [item.get("name") for item in trace],
                "docs": sorted(docs),
                "latency_ms": latency,
                "cost_cny": float((result.get("usage") or {}).get("total_cost_cny", 0.0)),
            }
        )

    passed = sum(1 for row in rows if row["passed"])
    by_kind: dict[str, dict[str, int]] = {}
    badcases: dict[str, int] = {}
    for row in rows:
        bucket = by_kind.setdefault(row["kind"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(bool(row["passed"]))
        if row["badcase"]:
            badcases[row["badcase"]] = badcases.get(row["badcase"], 0) + 1

    latencies = [row["latency_ms"] for row in rows]
    summary = {
        "cases": len(rows),
        "passed": passed,
        "pass_rate": round(passed / max(1, len(rows)), 4),
        "by_kind": {
            kind: {
                **value,
                "pass_rate": round(value["passed"] / max(1, value["total"]), 4),
            }
            for kind, value in sorted(by_kind.items())
        },
        "badcases": dict(sorted(badcases.items(), key=lambda kv: (-kv[1], kv[0]))),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": _p95(latencies),
        "total_cost_cny": round(sum(row["cost_cny"] for row in rows), 6),
        "wall_clock_s": round(time.perf_counter() - started, 2),
    }
    return {"summary": summary, "rows": rows, "markdown": render_task_report(summary, rows)}


def render_task_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 任务级评测报告",
        "",
        f"- 任务数：{summary['cases']}",
        f"- 通过：{summary['passed']}（{summary['pass_rate']:.1%}）",
        f"- P50 / P95：{summary['latency_p50_ms']} / {summary['latency_p95_ms']} ms",
        f"- 总成本：CNY {summary['total_cost_cny']:.6f}",
        "",
        "## 按类型",
        "",
        "| 类型 | 通过 / 总数 | 通过率 |",
        "| --- | --- | --- |",
    ]
    for kind, value in summary["by_kind"].items():
        lines.append(f"| {kind} | {value['passed']} / {value['total']} | {value['pass_rate']:.1%} |")
    lines += [
        "",
        "## Badcase 归因",
        "",
    ]
    if summary["badcases"]:
        for bucket, count in summary["badcases"].items():
            lines.append(f"- {bucket}: {count}")
    else:
        lines.append("- 无")
    lines += ["", "## 明细", "", "| ID | 类型 | 结果 | 归因 | 问题 |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| {row['id']} | {row['kind']} | {mark} | {row['badcase'] or '-'} | {row['question']} |"
        )
    return "\n".join(lines) + "\n"


def run_task_eval(
    agent: Any,
    out_dir: Path,
    *,
    limit: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    report = evaluate_tasks(agent, limit=limit, progress=progress)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_eval_report.md").write_text(report["markdown"], encoding="utf-8")
    return report
