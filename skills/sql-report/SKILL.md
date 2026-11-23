---
name: sql-report
description: 把统计、排名、金额类问题转成单条只读 SQL，并解释结果口径。
when_to_use: 各部门报销总额、费用排名、同比环比、员工数据统计、金额汇总。
version: 1
allowed-tools: sql_query, calculator
---

# 只读 SQL 报表

1. 只生成单条 `SELECT`；禁止 `INSERT/UPDATE/DELETE/DROP/ALTER`。
2. 明确三件事：时间范围、部门/人员范围、指标口径；缺失时先问，或给默认并标注。
3. 调用 `sql_query` 执行，不自己编造数字。
4. 结果用表格或排名呈现；注明“基于演示数据库”，不要外推。
5. 需要二次计算（占比、同比）时调用 `calculator`，并说明公式。
6. 查询失败时原样转述错误，给一条修正建议，不要假装成功。
