---
name: expense-qa
description: 回答差旅、报销、费用审批类制度问题，必须引用知识库原文。
when_to_use: 用户问住宿标准、报销额度、发票、审批流程、出差补贴、费用标准等。
version: 1
allowed-tools: knowledge_search, calculator
---

# 差旅报销问答

1. 先调用 `knowledge_search`，query 用用户原问题加关键实体（城市、金额、天数、费用类型）。
2. 只引用检索结果原文；金额、天数、比例必须带角标 `[n]`，角标对应 sources 的 index。
3. 涉及计算时调用 `calculator`，不要心算。
4. 输出结构：一句话结论 → 2–3 条依据（带角标）→ 一句下一步建议。
5. 知识库没有覆盖时，明确说“知识库中未提及”，不要编造数字或流程。
6. 差旅与报销口径冲突时，以财务制度文档为准，并说明依据。
