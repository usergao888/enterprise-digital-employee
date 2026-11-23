---
name: ticket-triage
description: 判断 IT/行政故障类型并创建工单；创建前必须人工确认。
when_to_use: 电脑蓝屏、网络断开、账号锁定、设备报修、工单进度、IT 故障。
version: 1
allowed-tools: knowledge_search, create_ticket, employee_profile
---

# 故障分诊与工单

1. 抽取三要素：设备/系统、现象、工号或部门；缺失时先追问一次，不要猜。
2. 先查知识库是否有自助修复步骤；有则先给步骤，再判断是否需要报修。
3. 需要报修时调用 `create_ticket`。这是 mutating 工具：
   - 未确认前不得声称工单已创建；
   - 只把 `action_id` 和待确认动作交给用户；
   - 用户确认后才可继续。
4. 输出结构：故障分类 → 自助步骤 → 待确认动作（action_id）。
5. 涉及员工信息时只查必要字段，不复述手机号、身份证等敏感数据。
