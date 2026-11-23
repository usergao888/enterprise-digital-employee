# Agent Skills + MCP（阶段 1）

## 1. Skills：流程知识，不是工具

目录约定：

```
skills/
├── expense-qa/SKILL.md
├── ticket-triage/SKILL.md
└── sql-report/SKILL.md
```

`SKILL.md` 的 frontmatter：

```yaml
---
name: expense-qa
description: 回答差旅、报销、费用审批类制度问题，必须引用知识库原文。
when_to_use: 住宿标准、报销额度、发票、审批流程、出差补贴。
version: 1
allowed-tools: knowledge_search, calculator
---
```

正文是步骤、工具调用、输出契约、失败处理。加载策略：

1. 启动时只读元数据，注入 planner prompt；
2. 用户问题命中某个 skill 时，才把正文注入 executor；
3. `describe_for_prompt()` 不含正文，`instructions_for()` 才含。

新增/修改技能后：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_skills -v
```

开关：`DE_ENABLE_SKILLS=false` 可关闭；`DE_SKILLS_DIR=./skills` 可换目录。

## 2. MCP：把只读工具暴露给外部 Agent

安装：

```powershell
.\.venv\Scripts\python.exe -m pip install mcp
```

启动（stdio）：

```powershell
.\.venv\Scripts\python.exe -m de_agent.mcp_server
```

暴露的工具：

| MCP tool | 内部工具 | 说明 |
|---|---|---|
| `knowledge_search` | `knowledge_search` | 检索知识库，返回带 chunk 的 JSON |
| `sql_query` | `sql_query` | 单条只读 SELECT |
| `employee_profile` | `get_employee_profile` | 按工号查演示员工数据 |

`create_ticket` **不暴露**：MCP 客户端不能绕过人工确认写入数据。

Claude Desktop / Cursor 配置示例（把路径换成本机绝对路径）：

```json
{
  "mcpServers": {
    "enterprise-agent": {
      "command": "F:\\dsh_space\\enterprise-agent\\.venv\\Scripts\\python.exe",
      "args": ["-m", "de_agent.mcp_server"],
      "cwd": "F:\\dsh_space\\enterprise-agent"
    }
  }
}
```

验证：客户端里应能看到 `knowledge_search`、`sql_query`、`employee_profile` 三个工具，且没有 `create_ticket`。

## 3. AGENTS.md

仓库根目录的 [AGENTS.md](../AGENTS.md) 是给编码 Agent 的指令：目录职责、命令、运行时规则、HITL 安全边界、技能编辑约定、测试要求。

它和 `SKILL.md` 的分工：

- `AGENTS.md`：这个仓库怎么开发/测试/不踩坑；
- `SKILL.md`：业务任务怎么做（差旅问答、工单分诊、SQL 报表）。

## 4. 验收

- `tests.test_skills` 4 个用例通过；
- `tests.test_mcp` 的纯逻辑用例通过，server 构建用例在未安装 `mcp` 时 skip；
- 全量测试 112 个；
- `python -m de_agent ask "电脑蓝屏了..."` 的 `skills` 字段包含 `ticket-triage`。
