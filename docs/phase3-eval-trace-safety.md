# 任务级 Eval + Trace + 权限/Sandbox（阶段 3）

## 1. 任务级 eval：30 条

`de_agent/task_eval.py` 在 15 条 golden 之外，加了 8 条检索消融题和 7 条制度题，
每条声明：期望工具、期望文档、必填引用、答案关键词。

```powershell
.\.venv\Scripts\python.exe -m de_agent taskeval
```

输出：`var/index/task_eval_report.md`，包含：

- 按类型（kb/sql/calc/ticket/profile/blocked）的通过率；
- badcase 归因：`guard / tool_missing / tool_error / retrieval / citation / generation`；
- 明细表：每条任务的检查项与失败桶。

**离线基线（FakeLLM）**：

| 指标 | 数值 |
|---|---|
| 通过 | 20 / 30（66.7%） |
| P50 / P95 | 14.7 / 27 ms |
| badcase | `tool_missing` 6，`generation` 4 |

这组数字比原来的“15 条 100%”难看，但它是真的：
- 6 个 `tool_missing` 是 FakeLLM planner 正则没覆盖新问法，不是检索坏了；
- 4 个 `generation` 是抽取式答案的措辞没命中关键词。

接真实模型后应重跑同一脚本；这正是任务级 eval 的意义——把“演示 100%”拆成可归因的失败。

## 2. Trace：每个 ask 一份 JSON

`de_agent/tracing.py` 写 OTel 形状的 spans：

```
var/index/traces/<date>/<trace_id>.json
```

一次 `ask` 的 span：

- `agent.ask`：总耗时、runtime、答案长度；
- `node.plan / node.execute / node.review ...`：节点耗时/更新；
- `llm.plan / llm.execute / llm.review`：调用次数、prompt/completion tokens、成本；
- `tool.knowledge_search / tool.create_ticket ...`：工具耗时、是否成功。

`ask()` 的返回值新增 `trace_id`；`DE_ENABLE_TRACING=false` 可关闭。
JSON 是 OTel 形状的本地源，接 Langfuse/OTLP 时写一个 exporter 即可，不需要改 Agent。

## 3. 权限：read_only / ask / auto

`de_agent/permissions.py` + `ToolRegistry`：

| 模式 | 读工具 | 写工具（create_ticket） |
|---|---|---|
| `read_only` | 允许 | 直接拒绝，返回 `permission_denied` |
| `ask` | 允许 | 暂存为 pending，等 `confirm` |
| `auto` | 允许 | 按 `DE_REQUIRE_CONFIRMATION` 决定是否暂存 |

```bash
DE_PERMISSION_MODE=read_only   # 演示/审计
DE_PERMISSION_MODE=ask         # 推荐生产默认
DE_PERMISSION_MODE=auto        # 保留原 demo 行为
```

## 4. Sandbox

`de_agent/sandbox.py`：

- `docker_command()`：`--network none`、`--read-only`、`--cap-drop ALL`、
  `--pids-limit`、`--memory`、`--user 1000:1000`、只读挂载；
- `run_python()`：`-I -S` 子进程 + 超时，用文件而不是管道收输出（兼容 Windows sandbox 对命名管道的限制）。

边界：`run_python()` 在没有 Docker 的 Windows 上**不是安全边界**；真正执行不可信代码必须用 Docker/microVM。

## 5. MCP 实跑

- 安装：`mcp 2.3.0`；
- SDK v2 把 `FastMCP` 改名为 `MCPServer`，`de_agent/mcp_server.py` 同时兼容 v1/v2；
- 用 **streamable HTTP** transport 实跑通过：
  - `list_tools` 返回 `knowledge_search / sql_query / employee_profile`；
  - `knowledge_search` 返回 1643 字符的 JSON；
  - `create_ticket` 不在工具列表里。
- 本机沙箱禁止子进程命名管道，stdio transport 的 MCP client 被 `WinError 5` 拦住；
  在普通终端里 stdio 可用，HTTP 已证明协议层/工具层是通的。

## 6. 测试

- 全量：**112 个测试，OK**（MCP 已安装，server 构建用例不再 skip）；
- 原 15 条 eval：质量指标不变（intent/tool/Recall 100%，keyword 86.7%）；
- 任务级 eval：离线 20/30，badcase 已归因；
- trace / permission / sandbox 各有独立测试。

## 7. 还没做

- OTLP/Langfuse exporter（当前是本地 JSONL/JSON 源）；
- 真实 Docker daemon 下的 codegen 沙箱执行；
- 任务级 eval 用真实模型重跑（这是下一步最重要的一次测量）。
