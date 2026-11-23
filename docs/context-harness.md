# Context Engineering + Agent Harness（阶段 2）

## 1. 上下文工程

`de_agent/context.py` 做三件事：

1. **分 section 的 token 预算**：`system / skills / memory / retrieval / tool / history`，默认总预算 6000 tokens。
2. **工具输出裁剪**：超过 `DE_TOOL_OUTPUT_MAX_CHARS`（默认 4000）时，保留 head/schema；
   如果是 JSON，先缩小字段和列表，保证返回的仍是**合法 JSON**，不会把 `sources` 截成半个对象。
3. **per-thread scratchpad**：`var/index/scratchpad/<thread>.md`，每轮记录 Q/A/工具名，
   跨会话可读，不占模型窗口。

配置：

```bash
DE_ENABLE_CONTEXT_BUDGET=true
DE_CONTEXT_TOTAL_TOKENS=6000
DE_TOOL_OUTPUT_MAX_CHARS=4000
DE_SCRATCHPAD_MAX_CHARS=4000
```

代码里已经接入：

- planner 的技能元数据按 `skills` section 截断；
- executor 的 plan / 技能正文按 section 截断；
- 每次工具结果进下一轮前先 `prune_tool_output`；
- `ask()` 结束后把本轮写入 scratchpad，`stats()["context"]` 可查。

## 2. Harness：多步任务、预算与续跑

`de_agent/harness.py` 是 Agent 外层的任务运行器：

```python
from de_agent.agents import build_digital_employee
from de_agent.harness import AgentHarness, RunBudget, TaskStep

agent = build_digital_employee()
harness = AgentHarness(agent)

steps = [
    TaskStep("leave", "年假有几天？"),
    TaskStep("sick", "病假需要什么材料？"),
    TaskStep("travel", "一线城市出差住宿每晚能报多少？"),
]

first = harness.run(steps, budget=RunBudget(max_steps=1, max_cost_cny=1.0))
print(first.status)        # budget_exceeded
print(first.steps_done)    # 1

# 新进程/新 Harness 也可以继续，读取同一个 runs_dir
resumed = AgentHarness(agent).resume(
    first.run_id,
    budget=RunBudget(max_steps=10, max_cost_cny=1.0),
)
print(resumed.status)      # completed
print(resumed.steps_done)  # 3
```

状态值：

| status | 含义 |
|---|---|
| `completed` | 所有 step 完成 |
| `budget_exceeded` | 触发 `max_steps / max_cost_cny / max_tokens / max_wall_clock_s` |
| `awaiting_approval` | step 里出现 mutating 工具，等待人工确认 |
| `failed` | `ask()` 抛异常，错误写入 `error` |

落盘位置：

```
var/index/harness/<run_id>/
├── state.json    # next_index / results / cost / tokens / pending_action
└── progress.md   # 人看的进度
```

审批中断的恢复（需 `DE_REQUIRE_CONFIRMATION=true`）：

```python
first = harness.run([TaskStep("ticket", "电脑蓝屏了，帮我报修一下")])
assert first.status == "awaiting_approval"

done = harness.resume(first.run_id, approved=True)   # 走 agent.confirm()
assert done.status == "completed"
```

`resume(approved=False)` 会调用 `agent.cancel()`，不会写工单。

## 3. 与 LangGraph checkpoint 的分工

| 层 | 负责什么 | 存在哪 |
|---|---|---|
| LangGraph checkpointer | 单次图执行在 `approval` 节点暂停/恢复 | `var/index/langgraph_checkpoints.sqlite` |
| Harness `state.json` | 任务跑到第几步、花了多少钱、每步结果 | `var/index/harness/<run_id>/state.json` |

两个都要有：LangGraph 保证“这一轮能续”，Harness 保证“整个任务能续”。

## 4. 测试

- `tests/test_context.py`：预算截断、JSON 裁剪仍合法、scratchpad、agent 落盘。
- `tests/test_harness.py`：预算停止后续跑、0 成本预算、审批通过/取消。
- 全量：112 个测试，OK。
- 评测：15 条质量指标不变（intent/tool/Recall 100%，keyword 86.7%）。

## 5. 边界

- token 计数用 `estimate_tokens`，是估算，不是账单；
- Harness 是**单写者** JSON，不要对同一 `run_id` 并跑两个进程；
- 步骤是顺序执行，没有做并行 DAG；
- scratchpad 是 per-thread 的纯文本，不是数据库，不要放密钥；
- 生产化还需要：动作过期清理、多实例锁、真实计费口径、审批人身份与审计。
