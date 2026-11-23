# 真 LangGraph 迁移 · 环境与验证（Phase 0）

> 日期：2026-11-23
> 状态：Phase 0 已实现并验收。
> 实测：`langgraph 1.2.24` / `langgraph-checkpoint-sqlite 3.1.3` / `numpy 2.4.6` / Python 3.11.9；112 个测试全绿；15 条质量指标与 custom 一致（延迟见报告）。
> 本文保留环境准备与排障；实现见 `de_agent/langgraph_adapter.py` 与 `de_agent/agents.py`。

---

## 0. 为什么先做这一步

旧版 `de_agent/langgraph_adapter.py` 只把节点挂到真 LangGraph 上，用的是 `MemorySaver`，没有：

- 持久 checkpointer（进程重启后恢复）
- `interrupt()` / `Command(resume=...)` 的人工确认
- `get_state()` / `update_state()` 的状态检查
- 与现有 SSE 事件对齐的流式映射

这些 API 在不同 LangGraph 版本里导入路径和行为有差异。先跑 smoke，把**版本 + 行为**固定下来，再写迁移补丁，避免“写完才发现 `interrupt` 在 0.2.28 里不存在”。

---

## 1. 一键脚本（Windows）

在仓库根目录：

```powershell
cd enterprise-agent
.\scripts\bootstrap-langgraph.cmd
```

`.cmd` 只是用**进程级** `-ExecutionPolicy Bypass` 调用 `.ps1`，不会修改系统或用户的执行策略。

如果直接跑 `.ps1` 报“无法加载文件……未对文件进行数字签名”，三种解法任选：

```powershell
# A. 用 .cmd（推荐，不改策略）
.\scripts\bootstrap-langgraph.cmd

# B. 只对当前 PowerShell 窗口放开
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\bootstrap-langgraph.ps1

# C. 显式启动一个绕过策略的 PowerShell 进程
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap-langgraph.ps1
```

> 不要改 `LocalMachine` 策略，也没必要 `Unblock-File`——报错是“未签名”，不是“来自网络”。

脚本会：

1. 用 `py -3.11`（或 `python`）建 `.venv`
2. 安装 `numpy`、`langgraph`、`langgraph-checkpoint-sqlite`
3. 跑 `scripts/langgraph_smoke.py`
4. 跑现有 unittest 作为基线

常用参数：

```powershell
# 只看命令，不建 venv、不装包
.\scripts\bootstrap-langgraph.cmd -DryRun

# 重建 venv
.\scripts\bootstrap-langgraph.cmd -Recreate

# 跳过测试，只跑 smoke
.\scripts\bootstrap-langgraph.cmd -SkipTests

# 公司/校园网必须走代理时
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
.\scripts\bootstrap-langgraph.cmd

# 当前窗口残留代理变量、但代理没开时：本次运行直连
.\scripts\bootstrap-langgraph.cmd -NoProxy

# 镜像不通时换源
.\scripts\bootstrap-langgraph.cmd -IndexUrl "https://pypi.org/simple/"
```

---

## 2. 手动安装（macOS / Linux / 脚本失败时）

```bash
cd enterprise-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements-langgraph.txt
python scripts/langgraph_smoke.py
python -m unittest discover -s tests -t .
```

Windows 手动版：

```powershell
cd enterprise-agent
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r scripts\requirements-langgraph.txt
python scripts\langgraph_smoke.py
python -m unittest discover -s tests -t .
```

---

## 3. 期望输出

```text
== LangGraph smoke test ==
installed versions:
  langgraph==0.6.x          # 以实际为准
  langgraph-checkpoint-sqlite==2.x
  numpy==2.x

[OK]   imports (StateGraph, MemorySaver, interrupt, Command)
[OK]   interrupt / get_state / Command(resume)
[OK]   SqliteSaver durability
RESULT: LANGGRAPH_SMOKE_OK

Ran 70 tests in ...s
OK
```

版本号不重要，**行为**重要。smoke 通过说明：

- `invoke` 在 `interrupt()` 处停住，结果里有 `__interrupt__`
- `get_state(config).next == ("approval",)`
- `Command(resume=True)` 能继续并写回状态
- `SqliteSaver` 关闭重开后，`get_state` 仍能读到已批准状态

---

## 4. Phase 0 验收清单

- [x] `RESULT: LANGGRAPH_SMOKE_OK`
- [x] `interrupt` → `get_state().next` → `Command(resume=...)` 通过
- [x] `SqliteSaver` 重开后状态还在
- [x] 现有测试全绿（112 个）
- [x] 记录版本：`langgraph`、`langgraph-checkpoint-sqlite`、Python
- [ ] 把 smoke 输出贴到迁移 issue/PR

---

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `No module named numpy` | 用 `.venv\Scripts\python.exe` 跑，不要用系统 Python |
| `No module named langgraph` | 依赖没装进 venv；重跑 `pip install -r scripts/requirements-langgraph.txt` |
| `cannot import name 'interrupt'` | LangGraph 太旧；`pip install -U "langgraph>=0.2.40"` |
| `No module named langgraph.checkpoint.sqlite` | 少装 `langgraph-checkpoint-sqlite` |
| `SqliteSaver` 构造报 `TypeError` | 版本差异；把报错和版本发我，补丁按实际签名写 |
| 安装超时 / SSL 错误 | 配 `HTTPS_PROXY`，或 `-IndexUrl` 换源；先在浏览器打开镜像确认可达 |
| `ProxyError ... WinError 10061` | 当前窗口残留 `HTTPS_PROXY`/`HTTP_PROXY` 指向未启动的代理；先 `Remove-Item Env:HTTPS_PROXY, Env:HTTP_PROXY, Env:ALL_PROXY`，或启动代理；脚本可加 `-NoProxy` |
| `Cache entry deserialization failed` | 上次中断留下的坏缓存；加 `--no-cache-dir` 重试（文档下方手动命令已带） |
| 测试卡住 | HTTP 端到端测试会起本地 server；看是否被安全软件拦端口 |
| `py -3.11` 不存在 | `-PythonVersion 3.12`，或直接装 3.11；脚本会回退到 `python` |
| 测试数量不是 112 | 记录实际数量；以你当前分支为准，不要改测试凑数 |

---

## 6. Phase 0 已落地

1. `config.py`：`DE_GRAPH_RUNTIME=auto|custom|langgraph`。
2. `langgraph_adapter.py`：`StateGraph(AgentState)` + `SqliteSaver` + `recursion_limit` config + `interrupt`/`Command`。
3. `agents.py`：`approval` 节点（`interrupt()` 前无副作用）；`ask/resume/confirm/cancel` 按运行时分流；`finalize` 按 `turn_id` 幂等；`hits` 改为可序列化 dict。
4. `tests/test_langgraph_runtime.py`：6 个端到端用例；现有测试固定 `graph_runtime=custom`。
5. `README.md` / `.env.example`：运行时开关、checkpoint、确认动作。

验收结果：

- `DE_GRAPH_RUNTIME=langgraph` 下 15 条评测与 custom 一致（intent/tool 100%、Recall@5 100%、keyword hit 86.7%、成本节省 56.6%）。
- 中断 → 审批 → 确认 → 恢复通过；重启后 `confirm(action_id)` 仍可用。
- 112 tests OK；无 dataclass checkpoint 警告。

下一步（Phase 2）：Context Engineering + Agent Harness，见 [Agent项目技术增补清单_2026-11.md](../../Agent项目技术增补清单_2026-11.md)。Phase 1 的 Skills / AGENTS.md 已落地，MCP server 已写好，装 `mcp` 后跑 `tests.test_mcp` 验证。

---

## 7. 这次不做什么

- 不改 `de_agent` 源码
- 不动 `graph.py` 自研引擎（保留 fallback）
- 不装 Qdrant/Redis/Langfuse（那是阶段 2/3）
- 不跑真实模型（需要 Key；先用 FakeLLM 验证运行时）
