# 真实模型重跑

> 当前机器没有 `DE_LLM_API_KEY`，所以还没有真实跑分。脚本只从环境变量读 Key，**不会写盘、不会打印**。
> 不要把手里的 Key 贴进聊天；在本机终端里设置即可。

## 1. 一条命令

PowerShell 当前窗口：

```powershell
$env:DE_LLM_API_KEY = "sk-..."    # 只在这个窗口生效
cd F:\dsh_space\enterprise-agent
.\scripts\run-real-eval.cmd
```

也可以把 Key 放进仓库根目录的 `.env`（已被 `.gitignore` 忽略，脚本不会打印值）：

```dotenv
DE_LLM_API_KEY=sk-...
```

`Settings.from_env()` 会自动加载 `.env`（`DE_ENV_FILE` 可改路径，进程环境变量优先）。runner 也会读它，所以直接：

```powershell
.\scripts\run-real-eval.cmd
```

如果想让 `python -m de_agent eval` 直接走真实模型，取消 `.env` 里 `DE_LLM_BACKEND=openai_compat` 等行的注释；只跑 runner 则不需要。

先看会做什么、不花钱：

```powershell
.\scripts\run-real-eval.cmd -DryRun
```

先花几十秒验证真实链路（只跑前 3 条，不跑消融）：

```powershell
.\scripts\run-real-eval.cmd -Limit 3
```

确认没问题再跑全量：

```powershell
.\scripts\run-real-eval.cmd
```

脚本会逐条打印 `[eval] 1/15 ...`、`[task] 1/30 ...` 到控制台；如果看到“终止批处理操作吗”，那是你按了 Ctrl+C，不是脚本卡死。

## 2. 端点与模型

默认 `deepseek-chat`（工具调用稳、速度快）。要更强推理：

```powershell
.\scripts\run-real-eval.cmd -Model deepseek-reasoner -SmallModel deepseek-chat
```

| 服务 | 参数 |
|---|---|
| DeepSeek（默认） | `-BaseUrl https://api.deepseek.com/v1 -Model deepseek-chat -SmallModel deepseek-chat` |
| 通义 Qwen | `-BaseUrl https://dashscope.aliyuncs.com/compatible-mode/v1 -Model qwen-max -SmallModel qwen-plus` |
| 本地 Ollama | `-BaseUrl http://127.0.0.1:11434/v1 -Model qwen2.5:14b -SmallModel qwen2.5:7b`，Key 随便填 `ollama` |
| vLLM / 其他 | 任何 OpenAI 兼容 `/v1/chat/completions` |

如果 `deepseek-reasoner` 在工具调用上报 400，换回 `deepseek-chat`：

```powershell
.\scripts\run-real-eval.cmd -Model deepseek-chat -SmallModel deepseek-chat
```

## 3. 它跑什么

1. **smoke**：`de_agent ask "年假有几天？"`，先抓 401/402/模型不支持 tool calls；
2. **15 条 eval**：`de_agent eval`（默认跳过消融，省调用）；
3. **30 条任务级 eval**：`de_agent taskeval`。

要完整跑消融实验（会多出分解开关的额外调用），加 `-IncludeAblation`：

```powershell
.\scripts\run-real-eval.cmd -IncludeAblation
```

报告落到独立目录，不覆盖离线报告：

```
var/index/real/
├── eval_report.md
├── eval_summary.json
├── task_eval_report.md
└── task_eval_summary.json
```

默认 `DE_EMBED_BACKEND=hashing`，先看“模型变强”的效果；语义检索另开一轮：

```powershell
$env:DE_EMBED_BACKEND = "openai_compat"
$env:DE_EMBED_MODEL = "text-embedding-v3"
```

## 4. 对比基线

离线 FakeLLM 的任务级基线：

| 指标 | FakeLLM |
|---|---|
| 通过 | 20 / 30（66.7%） |
| badcase | `tool_missing` 6，`generation` 4 |
| 15 条 eval | intent/tool/Recall 100%，keyword 86.7% |

真实模型重跑重点看：

- `tool_missing` 是否下降（FakeLLM 的规划正则是主要瓶颈）；
- `generation` 是否下降（真实模型应能复述知识库数字）；
- 延迟/成本是否在可接受范围；
- 有没有新的 `tool_error`（模型调用工具但参数不合法）。

**不要改测试或语料去凑 100%**；有 badcase 才有面试可讲的东西。

## 5. 成本

45 个问题，每个 3–6 次模型调用，prompt 约 1–2k tokens、completion 几百 tokens。
DeepSeek/Qwen 一轮通常**几元人民币以内**；本地 Ollama 只花电费。先用 `-SkipTaskEval` 只跑 15 条更省。

## 6. 常见错误

| 现象 | 处理 |
|---|---|
| 401 / invalid key | Key 没设进当前窗口，或复制时带了空格 |
| 402 / insufficient balance | 充值或换 Ollama |
| 400 tool_calls / function calling | `-Model deepseek-chat -SmallModel deepseek-chat`，或换支持 tools 的模型 |
| 超时 | `-BaseUrl` 是否可达；脚本默认 timeout 90s |
| 计划 JSON 解析失败 | `extract_json` 会处理 fenced JSON；持续失败就换模型 |
| 成本超预期 | `-SkipTaskEval` 先跑 15 条；`DE_MAX_ITERATIONS=4` |

## 7. 跑完之后

把 `var/index/real/task_eval_report.md` 和 `eval_report.md` 对比离线版本：

- 如果真实模型明显更好，更新 README 的指标，并把报告复制到 `docs/real_*`；
- 如果反而更差，先看 badcase 是工具参数、上下文还是模型推理，再决定改 prompt/代码；
- 只有真实跑出来的数字才写进简历。

这一步做完，项目故事就从“离线 100%”变成“有真实模型 A/B 数据”。

## 8. Key 安全

不会从代码路径泄露：

- Key 只存在 `Settings.llm_api_key` 和 HTTP `Authorization` 头；`Settings` 的 `repr` 已隐藏它；
- 评测报告、trace、回答、`stats()` 都不包含 Key；
- `.env` 与 `.env.*` 已被 `.gitignore` 忽略（`.env.example` 例外）；
- `run-real-eval.ps1` 只从环境变量/`.env` 读，不打印、不写盘。

但下面这些**代码之外**的渠道要注意：

| 渠道 | 风险 | 做法 |
|---|---|---|
| PowerShell 历史 | `$env:DE_LLM_API_KEY="sk-..."` 会进 `ConsoleHost_history.txt` | 优先用 `.env`；或事后清历史 |
| 聊天/截图 | 贴 Key 或截到 Key 会泄露 | 不要贴；泄露后立即在平台吊销/轮换 |
| `.env` 文件 | 明文；同步盘/备份/压缩包会带走 | 不放进 OneDrive 同步目录；权限收紧 |
| Git | `.gitignore` 只防未来，不防 `git add -f` 或历史 | 已提交要立刻轮换并清历史 |
| 日志/报错 | 正常不会带 Key；第三方错误体可能回显请求信息 | 分享日志前扫一遍 |
| HTTP 端点 | `BaseUrl` 用 `http://` 时 Key 明文传输 | 远程必须 HTTPS；Ollama 本机可接受 |
| 共享机器 | 管理员/调试器可读进程环境 | 用限额 Key，用完轮换 |

验证忽略规则：

```powershell
git check-ignore -v .env
git status --short        # 不应出现 .env
```
