# 企业数字员工 Agent（Enterprise Digital Employee）

![tests](https://github.com/usergao888/enterprise-digital-employee/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> 一个**能离线跑、有评测数据、可拆开讲**的 Python AI Agent 工程：LangGraph 风格状态图编排 + 混合检索 RAG + JSON-Schema Tool Calling + 向量库 + 多模型路由 + 引用溯源 + 多智能体审核闭环。

这不是一个"调用 OpenAI 接口的 demo"。它把企业知识问答里真正会卡住的地方都做了一遍：中文分块、双路召回、角标可追溯、工具参数校验、写完操作的 SLA、审核不通过自动重写、以及**用代码跑出来的指标**而不是简历上拍的百分比。

```
Python 3.10+ · numpy 是唯一硬依赖 · 70 个单元/集成测试 · 无需 GPU / 无需 API Key
```

---

## 一、它和你简历上三个项目的对应关系

我没有另起炉灶，而是把两份简历里**最有技术含量、最经得起追问**的部分合成一个工程 —— 因为面试只能深挖一两个项目，散着放三个"AI 项目"反而每个都问不深。

| 简历原项目 | 原本的技术栈 | 本工程中的落地 | 为什么保留 |
| --- | --- | --- | --- |
| **企业数字员工 AI Agent 系统** | Java + Spring AI + LangChain + RAG | `rag.py` + `tools.py` + 引用溯源 | 直接命中 JD 的 RAG / Tool Calling / 向量库三件套 |
| **多智能体协同办公助手** | 规划/执行/审核三 Agent | `agents.py` 的 plan→execute→review 图 | 多角色分工是 Agent 岗最常问的架构题 |
| **AI 零代码应用生成平台** | LangGraph4j + 质检重试 | `graph.py` 的自研状态图 + `node_review` 的**质检不通过自动重写** | LangGraph 是 JD 点名的框架；质检重试机制从"生成代码"搬到"生成答案" |
| 校园一站式服务平台 | Spring Cloud 微服务 | 只保留 `sql_query` 工具里的业务库建模 | 与 AI 岗位相关性低，不占简历篇幅 |

**语言从 Java 换成了 Python。** 这是有意为之：JD 第一句就是 "Python + 至少一种主流 Agent 开发框架"。Java 版的 LangChain4j 经历写在简历上，会被直接判定"不匹配"。

---

## 二、架构

```
                        ┌──────────────────────────────────────────────┐
   question ───────────► │  guard 节点：注入检测 / PII 脱敏 / 审计         │
                        └───────────────┬──────────────────────────────┘
                                        │ route_after_guard
                     blocked ───────────┴─────────── continue
                        │                              │
                        │                    ┌─────────▼─────────┐
                        │                    │  plan  规划智能体  │  拆解意图 → intent + steps
                        │                    └─────────┬─────────┘
                        │                    ┌─────────▼─────────────────────┐
                        │                    │  execute  执行智能体（工具循环）│◄──────┐
                        │                    │  knowledge_search / sql_query │       │
                        │                    │  calculator / create_ticket   │       │
                        │                    │  employee_profile / time      │       │
                        │                    └─────────┬─────────────────────┘       │
                        │                    ┌─────────▼─────────┐                  │
                        │                    │  review 审核智能体 │  verdict=revise ─┤
                        │                    │  + 确定性引用校验  │  (最多 max_revisions 次)
                        │                    └─────────┬─────────┘                  │
                        │                    ┌─────────▼─────────┐                  │
                        └───────────────────►│ finalize 组装出处  │                  │
                                             └─────────┬─────────┘                  │
                                                       ▼                            │
                                                   answer + [1][2] + usage ──────────┘
```

检索链路：

```
文档 → 结构化分块(按标题) → 窗口切分(带句子重叠)
     ├─ 向量检索 (numpy cosine / Chroma / Qdrant)
     └─ BM25 (中文 unigram + bigram，无 jieba 依赖)
                └→ RRF 倒数排名融合 → 重排(词面/结构，可换 cross-encoder) → top-k + 角标
```

---

## 三、快速开始

```bash
cd enterprise-agent
python -m de_agent ingest          # 建立知识库索引（8 篇制度文档 / 59 个 chunk）
python -m de_agent demo            # 跑一遍 8 个典型问题
python -m de_agent ask "一线城市出差住宿每晚能报多少？" --trace
python -m de_agent chat            # 多轮对话
python -m de_agent eval            # 跑评测集，产出 var/index/eval_report.md
python -m de_agent serve           # 打开 http://127.0.0.1:8710 有网页版
python -m unittest discover -s tests -t .    # 70 个测试
```

接真实模型（默认是离线后端，不需要 Key）：

```bash
export DE_LLM_BACKEND=openai_compat
export DE_LLM_BASE_URL=https://api.deepseek.com/v1
export DE_LLM_API_KEY=sk-xxx
export DE_LLM_MODEL=deepseek-reasoner DE_SMALL_MODEL=deepseek-chat
# 换语义 embedding：DE_EMBED_BACKEND=openai_compat DE_EMBED_MODEL=text-embedding-v3
```

---

## 四、评测结果（`python -m de_agent eval` 实跑）

15 条业务问题，覆盖制度问答 / 数据查询 / 工单 / 计算 / 安全拦截：

> 完整报告（可复跑）：[docs/eval_report.md](docs/eval_report.md)

| 指标 | 数值 |
| --- | --- |
| 意图路由准确率 | 100.0% |
| 工具选择准确率 | 100.0% |
| 检索 Recall@5 / MRR | 100.0% / 1.000 |
| 多文档覆盖率 | 100.0% |
| 引用合规率（角标存在且可回溯） | 100.0% |
| 答案关键信息命中 | 86.7% |
| P50 / P95 延迟 | 7.1 / 9.9 ms（离线后端，仅反映编排+检索开销） |
| 多模型路由成本 | ¥0.0833 vs 全量走大模型 ¥0.1917，**省 56.6%** |

检索策略消融（18 条含精确数字/编号与改写类问题）：

| 检索策略 | Recall@1 | MRR | 平均耗时 |
| --- | --- | --- | --- |
| 纯向量检索 | 88.9% | 0.944 | 1.36 ms |
| 纯 BM25 | 88.9% | 0.944 | 1.59 ms |
| 双路召回 + RRF | 94.4% | 0.972 | 1.39 ms |
| 双路召回 + RRF + 重排（默认） | 94.4% | 0.972 | 2.96 ms |

**关于这组数字的诚实说明**（面试被问到时必须主动讲，否则会被当成吹牛）：

1. 离线模式用的是**哈希向量**（非语义模型），所以"向量 vs BM25"的差距被系统性低估。换成真实 embedding 重跑同一脚本，语义召回的增益才会显出来 —— 脚本里 `retrieval_ablation()` 就是为这件事准备的。
2. 86.7% 不是 100%，因为抽取式答案只取最有依据的 3 句，有两题漏掉了关键数字（"身份证""电源"）。**我知道它为什么漏**：这两处的答案词（身份证、电源）根本不在问题里出现，词面匹配跨不过去 —— 换真实 LLM 或语义 embedding 才是正解。
3. **重排在本语料上没有额外 Recall@1 增益**（94.4% 持平，耗时翻倍）。融合已经把正确的 chunk 排到第一，重排只能在更大的候选池里发挥作用。它保留为可开关项，默认开、可关。
4. 延迟是离线后端的开销，**不代表真实模型耗时**，报告里已显式标注。

### 三个"我做了 A/B 之后才知道"的记录

这三条比涨几个点更有价值，因为它们是**推翻自己假设**的证据：

| 我做了什么 | 假设 | 实测结果 | 处置 |
| --- | --- | --- | --- |
| **Query Decomposition**（复合问题拆子查询 + 加权 RRF 融合） | 拆子查询能提升复合问题的答案覆盖率 | 多文档覆盖率、关键信息命中**都没有变化**；8 篇文档、59 个 chunk 的语料上，单次 top-5 本来就跨文档 | 功能保留、**默认关闭**（`DE_ENABLE_DECOMPOSITION`），等语料上量再开 |
| **`keyword_overlap` 去掉单字匹配** | 去掉噪声能提升重排质量 | 短句"拨打分机 8000 催单"不再靠一个"机"字挤掉正确答案；"电脑开不了机"现在会引用"先确认电源" | **保留**（这是真 bug 修复） |
| **检查点无上限 + 每节点全量重写** | 无界存储没什么影响 | 跑几百次评测后 P50 从 16 ms 退化到 77 ms，文件越长越慢 | 加线程上限 64 + 每 4 个节点批量落盘，恢复 **7 ms** |

---

## 五、五个关键设计决策

**1. 为什么自己写 `graph.py`，而不是直接 `pip install langgraph`？**
为了能在没有网络、没有 Key 的机器上跑测试，同时证明我理解 LangGraph 到底提供了什么。`de_agent/graph.py` 实现了 StateGraph 的节点/条件边/重试策略/递归上限/检查点/流式事件，并且 `node_map()` 把节点暴露成普通可调用对象，`langgraph_adapter.py` 可以直接把它们挂到**真·LangGraph** 上 —— 换运行时是改接线，不是改业务。
> 面试价值：能讲清 checkpointer 解决"断点续跑"、条件边解决"分支路由"、recursion_limit 防"死循环"，比只会写 `graph.add_node` 强一个档。

**2. 为什么混合检索而不是纯向量？**
企业问答里大量查询是精确匹配：`500 元`、`E1003`、`分机 8000`、`180 天`。向量模型对这些数字/编号的区分度很差，而 BM25 恰好最强。反过来"住一晚最多能报多少"这种改写，BM25 又抓不住。RRF 不需要两路分数同尺度，融合成本几乎为零。消融实验里纯向量的 Recall@1 就是比 BM25 低 5.5 个点。

**3. 调用框架为什么不用 `eval()`？**
`calculator` 工具用 AST 白名单，只认数字和四则/乘方节点。`eval("__import__('os').system(...)")` 是 Agent 落地的第一个真实事故来源 —— 当模型能被 prompt injection 影响时，"模型输出的表达式"必须当成不可信输入。同理 `guard_sql()` 强制单条 SELECT、拒绝写关键词、自动补 LIMIT。

**4. 审核智能体为什么是双通道（模型 + 规则）？**
纯模型审核会"和稀泥"：它看到有角标就放过，但角标可能指向一段根本不支撑结论的原文。所以 `citation_report()` 用**确定性规则**做第二道：角标指向的 chunk 是否包含问题关键词。模型负责"答没答到点上"，规则负责"引没引对地方"，两者取并集才判定 revise。

**5. 多模型路由怎么证明它真省钱？**
每个 stage 声明负载档位：plan/review 走强模型（要推理），execute 走便宜模型（长上下文、大量 token）；超长问题升级到强模型。`MeteredRouter` 对**每次调用**按 stage 记账，同时算出"全量走强模型"的基线成本，差值就是节省率。56.6% 是这么跑出来的，不是估的。

**6. 写操作为什么要单独造一个确认门？**
Agent 最危险的不是答错，是**在没人同意的情况下把事办了**。`create_ticket` 标了 `mutating=True`，开启确认后 `ToolRegistry.call()` 不执行、只返回 `pending_confirmation + action_id`；`node_execute` 拿到这个结果就停止工具循环，把"待确认动作"交给用户，`agent.confirm(id)` 才真正落库。测试里专门断言了"未确认前，答案里不得出现工单号，tickets.json 不得新增"。
> 这正是 LangGraph `interrupt()` 的语义，只是这里用应用层实现，换真 LangGraph 时可以直接替换。

---

## 六、与岗位 JD 的逐条对应

| JD 要求 | 本工程证据 |
| --- | --- |
| **Python** | 全量 Python 3.10+，`de_agent/` 12 个模块，70 个测试 |
| **至少一种主流 Agent 框架（LangChain/LangGraph/Dify）** | `graph.py` 自研 LangGraph 同构引擎 + `langgraph_adapter.py` 可直接切真 LangGraph |
| **RAG** | 结构化分块 → 双路召回 → 加权 RRF → 重排 → 引用溯源 → 消融实验（`rag.py`） |
| **Tool Calling** | 6 个工具、标准 JSON Schema、参数校验、异常兜底、写操作标记 + **人工确认门**（`tools.py`） |
| **向量库** | 内存版(numpy)+JSON 持久化，另附 Chroma / Qdrant 适配器（`vectorstore.py`） |
| 加分：大模型应用 | 多模型路由 + 成本核算 + 流式 SSE + 会话记忆 + Prompt 注入防护 |
| 加分：工程规范 | 类型注解、docstring 说明取舍、`unittest` 覆盖、可观测的 step trace、A/B 实验结论可复跑 |

---

## 七、面试可能追问（附答法要点）

1. **chunk 大小怎么定的？** 520 字 + 90 字重叠。企业制度常以"条款"为语义单元，先按 `##` 标题切，再在标题内做句子级窗口，避免把"适用范围"和"例外条款"切散。重叠保证跨窗口的条件句不被截断。可调，且有 `chunk_size` 参数和消融脚本支持复测。
2. **RRF 的 k 为什么取 60？** 来自原论文的经验值，作用是压平头部差距，让两路排名的"共识"比单路的高分更重要。k 越小越激进。这个参数在 `settings.rrf_k` 里。
3. **怎么防止模型瞎编？** 三层：① prompt 限定"只用工具返回的原文"；② 审核节点用模型判"答没答到点"；③ 确定性引用校验判"引的原文撑不撑得住"，`support_rate` 低于阈值就算不通过。
4. **工具调用失败怎么办？** 工具异常不抛出，转成 `{"error": ...}` 回给模型，模型有下一轮机会换工具或改参数；`execute` 节点本身带 `retry=1`；再不行返回兜底话术并把 trace 记进日志。
5. **多轮对话怎么带记忆？** 滑动窗口 4 轮 + 超阈值触发摘要压缩，摘要作为 system message 注入，避免上下文无限膨胀。
6. **怎么评估 RAG 好不好？** 见 `evaluation.py`：检索侧 Recall@k / MRR（有标注的期望文档），生成侧引用合规率 / 关键信息命中率，系统侧 P50/P95 延迟与成本。
7. **如果线上量级上来呢？** 向量库换 Qdrant（已留适配器）、embedding 换 batch 接口、把同步工具改异步、知识库按部门做租户隔离、审核节点做成可配置开关以降延迟。
8. **LangGraph 的 checkpointer 解决什么？** 断点续跑与人机协同：`create_ticket` 这种写操作可以 `interrupt()` 等人工确认，确认后从断点继续，而不是重跑整条链。
9. **Agent 怎么防止越权写数据？** `DE_REQUIRE_CONFIRMATION=true` 后，标记 `mutating=True` 的工具不会立即执行，而是返回 `pending_confirmation` + `action_id`，图在拿到这个结果时就停下来让 Agent 请求确认；`agent.confirm(action_id)` 才真正落库，`cancel` 则丢弃。测试里 `TestHumanInTheLoop` 覆盖了"未确认前不得声称已创建工单"。
10. **你的功能都有效果吗？** 不都有效。子查询分解我做了完整实现和 A/B，在本语料上**没有增益**，所以默认关闭；重排在本语料上也只有耗时没有 Recall 提升。反过来，单字匹配噪声修复让答案不再引用错段落，checkpointer 上限修复把 P50 从 77ms 拉回 7ms —— 这三条都写在评测报告里。

---

## 八、目录结构

```
enterprise-agent/
├── de_agent/
│   ├── config.py              # 环境变量配置 + 模型价目表
│   ├── textutil.py            # 中文分词(unigram+bigram)/句子切分/覆盖率
│   ├── embeddings.py          # 哈希向量(离线) + OpenAI 兼容 embedding
│   ├── vectorstore.py         # 内存向量库 + Chroma/Qdrant 适配器
│   ├── rag.py                 # 分块 / BM25 / RRF / 重排 / 引用校验
│   ├── tools.py               # 工具注册表 / JSON Schema 校验 / 6 个内置工具
│   ├── graph.py               # LangGraph 同构状态图引擎
│   ├── llm.py                 # OpenAI 兼容客户端 + 离线 FakeLLM + 多模型路由
│   ├── agents.py              # 三个智能体节点 / 路由 / 记忆 / 护栏
│   ├── evaluation.py          # 评测集 / 消融 / 报告生成
│   ├── cli.py                 # ingest/ask/chat/eval/graph/serve/demo
│   ├── server.py              # stdlib HTTP + SSE + 网页版
│   └── langgraph_adapter.py   # 切到真 LangGraph
├── data/kb/                   # 8 篇制度文档（差旅/考勤/IT/财务/入职/薪酬/产品）
├── tests/                     # 70 个测试（含 HTTP/SSE 端到端）
├── var/index/                 # 索引、工单、检查点、评测报告（运行产物）
├── requirements.txt · pyproject.toml · .env.example
└── README.md
```

---

## 九、已知边界与下一步

- 离线 `FakeLLM` 是**规则替身**，只用于跑通链路与做回归测试；它的"意图判断"是正则，不代表真实模型的泛化能力。
- 哈希向量不是语义模型，检索质量的真实上限要在接入 `text-embedding-v3` / BGE-M3 后重测。
- 未做：多轮指代改写（"那病假呢"这类省略主语的追问）、知识库增量更新与删除、RAGAS 式 faithfulness 评测、真实模型下的端到端延迟。
- 未做：`action_id` 持久化 —— 待确认动作存在内存里，进程重启就失效，要真正用于生产得落库并加过期时间。
- `create_ticket` 的确认目前是"应用层 interrupt"（图停下来 + 显式 confirm），真 LangGraph 下可换成原生 `interrupt()`，adapter 里已注明。

## 十、License

MIT。示例文档为虚构企业制度，不含任何真实公司数据。
