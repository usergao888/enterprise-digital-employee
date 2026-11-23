# Agent 知识点与面试题

> 只保留两件事：要学的知识点、面试会问到的问题。答案在仓库的 `README.md`、`docs/` 和代码里。

## 一、知识点

### 1. LangGraph / 状态图
- `StateGraph`、`State`/`TypedDict`、`Annotated` reducer、node/edge/conditional edge、`START`/`END`
- `compile(checkpointer=...)`、`recursion_limit`、`thread_id`
- `MemorySaver` / `SqliteSaver` / `PostgresSaver`；`get_state` / `update_state` / time travel
- `interrupt()` / `Command(resume=...)`；恢复时节点从头重跑；interrupt 前不能有副作用
- `Send`（map-reduce/fan-out）、subgraph、`stream_mode`（values/updates/messages/custom）
- LangGraph4j：`StateGraph`、`AsyncNodeAction`、`cloneState`、状态必须可序列化

### 2. Agent Runtime / Harness
- Agent loop：observe → think → act；停止条件（答案/最大步数/token/成本/无进展/人工中止）
- 状态存储、事件日志、checkpoint/resume、幂等键
- 预算：`max_steps`、`max_cost`、`max_tokens`、`max_wall_clock`
- HITL：暂停、审批、取消、超时、恢复；action 过期与多实例锁
- sub-agent 并发与上下文隔离；写操作留在主 Agent

### 3. Context Engineering
- 上下文是预算，不是垃圾桶；context rot / lost-in-the-middle
- 分 section 预算：system / skills / memory / retrieval / tool / history
- compaction：摘要、保留决策/待办/引用、丢原始工具输出
- scratchpad：把进度写到窗口外；per-thread 笔记
- 工具输出裁剪：schema + head/tail，裁剪后仍是合法 JSON
- JIT 检索、按需加载；prompt caching
- 记忆：短时窗口、摘要、长期记忆、记忆检索与隐私

### 4. RAG
- 分块：固定窗口、结构感知、语义分块、parent-child、overlap、metadata
- 中文切词：jieba / unigram+bigram / 字级 BM25 的取舍
- BM25：`k1`、`b`、IDF 变体；为什么对数字/编号强
- Embedding：维度、归一化、cosine/dot/L2；BGE-M3、text-embedding-v3
- ANN：HNSW（`M`/`efConstruction`/`efSearch`）、IVF（`nlist`/`nprobe`）、PQ；召回/延迟/内存取舍
- 向量库：Qdrant/Chroma/Milvus/FAISS/pgvector；payload filter、upsert/delete、持久化、多租户
- 混合检索：dense + sparse；RRF 公式与 `k=60`；加权 RRF
- 重排：cross-encoder、bge-reranker、候选池与延迟
- 查询侧：改写、指代消解、HyDE、multi-query、decomposition、路由
- 引用/归因：角标、citation report、支撑率
- 评测：Recall@k、MRR、NDCG、context precision/recall、faithfulness、answer relevancy、RAGAS、LLM-as-judge
- 进阶：GraphRAG、LightRAG、RAPTOR、Self-RAG/CRAG、agentic RAG
- 运维：增量索引、删除、版本、重嵌入、ACL

### 5. Tool Calling / MCP / Skills
- JSON Schema：required/type/enum/min/max/length；参数校验
- Function calling wire format：`tools`、`tool_calls[].id`、`tool` role、arguments 是 JSON 字符串
- 并行 tool calls、流式参数增量、非法 JSON 修复、工具异常转结构化错误
- 工具设计：少而好用、幂等、超时、allowlist、read-only vs mutating
- MCP：JSON-RPC、tools/resources/prompts、stdio/streamable HTTP、stateless core、OAuth/OIDC、registry
- MCP v1 `FastMCP` → v2 `MCPServer`；只读工具才适合暴露
- Agent Skills：`SKILL.md`、frontmatter、渐进加载（元数据常驻/正文按需）
- `AGENTS.md`：仓库级指令；Skills 与 tools 的职责边界
- A2A：Agent Card、task lifecycle、JSON-RPC/SSE/webhook；MCP vs A2A

### 6. Eval / Observability
- 任务级 eval：golden set、pass@1、成功率、工具准确率、轨迹、人工接管
- LLM-as-judge：rubric、校准、位置/冗长/自偏好偏差
- badcase 归因：guard / tool_missing / tool_error / retrieval / citation / generation
- 指标：P50/P95、成本、token、缓存命中、失败分类
- Tracing：trace/span、OTel GenAI、token/成本、Langfuse/LangSmith/Phoenix
- CI eval gate：阈值、回归、真实模型 A/B

### 7. 安全 / 生产
- Prompt injection：直接/间接（文档、网页、工具返回）、jailbreak
- 防御：指令层级、spotlighting、输出校验、最小权限、HITL、出网控制
- 权限模式：`read_only` / `ask` / `auto`；写操作审批与审计
- 沙箱：Docker `--network none --read-only --cap-drop ALL --pids-limit`、seccomp、gVisor、Firecracker、e2b
- 数据：PII、密钥、租户隔离、审计日志、保留策略
- 代码生成：XSS、CSP、路径穿越、依赖供应链
- 服务：FastAPI/SSE、异步、超时、背压、缓存、限流（Lua 原子性）、Docker/K8s
- 成本：模型路由、预算、缓存、降级/回退

### 8. 模型 / 算法（选学）
- Transformer：attention Q/K/V、多头、FFN、残差/LN、RoPE
- 分词：BPE/SentencePiece；训练：SFT/DPO
- PEFT：LoRA/QLoRA（rank、alpha、4-bit NF4）
- 推理：KV cache、PagedAttention/vLLM、continuous batching、量化
- Embedding/reranker 微调：对比学习、hard negative、C-MTEB

## 二、面试题

### LangGraph / 编排
1. 为什么自研状态图后又迁到真 LangGraph？迁移时最大的坑是什么？
2. checkpointer 到底存什么？`thread_id` 怎么隔离会话？
3. `interrupt()` 恢复时，节点为什么会从头重跑？怎么保证不重复写？
4. 为什么把审批单独做成 `approval` 节点，而不是在 `execute` 里 interrupt？
5. `Command(resume=...)` 的恢复流程是什么？进程重启后靠什么恢复？
6. `Send` 和普通边有什么区别？什么场景用 subgraph？
7. `stream_mode` 的 values/updates/messages/custom 各看什么？
8. recursion_limit 防什么？死循环还有哪些检测手段？
9. 条件边路由函数返回未映射的 key 会怎样？
10. LangGraph4j 的 channel 默认值和 Serializable 坑是什么？

### Agent Runtime / Context
11. 你的 Agent loop 怎么停？预算在哪一层扣？
12. Harness 的 `state.json` 和 LangGraph checkpoint 分别负责什么？
13. 20 步任务中途 kill，重启后怎么续跑？哪些操作必须幂等？
14. 多 Agent 什么时候不该用？sub-agent 的上下文怎么隔离？
15. 上下文窗口变大了，为什么还需要压缩？
16. 你的 ContextBudget 怎么分 section？超限先丢什么？
17. scratchpad 和普通记忆的区别？为什么不直接塞回 prompt？
18. 工具输出裁剪怎么保证还是合法 JSON？
19. 多轮指代（“那病假呢”）怎么改写？改写失败怎么兜底？
20. 记忆的隐私和租户隔离怎么做？

### RAG
21. chunk 大小怎么定？为什么 520/90？先按标题切有什么好处？
22. 为什么混合检索而不是纯向量？BM25 在中文里怎么切？
23. RRF 公式是什么？`k=60` 的作用？
24. HNSW 的 `M`/`efConstruction`/`efSearch` 分别调什么？
25. cosine、点积、L2 怎么选？为什么要归一化？
26. 向量库和 FAISS 的区别？Qdrant 的 payload filter 怎么用？
27. rerank 为什么常有效？为什么在你的语料上没有增益？
28. cross-encoder 为什么不能直接拿来做召回？
29. query decomposition 什么时候有效？你实测为什么关闭？
30. 怎么防止模型瞎编？引用校验做了什么？
31. 引用角标指向的原文不支撑结论，怎么发现？
32. RAGAS 的 faithfulness/answer relevancy/context precision 各测什么？
33. LLM-as-judge 有哪些偏差？怎么校准？
34. GraphRAG 的 local/global search 区别？成本为什么高？
35. 知识库增量更新/删除/重嵌入怎么做？

### Tool Calling / MCP / Skills
36. Function calling 的完整消息流是什么？`tool_call_id` 怎么回填？
37. 模型不调用工具/参数幻觉/并行调用，分别怎么处理？
38. 流式 tool call 的 arguments 怎么拼接？
39. 为什么 calculator 用 AST 白名单而不是 `eval`？
40. SQL 只读守卫怎么做？字符串黑名单够吗？
41. 工具 schema 怎么写才能让模型少犯错？
42. MCP 的 tools/resources/prompts 各是什么？stdio 和 HTTP 怎么选？
43. MCP 2026 的无状态核心改了什么？v2 为什么把 `FastMCP` 改名？
44. 为什么 `create_ticket` 不暴露成 MCP 工具？
45. Skills 和 tools 的区别？渐进加载具体怎么实现？
46. `AGENTS.md` 和 `SKILL.md` 的职责边界？
47. MCP 和 A2A 的区别？A2A 的 Agent Card 是什么？
48. MCP server 的鉴权、审计、限流怎么做？

### Eval / Trace / 生产
49. 为什么 15 条 eval 100%，30 条任务级只有 66.7%？badcase 怎么归因？
50. 任务级 eval 和 QA eval 的区别？轨迹怎么评？
51. 你的 trace 里有哪些 span？怎么判断是检索问题还是生成问题？
52. OTel GenAI 的 span 该带哪些属性？Langfuse 怎么接？
53. 线上 Agent 怎么排障？P50/P95、token、成本看什么？
54. 多模型路由省 56.6% 怎么算？基线是什么？
55. 真实模型重跑怎么设计？怎么防止 Key 泄露？
56. 缓存击穿/雪崩怎么防？限流为什么用 Lua？
57. SSE 和 WebSocket 怎么选？异步服务怎么背压？
58. Docker/K8s 部署 Agent 时，状态放哪？多实例怎么共享？
59. 间接 prompt injection 怎么防？沙箱参数各防什么？
60. 你的项目最大技术难点？如果重做会怎么改？

### 模型 / 算法（选答）
61. LoRA 的 rank/alpha 怎么选？QLoRA 的 4-bit 是什么？
62. vLLM 的 PagedAttention 解决什么？TTFT/TPOT 怎么优化？
63. embedding 微调的对比学习和 hard negative 怎么做？
64. SFT 和 DPO 的区别？数据质量怎么保证？
