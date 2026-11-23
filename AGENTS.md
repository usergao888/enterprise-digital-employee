# AGENTS.md

This file tells coding agents (and new contributors) how to work in this repo.

## What this is

`enterprise-agent` is a Python AI Agent for internal enterprise Q&A:
state-graph orchestration, hybrid RAG, JSON-Schema tool calling, human approval
for mutating tools, citation enforcement, and multi-model cost routing.
The custom `de_agent/graph.py` engine is the zero-dependency fallback; the real
LangGraph runtime is optional (`DE_GRAPH_RUNTIME=auto|custom|langgraph`).

## Layout

- `de_agent/` — agent, RAG, tools, graph runtimes, CLI, HTTP server
- `de_agent/skills.py` — SKILL.md loader (metadata first, body on demand)
- `de_agent/context.py` — token budget, tool-output pruning, per-thread scratchpad
- `de_agent/harness.py` — multi-step task harness with budgets and resume
- `de_agent/task_eval.py` — 30-task eval + badcase attribution
- `de_agent/tracing.py` — per-ask OTel-shaped JSON traces
- `de_agent/permissions.py` — read_only / ask / auto tool policy
- `de_agent/sandbox.py` — hardened Docker command + degraded local runner
- `de_agent/mcp_server.py` — optional MCP server (read-only tools only)
- `skills/<name>/SKILL.md` — procedural knowledge, not tools
- `data/kb/` — markdown knowledge base (fictional policies)
- `tests/` — stdlib `unittest`; `var/` is disposable runtime state
- `scripts/` — bootstrap / smoke scripts

## Commands

```bash
python -m de_agent ingest
python -m de_agent ask "一线城市出差住宿每晚能报多少？" --trace
python -m de_agent chat          # HITL: confirm with y/N
python -m de_agent eval          # writes var/index/eval_report.md
python -m de_agent taskeval      # 30-task eval + badcase buckets
python -m de_agent eval --json   # pure JSON on stdout
python -m de_agent serve
python -m unittest discover -s tests -t .
```

Windows bootstrap for the real LangGraph runtime:

```powershell
.\scripts\bootstrap-langgraph.cmd
```

Real-model eval (needs `DE_LLM_API_KEY` in the environment or a gitignored `.env`):

```powershell
.\scripts\run-real-eval.cmd -DryRun
.\scripts\run-real-eval.cmd -Limit 3
.\scripts\run-real-eval.cmd
```

## Runtime rules

- `DE_GRAPH_RUNTIME=auto` (default): LangGraph if installed, else custom.
- `DE_GRAPH_RUNTIME=custom`: force the bundled engine (used by most tests).
- `DE_GRAPH_RUNTIME=langgraph`: require the real runtime.
- LangGraph checkpoints: `var/index/langgraph_checkpoints.sqlite`.
- `DE_REQUIRE_CONFIRMATION=true`: `create_ticket` stops at the `approval` node;
  never claim a write happened before it is confirmed.

## Safety rules

- Mutating tools (`create_ticket`) must stay behind confirmation. Do not expose
  them as read-only MCP tools.
- `DE_PERMISSION_MODE=read_only|ask|auto` is enforced in `ToolRegistry`; do not
  bypass it by calling tool handlers directly.
- Untrusted code goes through `sandbox.docker_command`; `run_python` is a
  degraded local fallback, not a security boundary.
- Never use `eval()` for calculator input; keep the AST whitelist.
- Keep SQL single-statement and read-only; do not weaken `guard_sql`.
- Do not commit `var/`, `.env`, API keys, or checkpoint databases.
- Do not run generated/foreign code outside the sandbox described in docs.

## Editing skills

- One folder per skill: `skills/<name>/SKILL.md`.
- Frontmatter keys: `name`, `description`, `when_to_use`, `version`,
  `allowed-tools`.
- Body = steps, tool calls, output contract, failure handling.
- Metadata is always available to the planner; the body is injected only when
  the skill is selected. Keep bodies under ~80 lines.
- After editing skills, run `python -m unittest tests.test_skills -v`.

## Testing expectations

- New behavior needs a unit test; optional dependencies (LangGraph, MCP) must be
  `skipUnless`-guarded.
- Keep the custom-engine suite green (`graph_runtime="custom"` in tests that
  assert checkpoint files).
- Update `README.md` numbers only from an actual run.
