# MCP Server

micro-model-agent can run as an MCP stdio server and expose the local tool loop to
Copilot or another MCP client.

The server accepts a startup repository root:

```bash
python -m micro_model_agent.interfaces.mcp_server \
  --repository-root /path/to/workspace
```

The same default can be supplied through `MICRO_MODEL_AGENT_REPOSITORY_ROOT`.
Individual MCP tool calls can still override `repository_root`.

For globally configured MCP servers, prefer creating or selecting a per-chat
workspace:

```text
micro_agent_init_workspace
path: "/path/to/chat-or-test-workspace"
name: "optional-human-name"
```

The tool returns a `workspace.id`. Pass that ID as `workspace_id` to
`micro_agent_start_trace`, `micro_agent_run_loop`, `micro_agent_stop_trace`,
`micro_agent_review_trace`, and debug trace tools. This avoids multiple Codex
chats sharing one global default directory.

## VS Code / Copilot Configuration

Create `.vscode/mcp.json` in this repository:

```json
{
  "servers": {
    "micro-model-agent": {
      "type": "stdio",
      "command": "wsl",
      "args": [
        "--cd",
        "/mnt/d/Projects/code/micro-model-agent",
        "--exec",
        "env",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        ".venv-wsl/bin/python",
        "-m",
        "micro_model_agent.interfaces.mcp_server"
      ]
    }
  }
}
```

This launches the server inside WSL so it can use direct Transformers inference.
Model resolution uses this order:

1. `micro_agent_run_loop` `base_model` / `adapter_path` arguments.
2. `MICRO_MODEL_AGENT_BASE_MODEL` / `MICRO_MODEL_AGENT_ADAPTER_PATH`.
3. The selected promoted adapter in `.micro_model_agent/config.json`.

If no adapter path is configured, the loop runs the resolved base model without
fine-tuning. A missing adapter is not replaced with a hard-coded fallback.

Select a promoted adapter for the repository before starting MCP:

```bash
uv run micro-agent promote list
uv run micro-agent promote select --artifact-id <artifact-id> --confirm
uv run micro-agent serve-mcp
```

## Exposed Tools

By default, the MCP server keeps the outer client's tool surface intentionally
small:

- `micro_agent_run_loop`: run the Qwen-backed orchestration loop.
- `micro_agent_init_workspace`: create or register a workspace directory and
  return a `workspace_id` for subsequent calls.

The agent loop is responsible for selecting the internal repository tools. This
keeps Copilot or another MCP host from seeing and directly choosing unnecessary
micro-model-agent tools.

When `.micro_model_agent/config.json` is missing, the server also exposes:

- `micro_agent_init`: initialize local micro-model-agent metadata for the
  repository.

`micro_agent_init` is idempotent. After it initializes the repo successfully, the
server removes it from the FastMCP tool registry and sends
`notifications/tools/list_changed` so clients that support live MCP tool refresh
can hide it without a restart. Some clients cache tools more aggressively; in
those cases, restart the MCP server or use the client's cached-tool reset command.

To force the init tool to be visible, set:

```text
MICRO_MODEL_AGENT_MCP_EXPOSE_INIT=1
```

Debug and inspection tools are hidden unless explicitly enabled:

- `micro_agent_builtin_tool`: execute one built-in repository tool directly.
- `micro_agent_read_trace`: read a stored workflow trace by id.
- `micro_agent_list_builtin_tools`: list built-in tool descriptions and defaults.

To expose them, set:

```text
MICRO_MODEL_AGENT_MCP_DEBUG_TOOLS=1
```

Default agent tools include read/search plus dry-run patch and file-write proposals:

```text
repo.search, repo.read, repo.semantic_search, repo.write_patch, repo.write_files, git.diff
```

Always pass MicroModelAgent tool names in `available_tools` and
`required_tools`, not Codex tool names. For example, use `repo.write_files` or
`repo.write_patch`, not `apply_patch`; use `test.run`, not `run_tests`. The MCP
boundary accepts a few legacy aliases for compatibility, but prompts and traces
should use the canonical names above.

Patch application remains dry-run unless `apply_patches` is explicitly true.
MCP model runs also default `expose_tool_schemas` to true so the model sees the exact
tool argument schemas. Prefer `repo.write_files` for greenfield scaffolds and
new files; use `repo.write_patch` for precise edits to existing files.

`micro_agent_run_loop` is bounded by both model turns and tool calls. The inner
prompt includes a `loop_budget` object with the current turn, remaining turns,
tool calls made, remaining tool calls, and whether the current turn is
final-response-only. Callers should set `max_turns` and `max_tool_calls`
deliberately for the task size instead of relying on a long MCP timeout. Results
include `turns_used`, `tool_calls_made`, and the configured `loop_budget` so the
consumer can tell whether the run ended naturally or ran into orchestration
limits.

Run profiles provide larger preset budgets:

| Profile | Turns | Tool calls | Max new tokens | Tool-result prompt chars |
| --- | ---: | ---: | ---: | ---: |
| `quick` | 12 | 8 | 2048 | 8000 |
| `standard` | 24 | 24 | 8192 | 16000 |
| `extended` | 48 | unlimited | 32768 | 32000 |

When `allow_test_run=true` and no explicit `test_command_name` /
`test_command_args` are supplied, MCP allowlists a default command named
`pytest` that runs `python3 -m pytest -q`. The native schema for `test.run`
includes `allowed_command_names`; the model must use one of those names exactly.
For repair prompts that mention pytest, failing imports, or verification, the
loop requires a passing `test.run` after the latest write before it accepts a
final response.

## Useful First Prompt

After starting the MCP server in Copilot Agent mode, try:

```text
Use micro_agent_run_loop to read README.md and summarize the project status in one sentence.
Use available_tools ["repo.read"], max_tool_calls 1, and max_turns 4.
```

The first model-backed call can take about 90 seconds while the 7B model loads.
The server process caches the loaded provider for later calls with the same
model settings.

## Base-Model Trace Collection

For real data collection, use the base coder model with explicit runtime schemas
instead of a trained adapter:

```text
Use micro_agent_run_loop with goal "<repo task>".
Set base_model "Qwen/Qwen2.5-Coder-7B-Instruct", use_adapter false,
apply_patches true when you want real edits, and include
the canonical tools needed for the task, such as `repo.search`, `repo.read`,
`repo.write_files`, `repo.write_patch`, `test.run`, or `git.diff`.
`expose_tool_schemas` is true by default.
```

For MCP runs, workflow traces are stored under the MCP server's repository root,
even when the loop operates on a registered workspace:

```text
.micro_model_agent/traces/workflows.jsonl
.micro_model_agent/traces/<trace-id>/metadata.json
.micro_model_agent/traces/<trace-id>/<request-step-id>/request.txt
.micro_model_agent/traces/<trace-id>/<request-step-id>/response.txt
```

Review labels are stored separately from the raw trace log:

```bash
uv run micro-agent dataset review-trace \
  --trace-id <trace-id> \
  --outcome accepted \
  --quality good \
  --reviewer-notes "Useful real task trace."
```

Export only reviewed traces when building a curated real-data dataset:

```bash
uv run micro-agent dataset export-traces \
  --label-mode reviewed \
  --outcome accepted \
  --quality good \
  --require-tool-call \
  --output .micro_model_agent/datasets/reviewed_real_traces.jsonl
```

Rejected raw traces should be used for analysis and later corrected examples,
not as direct SFT targets.

## Comparison Trace Sessions

When an MCP consumer such as Codex is connected, use comparison trace sessions to
record both the local model's shadow attempt and the consumer's actual work.

1. Start a session:

```text
micro_agent_start_trace
goal: "<repo task>"
context: "Any extra task context visible to the consumer."
```

2. Ask the local model to attempt the same task, passing the session id:

```text
micro_agent_run_loop
goal: "<repo task>"
comparison_session_id: "<session id>"
base_model: "Qwen/Qwen2.5-Coder-7B-Instruct"
use_adapter: false
expose_tool_schemas: true
```

3. The MCP consumer does the real work using its normal tools.

4. Stop the session with the consumer's actual result:

```text
micro_agent_stop_trace
session_id: "<session id>"
actual_summary: "What the consumer actually did."
changed_files: ["path/to/file.py"]
tests: ["pytest path/to/test.py"]
```

5. Review the comparison:

```text
micro_agent_review_trace
session_id: "<session id>"
local_model_quality: "good|mixed|bad|unknown"
consumer_quality: "good|mixed|bad|unknown"
comparison_notes: "Where the local model matched or diverged."
```

For MCP tools, comparison sessions are stored append-only under the MCP server's
repository root, while each session's `repository_root` records the workspace
that was actually evaluated. This keeps comparison evidence centralized even
when a task runs in a temporary or registered workspace.

```text
.micro_model_agent/traces/comparison_sessions.jsonl
```

## MCP Prompts

The server also exposes reusable MCP prompts for clients that surface prompt
templates:

- `compare_local_model_on_task`: full shadow-evaluation workflow.
- `collect_real_trace`: run base Qwen with explicit schemas and capture a trace.
- `review_comparison_trace`: review a completed comparison session.
- `smoke_test_micro_agent`: minimal server/tool-loop smoke test.

Do not rely on prompts as the only guidance path. Codex is documented to consume
server instructions and tool descriptions, so the same workflow is summarized in
the MCP server instructions as well.

## Promoted Adapter Smoke Test

After selecting a promoted adapter, run a narrow MCP smoke through the public
tool:

```text
Use micro_agent_run_loop with goal "Read docs/architecture.md and summarize the dependency direction."
Use available_tools ["repo.read"], max_tool_calls 1, max_turns 4, and expose_tool_schemas true.
```

Verify that the response is concise, `tool_calls_made` is `1`, the single tool
step is `repo.read`, and the returned `model` object shows the selected
promotion artifact id from `.micro_model_agent/config.json`. Patch-capable tools
remain dry-run unless `apply_patches` is explicitly true, and `test.run` remains
unavailable unless `allow_test_run` or a test command allowlist is supplied.
