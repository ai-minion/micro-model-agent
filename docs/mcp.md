# MCP Server

micro-model-agent can run as an MCP stdio server and expose the local tool loop to
Copilot or another MCP client.

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

This launches the server inside WSL so it can use direct Transformers adapter
inference. Model resolution uses this order:

1. `micro_agent_run_loop` `base_model` / `adapter_path` arguments.
2. `MICRO_MODEL_AGENT_BASE_MODEL` / `MICRO_MODEL_AGENT_ADAPTER_PATH`.
3. The selected promoted adapter in `.micro_model_agent/config.json`.
4. The legacy default local PEFT adapter path:

```text
.micro_model_agent/training/runs/qwen-coder-7b-tool-schema-20260613-205520/adapter
```

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

Default agent tools are read-oriented:

```text
repo.search, repo.read, repo.semantic_search, git.diff
```

`repo.write_patch` is not enabled by default. When it is enabled through
`available_tools`, patch application remains dry-run unless `apply_patches` is
explicitly true.

## Useful First Prompt

After starting the MCP server in Copilot Agent mode, try:

```text
Use micro_agent_run_loop to read README.md and summarize the project status in one sentence.
Use available_tools ["repo.read"], max_tool_calls 1, max_turns 4, and schema_prompt false.
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
schema_prompt true, capture_prompts true, apply_patches true when you want real
edits, and include the tools needed for the task.
```

This stores workflow traces under:

```text
.micro_model_agent/traces/workflows.jsonl
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

## Promoted Adapter Smoke Test

After selecting a promoted adapter, run a narrow MCP smoke through the public
tool:

```text
Use micro_agent_run_loop with goal "Read docs/architecture.md and summarize the dependency direction."
Use available_tools ["repo.read"], max_tool_calls 1, max_turns 4, and schema_prompt true.
```

Verify that the response is concise, `tool_calls_made` is `1`, the single tool
step is `repo.read`, and the returned `model` object shows the selected
promotion artifact id from `.micro_model_agent/config.json`. Patch-capable tools
remain dry-run unless `apply_patches` is explicitly true, and `test.run` remains
unavailable unless `allow_test_run` or a test command allowlist is supplied.
