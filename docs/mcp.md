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

This launches the server inside WSL so it can use the cached
`Qwen/Qwen2.5-Coder-7B-Instruct` base model and the local PEFT adapter at:

```text
.micro_model_agent/training/runs/qwen-coder-7b-tool-schema-20260613-205520/adapter
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
