"""FastMCP server construction for MicroModelAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from mcp.server import NotificationOptions
from mcp.server.fastmcp import FastMCP

from micro_model_agent.interfaces.mcp.compat import McpTransport
from micro_model_agent.interfaces.mcp.prompts.registry import register_workflow_prompts
from micro_model_agent.interfaces.mcp.tools.registry import register_mcp_tools
from micro_model_agent.interfaces.mcp.tools.run_loop import run_agent_loop


def create_mcp_server(
    *,
    repository_root: str | Path = ".",
    expose_debug_tools: bool | None = None,
    expose_init_tool: bool | None = None,
) -> FastMCP:
    """Create the MicroModelAgent MCP server."""

    # FastMCP handles protocol details; this function only registers Python callables.
    server = FastMCP(
        "MicroModelAgent",
        instructions=(
            "Use MicroModelAgent to collect local-model coding traces. Preferred comparison "
            "workflow: call micro_agent_init_workspace when a chat needs its own directory, "
            "then call micro_agent_start_trace, call micro_agent_run_loop with workspace_id, "
            "comparison_session_id, base_model='Qwen/Qwen2.5-Coder-7B-Instruct', "
            "use_adapter=false, schema_prompt=true, capture_prompts=true, and a run_profile "
            "that matches the task size. One run loop has a limited turn budget; split large "
            "tasks into scaffold, repair, and test passes rather than asking one loop to do "
            "everything. Do the real work yourself, call micro_agent_stop_trace, and finish "
            "with micro_agent_review_trace."
        ),
    )
    enable_tool_list_changed_capability(server)
    default_repository_root = str(Path(repository_root))

    register_workflow_prompts(server)
    register_mcp_tools(
        server,
        default_repository_root=default_repository_root,
        expose_debug_tools=expose_debug_tools,
        expose_init_tool=expose_init_tool,
        run_agent_loop_handler=run_agent_loop,
    )
    return server


def serve(transport: str = "stdio", *, repository_root: str | Path = ".") -> None:
    """Run the MCP server."""

    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(f"unknown MCP transport: {transport}")
    create_mcp_server(repository_root=repository_root).run(transport=cast(McpTransport, transport))


def enable_tool_list_changed_capability(server: FastMCP) -> None:
    """Advertise dynamic tool-list updates for FastMCP versions without a public knob."""

    # Some FastMCP versions do not expose this capability directly, so the code
    # wraps the low-level initialization function.
    lowlevel_server = cast(Any, server)._mcp_server
    original_create_initialization_options = lowlevel_server.create_initialization_options

    def create_initialization_options(
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> Any:
        if notification_options is None:
            notification_options = NotificationOptions(tools_changed=True)
        else:
            notification_options.tools_changed = True
        return original_create_initialization_options(
            notification_options,
            experimental_capabilities,
        )

    lowlevel_server.create_initialization_options = create_initialization_options
