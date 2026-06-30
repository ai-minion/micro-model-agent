"""MCP server entrypoint for MicroModelAgent.

MCP exposes MicroModelAgent capabilities to external clients as tools. This file
builds the server, registers tools, and keeps patch application conservative by
default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from micro_model_agent.interfaces.mcp.compat import MCP_REPOSITORY_ROOT_ENV
from micro_model_agent.interfaces.mcp.policy.tool_names import (
    allowed_test_commands as _allowed_test_commands,
)
from micro_model_agent.interfaces.mcp.policy.tool_names import (
    allowed_tool_names as _allowed_tool_names,
)
from micro_model_agent.interfaces.mcp.policy.tool_names import (
    required_tool_names as _required_tool_names,
)
from micro_model_agent.interfaces.mcp.policy.tool_names import (
    run_profile_settings as _run_profile_settings,
)
from micro_model_agent.interfaces.mcp.server import create_mcp_server, serve
from micro_model_agent.interfaces.mcp.tools.builtin import call_builtin_tool, list_builtin_tools
from micro_model_agent.interfaces.mcp.tools.run_loop import (
    base_model_from_adapter as _base_model_from_adapter,
)
from micro_model_agent.interfaces.mcp.tools.run_loop import (
    model_provider_for_loop as _model_provider,
)
from micro_model_agent.interfaces.mcp.tools.run_loop import (
    resolve_model_settings as _resolve_model_settings,
)
from micro_model_agent.interfaces.mcp.tools.run_loop import run_agent_loop
from micro_model_agent.interfaces.mcp.tools.run_loop import (
    string_config_value as _string_config_value,
)
from micro_model_agent.interfaces.mcp.tools.run_loop import (
    tool_prompt_schemas as _tool_prompt_schemas,
)
from micro_model_agent.interfaces.mcp.traces import (
    read_trace,
    record_comparison_event,
    review_comparison_trace,
    start_comparison_trace,
    stop_comparison_trace,
)
from micro_model_agent.interfaces.mcp.traces import workflow_trace_store as _workflow_trace_store
from micro_model_agent.interfaces.mcp.workspace import (
    init_repository,
    init_workspace,
    path_from_user_input,
)

__all__ = [
    "call_builtin_tool",
    "create_mcp_server",
    "init_repository",
    "init_workspace",
    "list_builtin_tools",
    "main",
    "read_trace",
    "record_comparison_event",
    "review_comparison_trace",
    "run_agent_loop",
    "serve",
    "start_comparison_trace",
    "stop_comparison_trace",
    "_allowed_test_commands",
    "_allowed_tool_names",
    "_base_model_from_adapter",
    "_model_provider",
    "_path_from_user_input",
    "_required_tool_names",
    "_resolve_model_settings",
    "_run_profile_settings",
    "_string_config_value",
    "_tool_prompt_schemas",
    "_workflow_trace_store",
]


def main() -> None:
    """Console entrypoint for `python -m micro_model_agent.interfaces.mcp_server`."""

    parser = argparse.ArgumentParser(description="Serve MicroModelAgent over MCP.")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=("stdio", "sse", "streamable-http"),
        help="MCP transport.",
    )
    parser.add_argument(
        "--repository-root",
        default=os.environ.get(MCP_REPOSITORY_ROOT_ENV, "."),
        help=(
            "Default repository root for MCP tools. Also configurable with "
            f"{MCP_REPOSITORY_ROOT_ENV}."
        ),
    )
    args = parser.parse_args()
    serve(transport=args.transport, repository_root=args.repository_root)


def _path_from_user_input(value: str, *, wsl_mount_root: Path = Path("/mnt")) -> Path:
    """Compatibility wrapper for the moved MCP workspace path helper."""

    return path_from_user_input(value, wsl_mount_root=wsl_mount_root)


if __name__ == "__main__":
    main()
