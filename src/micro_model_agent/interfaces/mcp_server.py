"""MCP server entrypoint for MicroModelAgent.

MCP exposes MicroModelAgent capabilities to external clients as tools. This file
builds the server, registers tools, and keeps patch application conservative by
default.
"""

from __future__ import annotations

import argparse
import os

from micro_model_agent.interfaces.mcp.compat import MCP_REPOSITORY_ROOT_ENV
from micro_model_agent.interfaces.mcp.server import create_mcp_server, serve
from micro_model_agent.interfaces.mcp.tools.builtin import call_builtin_tool, list_builtin_tools
from micro_model_agent.interfaces.mcp.tools.run_loop import run_agent_loop
from micro_model_agent.interfaces.mcp.traces import (
    read_trace,
    record_comparison_event,
    review_comparison_trace,
    start_comparison_trace,
    stop_comparison_trace,
)
from micro_model_agent.interfaces.mcp.workspace import (
    init_repository,
    init_workspace,
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


if __name__ == "__main__":
    main()
