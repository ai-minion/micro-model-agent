"""MCP server CLI command."""

from __future__ import annotations

from pathlib import Path

import typer


def register_mcp_command(app: typer.Typer) -> None:
    """Register the MCP server command."""

    app.command("serve-mcp")(serve_mcp)


def serve_mcp(
    transport: str = typer.Option(
        "stdio",
        help="MCP transport: stdio, sse, or streamable-http.",
    ),
    repository_root: Path = typer.Option(
        Path("."),
        help="Repository root used to decide whether the MCP init tool is needed.",
    ),
) -> None:
    """Serve MicroModelAgent over MCP."""

    from micro_model_agent.interfaces.mcp_server import serve

    serve(transport=transport, repository_root=repository_root)
