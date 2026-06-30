"""Exposure decisions for optional MCP tools."""

from __future__ import annotations

import os
from pathlib import Path

from micro_model_agent.infrastructure.repository_metadata import is_repository_initialized
from micro_model_agent.interfaces.mcp.compat import MCP_DEBUG_TOOLS_ENV, MCP_EXPOSE_INIT_ENV


def should_expose_debug_tools(explicit: bool | None) -> bool:
    """Decide whether debug tools should be registered."""

    if explicit is not None:
        return explicit
    return truthy_env(MCP_DEBUG_TOOLS_ENV)


def should_expose_init_tool(repository_root: str | Path, explicit: bool | None) -> bool:
    """Show the init tool only when requested or when the repository is uninitialized."""

    if explicit is not None:
        return explicit
    if truthy_env(MCP_EXPOSE_INIT_ENV):
        return True
    return not is_repository_initialized(repository_root)


def truthy_env(name: str) -> bool:
    """Interpret common truthy environment variable values."""

    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}
