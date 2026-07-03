"""MCP debug bridge for built-in repository tools."""

from __future__ import annotations

from typing import Any

from micro_model_agent.interfaces.composition import (
    builtin_tools_response,
    execute_builtin_tool_request,
)
from micro_model_agent.interfaces.mcp.compat import DEFAULT_MCP_AVAILABLE_TOOLS


async def call_builtin_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    repository_root: str = ".",
    apply_patches: bool = False,
    test_command_name: str | None = None,
    test_command_args: list[str] | None = None,
) -> dict[str, Any]:
    """Execute one built-in tool through MCP."""

    return await execute_builtin_tool_request(
        tool_name=tool_name,
        arguments=arguments,
        repository_root=repository_root,
        apply_patches=apply_patches,
        test_command_name=test_command_name,
        test_command_args=test_command_args,
    )


def list_builtin_tools() -> dict[str, Any]:
    """Return built-in tool names and descriptions."""

    return builtin_tools_response(default_enabled_tool_names=DEFAULT_MCP_AVAILABLE_TOOLS)
