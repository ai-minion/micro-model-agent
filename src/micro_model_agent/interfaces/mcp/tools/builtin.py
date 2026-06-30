"""MCP debug bridge for built-in repository tools."""

from __future__ import annotations

from typing import Any

from micro_model_agent.domain.contracts import ToolCall
from micro_model_agent.infrastructure.composition import build_builtin_tool_executor
from micro_model_agent.infrastructure.tools.catalog import BUILTIN_TOOL_SPECS
from micro_model_agent.interfaces.mcp.compat import DEFAULT_MCP_AVAILABLE_TOOLS
from micro_model_agent.interfaces.mcp.policy.patch_policy import PatchPolicyToolExecutor
from micro_model_agent.interfaces.mcp.policy.tool_names import allowed_test_commands


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

    if tool_name not in BUILTIN_TOOL_SPECS:
        return {"ok": False, "error": f"unknown tool: {tool_name}"}

    # Direct tool execution uses the same validation and patch policy as the agent loop.
    executor = PatchPolicyToolExecutor(
        build_builtin_tool_executor(
            repository_root,
            allowed_test_commands(test_command_name, test_command_args),
        ),
        apply_patches=apply_patches,
    )
    result = await executor.execute(ToolCall(tool_name=tool_name, arguments=arguments))
    return {
        "ok": result.ok,
        "tool_name": result.tool_name,
        "output": result.output,
        "error": result.error,
    }


def list_builtin_tools() -> dict[str, Any]:
    """Return built-in tool names and descriptions."""

    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "default_enabled_for_mcp": spec.name in DEFAULT_MCP_AVAILABLE_TOOLS,
            }
            for spec in BUILTIN_TOOL_SPECS.values()
        ],
        "default_agent_tools": list(DEFAULT_MCP_AVAILABLE_TOOLS),
    }
