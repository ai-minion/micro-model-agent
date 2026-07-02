"""Runtime helpers for built-in tool execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from micro_model_agent.application.ports.contracts import ToolExecutor
from micro_model_agent.domain.contracts import ToolCall, ToolResult
from micro_model_agent.infrastructure.tools.catalog import BUILTIN_TOOL_SPECS
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.infrastructure.tools.executor import BuiltinToolExecutor

__all__ = [
    "PatchPolicyToolExecutor",
    "allowed_test_commands",
    "build_builtin_tool_executor",
    "builtin_tool_exists",
    "builtin_tool_names",
    "builtin_tool_summaries",
    "builtin_tools_response",
    "execute_builtin_tool_request",
]


class PatchPolicyToolExecutor:
    """Tool executor wrapper that keeps repository writes dry-run unless enabled."""

    def __init__(self, wrapped: ToolExecutor, *, apply_patches: bool) -> None:
        self.wrapped = wrapped
        self.apply_patches = apply_patches

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.tool_name not in {"repo.write_patch", "repo.write_files"}:
            return await self.wrapped.execute(tool_call)

        if self.apply_patches:
            arguments = {
                **tool_call.arguments,
                "dry_run": tool_call.arguments.get("dry_run", False),
                "require_approval": False,
            }
            return await self.wrapped.execute(replace(tool_call, arguments=arguments))

        arguments = {
            **tool_call.arguments,
            "dry_run": True,
            "require_approval": True,
        }
        return await self.wrapped.execute(replace(tool_call, arguments=arguments))


def allowed_test_commands(
    test_command_name: str | None,
    test_command_args: Sequence[str] | None,
    *,
    use_default_pytest: bool = False,
) -> dict[str, AllowedTestCommand]:
    """Build the allowlist consumed by the test.run tool."""

    if use_default_pytest:
        return {"pytest": AllowedTestCommand(("python3", "-m", "pytest", "-q"))}
    if not test_command_name or not test_command_args:
        return {}
    return {test_command_name: AllowedTestCommand(tuple(test_command_args))}


def build_builtin_tool_executor(
    repository_root: str | Path,
    allowed_commands: Mapping[str, AllowedTestCommand | Sequence[str]],
) -> BuiltinToolExecutor:
    """Build the standard repository tool executor."""

    return BuiltinToolExecutor(repository_root, allowed_commands)


def builtin_tool_names() -> tuple[str, ...]:
    """Return all known built-in tool names."""

    return tuple(BUILTIN_TOOL_SPECS)


def builtin_tool_exists(tool_name: str) -> bool:
    """Return whether a built-in tool is registered."""

    return tool_name in BUILTIN_TOOL_SPECS


def builtin_tool_summaries(
    *,
    default_enabled_tool_names: Sequence[str] = (),
) -> list[dict[str, str | bool]]:
    """Return JSON-ready built-in tool metadata for interface listings."""

    default_enabled = set(default_enabled_tool_names)
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "default_enabled_for_mcp": spec.name in default_enabled,
        }
        for spec in BUILTIN_TOOL_SPECS.values()
    ]


def builtin_tools_response(
    *,
    default_enabled_tool_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the built-in tool listing response used by debug interfaces."""

    return {
        "tools": builtin_tool_summaries(
            default_enabled_tool_names=default_enabled_tool_names,
        ),
        "default_agent_tools": list(default_enabled_tool_names),
    }


async def execute_builtin_tool_request(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    repository_root: str | Path = ".",
    apply_patches: bool = False,
    test_command_name: str | None = None,
    test_command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute one built-in tool and return a JSON-ready result."""

    if not builtin_tool_exists(tool_name):
        return {"ok": False, "error": f"unknown tool: {tool_name}"}

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
