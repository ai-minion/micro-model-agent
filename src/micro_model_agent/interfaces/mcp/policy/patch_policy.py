"""Patch-write safety policy for MCP tool execution."""

from __future__ import annotations

from dataclasses import replace

from micro_model_agent.application.ports import ToolExecutor
from micro_model_agent.domain.contracts import ToolCall, ToolResult


class PatchPolicyToolExecutor:
    """Tool executor wrapper that keeps patch writes dry-run unless explicitly enabled."""

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
            approved_call = replace(tool_call, arguments=arguments)
            return await self.wrapped.execute(approved_call)

        # MCP callers can preview writes by default, but real application is
        # forced back to dry-run unless apply_patches=True was requested.
        arguments = {
            **tool_call.arguments,
            "dry_run": True,
            "require_approval": True,
        }
        safe_call = replace(tool_call, arguments=arguments)
        return await self.wrapped.execute(safe_call)
