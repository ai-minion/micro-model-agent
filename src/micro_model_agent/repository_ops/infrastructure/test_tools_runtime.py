"""Tests for built-in tool runtime helpers."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from micro_model_agent.execution.domain.value_objects import (
    ToolCall,
    ToolResult,
)
from micro_model_agent.repository_ops.infrastructure.tools_runtime import (
    PatchPolicyToolExecutor,
    allowed_test_commands,
    builtin_tool_exists,
    builtin_tool_names,
    builtin_tools_response,
    execute_builtin_tool_request,
)


class RecordingExecutor:
    """Small fake executor that records the last tool call it received."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        self.calls.append(tool_call)
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            ok=True,
            output=dict(tool_call.arguments),
        )


def test_allowed_test_commands_builds_pytest_default() -> None:
    commands = allowed_test_commands(None, None, use_default_pytest=True)

    assert list(commands) == ["pytest"]
    assert commands["pytest"].args == ("python3", "-m", "pytest", "-q")


def test_builtin_tool_metadata_helpers_return_json_ready_records() -> None:
    response = builtin_tools_response(default_enabled_tool_names=("repo.read",))

    assert "repo.read" in builtin_tool_names()
    assert builtin_tool_exists("repo.read") is True
    assert builtin_tool_exists("missing.tool") is False
    assert {
        "name": "repo.read",
        "description": "Read one or more repository-relative text files, optionally by line range.",
        "default_enabled_for_mcp": True,
    } in response["tools"]
    assert response["default_agent_tools"] == ["repo.read"]


def test_patch_policy_forces_dry_run_without_apply_permission() -> None:
    executor = RecordingExecutor()
    policy = PatchPolicyToolExecutor(executor, apply_patches=False)
    tool_call = ToolCall(
        id=uuid4(),
        tool_name="repo.write_files",
        arguments={
            "files": [{"path": "README.md", "content": "# Demo\n"}],
            "dry_run": False,
        },
    )

    result = asyncio.run(policy.execute(tool_call))

    assert result.ok is True
    assert executor.calls[0].arguments["dry_run"] is True
    assert executor.calls[0].arguments["require_approval"] is True


def test_patch_policy_disables_approval_when_apply_permission_is_set() -> None:
    executor = RecordingExecutor()
    policy = PatchPolicyToolExecutor(executor, apply_patches=True)
    tool_call = ToolCall(
        id=uuid4(),
        tool_name="repo.write_patch",
        arguments={"patch": "diff --git a/README.md b/README.md\n", "dry_run": False},
    )

    result = asyncio.run(policy.execute(tool_call))

    assert result.ok is True
    assert executor.calls[0].arguments["dry_run"] is False
    assert executor.calls[0].arguments["require_approval"] is False


def test_execute_builtin_tool_request_reports_unknown_tool(tmp_path) -> None:
    result = asyncio.run(
        execute_builtin_tool_request(
            tool_name="missing.tool",
            arguments={},
            repository_root=tmp_path,
        )
    )

    assert result == {"ok": False, "error": "unknown tool: missing.tool"}
