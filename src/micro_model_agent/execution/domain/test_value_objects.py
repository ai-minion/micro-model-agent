"""Tests for execution domain value objects."""

from __future__ import annotations

from uuid import uuid4

from micro_model_agent.execution.domain.value_objects import (
    AgentProfile,
    ModelProfile,
    ToolCall,
    ToolDefinition,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)

# ---------------------------------------------------------------------------
# WorkflowStatus
# ---------------------------------------------------------------------------


def test_workflow_status_values() -> None:
    assert WorkflowStatus.PENDING.value == "pending"
    assert WorkflowStatus.RUNNING.value == "running"
    assert WorkflowStatus.SUCCEEDED.value == "succeeded"
    assert WorkflowStatus.FAILED.value == "failed"
    assert WorkflowStatus.CANCELLED.value == "cancelled"


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


def test_tool_call_auto_id_and_timestamp() -> None:
    call = ToolCall(tool_name="repo_read", arguments={"path": "main.py"})
    assert call.tool_name == "repo_read"
    assert call.arguments == {"path": "main.py"}
    assert call.id is not None
    assert call.created_at is not None


def test_tool_call_is_frozen() -> None:
    call = ToolCall(tool_name="git_diff", arguments={})
    try:
        call.tool_name = "other"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


def test_tool_result_ok() -> None:
    result = ToolResult(
        tool_call_id=uuid4(),
        tool_name="repo_read",
        ok=True,
        output={"content": "def main(): ..."},
    )
    assert result.ok is True
    assert result.error is None


def test_tool_result_error() -> None:
    result = ToolResult(
        tool_call_id=uuid4(),
        tool_name="repo_read",
        ok=False,
        error="file not found",
    )
    assert result.ok is False
    assert result.error == "file not found"
    assert result.output == {}


# ---------------------------------------------------------------------------
# WorkflowStep
# ---------------------------------------------------------------------------


def test_workflow_step_defaults() -> None:
    step = WorkflowStep(name="retrieve_context")
    assert step.name == "retrieve_context"
    assert step.status == WorkflowStatus.PENDING
    assert step.tool_call is None
    assert step.tool_result is None
    assert step.output == {}
    assert step.id is not None


def test_workflow_step_with_tool_call() -> None:
    call = ToolCall(tool_name="repo_search", arguments={"query": "auth"})
    step = WorkflowStep(name="search", tool_call=call, status=WorkflowStatus.RUNNING)
    assert step.tool_call is call
    assert step.status == WorkflowStatus.RUNNING


# ---------------------------------------------------------------------------
# WorkflowTrace
# ---------------------------------------------------------------------------


def test_workflow_trace_defaults() -> None:
    trace = WorkflowTrace(goal="fix the null pointer bug")
    assert trace.goal == "fix the null pointer bug"
    assert trace.status == WorkflowStatus.PENDING
    assert trace.steps == []
    assert trace.final_output == {}
    assert trace.id is not None
    assert trace.created_at is not None
    assert trace.updated_at is not None


def test_workflow_trace_is_frozen() -> None:
    trace = WorkflowTrace(goal="test")
    try:
        trace.goal = "other"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# AgentProfile / ModelProfile / ToolDefinition
# ---------------------------------------------------------------------------


def test_agent_profile() -> None:
    profile = AgentProfile(
        name="coding-agent",
        description="Writes and applies patches",
        default_model="tinyllama",
        allowed_tools=["repo_read", "repo_write_patch"],
    )
    assert profile.name == "coding-agent"
    assert "repo_read" in profile.allowed_tools


def test_model_profile() -> None:
    profile = ModelProfile(
        provider="ollama",
        model_name="tinyllama:latest",
        context_window=8192,
    )
    assert profile.provider == "ollama"
    assert profile.context_window == 8192
    assert profile.capabilities == []


def test_tool_definition() -> None:
    tool = ToolDefinition(
        name="repo_read",
        description="Read a file",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    assert tool.name == "repo_read"
    assert tool.requires_approval is False
