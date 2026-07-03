"""Tests for the application-level tool-loop workflow."""

from __future__ import annotations

import asyncio

from micro_model_agent.execution.application.tool_loop import (
    PrepareToolLoopRequest,
    RunToolLoopRequest,
    RunToolLoopWorkflow,
    ToolLoopAgentResult,
    ToolLoopAgentTask,
    ToolLoopBudget,
    prepare_tool_loop_run,
)
from micro_model_agent.execution.domain.value_objects import (
    WorkflowStatus,
    WorkflowTrace,
)


class FakeToolLoopRunner:
    """Application-only fake runner that records the task it receives."""

    def __init__(self) -> None:
        self.tasks: list[ToolLoopAgentTask] = []

    async def run(self, task: ToolLoopAgentTask) -> ToolLoopAgentResult:
        self.tasks.append(task)
        trace = WorkflowTrace(
            goal=task.goal,
            status=WorkflowStatus.SUCCEEDED,
            final_output={"response": "done"},
        )
        return ToolLoopAgentResult(
            trace_id=trace.id,
            ok=True,
            response="done",
            turns_used=1,
            tool_calls_made=0,
            trace=trace,
        )


def test_run_tool_loop_workflow_maps_request_to_runner_task() -> None:
    runner = FakeToolLoopRunner()
    workflow = RunToolLoopWorkflow(runner)

    result = asyncio.run(
        workflow.run(
            RunToolLoopRequest(
                goal="Read the README.",
                available_tools=("repo.read",),
                required_tools=("repo.read",),
                max_turns=3,
                context="Use concise output.",
                tool_schemas={"repo.read": {"arguments_schema": {"type": "object"}}},
                require_tool_call=False,
                max_tool_result_prompt_chars=512,
                max_tool_calls=1,
                capture_prompts=True,
                run_metadata={"interface": "test"},
                model_timeout_seconds=2.5,
            )
        )
    )

    task = runner.tasks[0]
    assert task.goal == "Read the README."
    assert task.available_tools == ("repo.read",)
    assert task.required_tools == ("repo.read",)
    assert task.max_turns == 3
    assert task.context == "Use concise output."
    assert task.tool_schemas == {"repo.read": {"arguments_schema": {"type": "object"}}}
    assert task.require_tool_call is False
    assert task.max_tool_result_prompt_chars == 512
    assert task.max_tool_calls == 1
    assert task.capture_prompts is True
    assert task.run_metadata == {"interface": "test"}
    assert task.model_timeout_seconds == 2.5
    assert result.ok is True
    assert result.response == "done"
    assert result.trace.status is WorkflowStatus.SUCCEEDED


def test_prepare_tool_loop_run_applies_profile_and_selects_tools() -> None:
    prepared = prepare_tool_loop_run(
        PrepareToolLoopRequest(
            goal="Inspect repo.",
            available_tools=("repo.read", "git.diff", "test.run"),
            required_tools=("repo.read",),
            known_tools=("repo.read", "git.diff", "test.run"),
            repository_has_git=False,
            allow_test_run=True,
            run_profile="quick",
            budget=ToolLoopBudget(
                max_turns=2,
                max_tool_calls=1,
                max_new_tokens=64,
                max_tool_result_prompt_chars=256,
            ),
            schema_prompt=True,
            capture_prompts=True,
            run_metadata={"interface": "test"},
        ),
        tool_schema_builder=lambda names: {name: {"schema": name} for name in names},
    )

    assert prepared.available_tools == ("repo.read", "test.run")
    assert prepared.required_tools == ("repo.read",)
    assert prepared.budget.max_turns == 12
    assert prepared.budget.max_tool_calls == 8
    assert prepared.budget.max_new_tokens == 2048
    assert prepared.budget.max_tool_result_prompt_chars == 8000
    assert prepared.budget.model_timeout_seconds == 60.0
    assert prepared.request.tool_schemas == {
        "repo.read": {"schema": "repo.read"},
        "test.run": {"schema": "test.run"},
    }
    assert prepared.request.run_metadata == {
        "interface": "test",
        "schema_prompt": True,
        "capture_prompts": True,
        "available_tools": ["repo.read", "test.run"],
        "required_tools": ["repo.read"],
        "model_timeout_seconds": 60.0,
        "run_profile": "quick",
    }


def test_prepare_tool_loop_run_rejects_unknown_tools() -> None:
    try:
        prepare_tool_loop_run(
            PrepareToolLoopRequest(
                goal="Inspect repo.",
                available_tools=("repo.read", "unknown.tool"),
                known_tools=("repo.read",),
            )
        )
    except ValueError as exc:
        assert str(exc) == "unknown built-in tool: unknown.tool"
    else:
        raise AssertionError("expected unknown tool to be rejected")
