"""Tests for the application-level tool-loop workflow."""

from __future__ import annotations

import asyncio

from micro_model_agent.application.tool_loop import (
    RunToolLoopRequest,
    RunToolLoopWorkflow,
    ToolLoopAgentResult,
    ToolLoopAgentTask,
)
from micro_model_agent.domain.contracts import WorkflowStatus, WorkflowTrace


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
