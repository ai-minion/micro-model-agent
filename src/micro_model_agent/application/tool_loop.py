"""Application use case for model-driven tool loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from micro_model_agent.domain.contracts import WorkflowTrace

DEFAULT_TOOL_NAMES: tuple[str, ...] = (
    "repo.search",
    "repo.read",
    "repo.semantic_search",
    "repo.write_patch",
    "repo.write_files",
    "test.run",
    "git.diff",
)

type PromptContext = dict[str, Any] | str


@dataclass(frozen=True, slots=True)
class ToolLoopAgentTask:
    """Input for a model-driven tool-call loop."""

    goal: str
    available_tools: tuple[str, ...] = DEFAULT_TOOL_NAMES
    # max_turns prevents an unproductive model/tool conversation from running forever.
    max_turns: int = 8
    context: PromptContext = field(default_factory=dict)
    tool_schemas: dict[str, Any] = field(default_factory=dict)
    require_tool_call: bool = True
    max_tool_result_prompt_chars: int = 12_000
    # max_tool_calls can force the model to stop gathering data and answer from
    # the tool results it already has.
    max_tool_calls: int | None = None
    required_tools: tuple[str, ...] = ()
    capture_prompts: bool = False
    run_metadata: dict[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ToolLoopAgentResult:
    """Structured result returned by the tool-loop agent."""

    trace_id: UUID
    ok: bool
    response: str
    turns_used: int
    tool_calls_made: int
    trace: WorkflowTrace


class ToolLoopRunner(Protocol):
    """Application port for a model-driven tool loop implementation."""

    async def run(self, task: ToolLoopAgentTask) -> ToolLoopAgentResult:
        """Run a tool-loop task and return its structured result."""


@dataclass(frozen=True, slots=True)
class RunToolLoopRequest:
    """Protocol-neutral request for the model-driven tool loop use case."""

    goal: str
    available_tools: tuple[str, ...] = DEFAULT_TOOL_NAMES
    required_tools: tuple[str, ...] = ()
    max_turns: int = 8
    context: PromptContext = field(default_factory=dict)
    tool_schemas: dict[str, Any] = field(default_factory=dict)
    require_tool_call: bool = True
    max_tool_result_prompt_chars: int = 12_000
    max_tool_calls: int | None = None
    capture_prompts: bool = False
    run_metadata: dict[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class RunToolLoopResult:
    """Protocol-neutral result for the model-driven tool loop use case."""

    trace_id: UUID
    ok: bool
    response: str
    turns_used: int
    tool_calls_made: int
    trace: WorkflowTrace


class RunToolLoopWorkflow:
    """Run a model-driven tool loop behind an application-level request contract."""

    def __init__(self, runner: ToolLoopRunner) -> None:
        self.runner = runner

    async def run(self, request: RunToolLoopRequest) -> RunToolLoopResult:
        """Run the tool loop and return a stable application result."""

        result = await self.runner.run(
            ToolLoopAgentTask(
                goal=request.goal,
                available_tools=request.available_tools,
                required_tools=request.required_tools,
                max_turns=request.max_turns,
                context=request.context,
                tool_schemas=request.tool_schemas,
                require_tool_call=request.require_tool_call,
                max_tool_result_prompt_chars=request.max_tool_result_prompt_chars,
                max_tool_calls=request.max_tool_calls,
                capture_prompts=request.capture_prompts,
                run_metadata=request.run_metadata,
                model_timeout_seconds=request.model_timeout_seconds,
            )
        )
        return RunToolLoopResult(
            trace_id=result.trace_id,
            ok=result.ok,
            response=result.response,
            turns_used=result.turns_used,
            tool_calls_made=result.tool_calls_made,
            trace=result.trace,
        )
