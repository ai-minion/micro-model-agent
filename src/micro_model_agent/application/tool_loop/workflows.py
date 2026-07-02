"""Application use case for model-driven tool loops."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
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

type RunProfile = Literal["quick", "standard", "extended"]
type PromptContext = dict[str, Any] | str
type ToolSchemaBuilder = Callable[[tuple[str, ...]], dict[str, Any]]


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
class ToolLoopBudget:
    """Loop and generation limits for one tool-loop run."""

    max_turns: int = 8
    max_tool_calls: int | None = None
    max_new_tokens: int = 384
    max_tool_result_prompt_chars: int = 12_000
    model_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class PrepareToolLoopRequest:
    """Inputs needed to turn adapter options into an application loop request."""

    goal: str
    available_tools: tuple[str, ...] | None = None
    required_tools: tuple[str, ...] | None = None
    default_tools: tuple[str, ...] = DEFAULT_TOOL_NAMES
    known_tools: tuple[str, ...] = DEFAULT_TOOL_NAMES
    repository_has_git: bool = True
    allow_test_run: bool = False
    run_profile: RunProfile | None = None
    budget: ToolLoopBudget = field(default_factory=ToolLoopBudget)
    context: PromptContext = field(default_factory=dict)
    schema_prompt: bool = True
    require_tool_call: bool = True
    capture_prompts: bool = False
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedToolLoopRun:
    """Prepared application request plus policy decisions made for it."""

    request: RunToolLoopRequest
    budget: ToolLoopBudget
    available_tools: tuple[str, ...]
    required_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunToolLoopResult:
    """Protocol-neutral result for the model-driven tool loop use case."""

    trace_id: UUID
    ok: bool
    response: str
    turns_used: int
    tool_calls_made: int
    trace: WorkflowTrace


def run_profile_settings(profile: RunProfile | None) -> dict[str, int | float | None]:
    """Return loop-budget defaults for common local-model task sizes."""

    if profile is None:
        return {}
    if profile == "quick":
        return {
            "max_turns": 12,
            "max_tool_calls": 8,
            "max_new_tokens": 2048,
            "max_tool_result_prompt_chars": 8000,
            "model_timeout_seconds": 60.0,
        }
    if profile == "standard":
        return {
            "max_turns": 24,
            "max_tool_calls": 24,
            "max_new_tokens": 8192,
            "max_tool_result_prompt_chars": 16000,
            "model_timeout_seconds": 180.0,
        }
    if profile == "extended":
        return {
            "max_turns": 48,
            "max_tool_calls": None,
            "max_new_tokens": 32768,
            "max_tool_result_prompt_chars": 32000,
            "model_timeout_seconds": 600.0,
        }
    raise ValueError(f"unknown run_profile: {profile}")


def apply_run_profile_budget(
    budget: ToolLoopBudget,
    profile: RunProfile | None,
) -> ToolLoopBudget:
    """Apply a named profile over explicit/default budget values."""

    settings = run_profile_settings(profile)
    max_turns = budget.max_turns
    max_tool_calls = budget.max_tool_calls
    max_new_tokens = budget.max_new_tokens
    max_tool_result_prompt_chars = budget.max_tool_result_prompt_chars
    model_timeout_seconds = budget.model_timeout_seconds

    if "max_turns" in settings:
        raw_max_turns = settings["max_turns"]
        assert raw_max_turns is not None
        max_turns = int(raw_max_turns)
    if "max_tool_calls" in settings:
        raw_max_tool_calls = settings["max_tool_calls"]
        max_tool_calls = int(raw_max_tool_calls) if raw_max_tool_calls is not None else None
    if "max_new_tokens" in settings:
        raw_max_new_tokens = settings["max_new_tokens"]
        assert raw_max_new_tokens is not None
        max_new_tokens = int(raw_max_new_tokens)
    if "max_tool_result_prompt_chars" in settings:
        raw_prompt_chars = settings["max_tool_result_prompt_chars"]
        assert raw_prompt_chars is not None
        max_tool_result_prompt_chars = int(raw_prompt_chars)
    if "model_timeout_seconds" in settings:
        raw_timeout = settings["model_timeout_seconds"]
        model_timeout_seconds = float(raw_timeout) if raw_timeout is not None else None

    return ToolLoopBudget(
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_new_tokens=max_new_tokens,
        max_tool_result_prompt_chars=max_tool_result_prompt_chars,
        model_timeout_seconds=model_timeout_seconds,
    )


def normalized_tool_names(
    tool_names: Sequence[str] | None,
    *,
    default_tools: tuple[str, ...],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Expand optional aliases and preserve first-seen order."""

    if not tool_names:
        return default_tools

    normalized: list[str] = []
    alias_map = aliases or {}
    for tool_name in tool_names:
        normalized.extend(alias_map.get(tool_name, (tool_name,)))
    return tuple(dict.fromkeys(normalized))


def select_loop_tool_names(
    tool_names: Sequence[str] | None,
    *,
    default_tools: tuple[str, ...],
    known_tools: Sequence[str],
    repository_has_git: bool,
    allow_test_run: bool,
) -> tuple[str, ...]:
    """Validate and narrow tool names according to application loop policy."""

    known_tool_names = set(known_tools)
    selected: list[str] = []
    for tool_name in normalized_tool_names(tool_names, default_tools=default_tools):
        if tool_name not in known_tool_names:
            raise ValueError(f"unknown built-in tool: {tool_name}")
        if tool_name == "git.diff" and not repository_has_git:
            continue
        if tool_name == "test.run" and not allow_test_run:
            continue
        selected.append(tool_name)
    return tuple(dict.fromkeys(selected))


def prepare_tool_loop_run(
    request: PrepareToolLoopRequest,
    *,
    tool_schema_builder: ToolSchemaBuilder | None = None,
) -> PreparedToolLoopRun:
    """Prepare a transport-neutral tool-loop request from runtime options."""

    budget = apply_run_profile_budget(request.budget, request.run_profile)
    available_tools = select_loop_tool_names(
        request.available_tools,
        default_tools=request.default_tools,
        known_tools=request.known_tools,
        repository_has_git=request.repository_has_git,
        allow_test_run=request.allow_test_run,
    )
    required_tools = normalized_tool_names(
        request.required_tools,
        default_tools=(),
    )
    tool_schemas = (
        tool_schema_builder(available_tools)
        if request.schema_prompt and tool_schema_builder is not None
        else {}
    )
    run_metadata = {
        **request.run_metadata,
        "schema_prompt": request.schema_prompt,
        "capture_prompts": request.capture_prompts,
        "available_tools": list(available_tools),
        "required_tools": list(required_tools),
    }
    if budget.model_timeout_seconds is not None:
        run_metadata["model_timeout_seconds"] = budget.model_timeout_seconds
    if request.run_profile is not None:
        run_metadata["run_profile"] = request.run_profile

    return PreparedToolLoopRun(
        request=RunToolLoopRequest(
            goal=request.goal,
            available_tools=available_tools,
            required_tools=required_tools,
            max_turns=budget.max_turns,
            context=request.context,
            tool_schemas=tool_schemas,
            require_tool_call=request.require_tool_call,
            max_tool_result_prompt_chars=budget.max_tool_result_prompt_chars,
            max_tool_calls=budget.max_tool_calls,
            capture_prompts=request.capture_prompts,
            run_metadata=run_metadata,
            model_timeout_seconds=budget.model_timeout_seconds,
        ),
        budget=budget,
        available_tools=available_tools,
        required_tools=required_tools,
    )


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
