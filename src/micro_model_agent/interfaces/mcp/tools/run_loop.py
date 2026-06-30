"""MCP-facing local model run-loop helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.agents.tool_loop_agent import ToolLoopAgent
from micro_model_agent.application.ports import ModelProvider
from micro_model_agent.application.tool_loop import RunToolLoopRequest, RunToolLoopWorkflow
from micro_model_agent.infrastructure.composition import (
    RuntimeModelOptions,
    base_model_from_adapter,
    build_builtin_tool_executor,
    build_model_provider,
    resolve_model_options,
    string_config_value,
)
from micro_model_agent.infrastructure.composition import (
    tool_prompt_schemas as build_tool_prompt_schemas,
)
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.interfaces.mcp.compat import DEFAULT_7B_ADAPTER_PATH, RunProfile
from micro_model_agent.interfaces.mcp.policy.patch_policy import PatchPolicyToolExecutor
from micro_model_agent.interfaces.mcp.policy.tool_names import (
    allowed_test_commands,
    allowed_tool_names,
    required_tool_names,
    run_profile_settings,
)
from micro_model_agent.interfaces.mcp.traces import append_comparison_event
from micro_model_agent.interfaces.mcp.traces import (
    workflow_trace_store as mcp_workflow_trace_store,
)

__all__ = [
    "base_model_from_adapter",
    "model_provider_for_loop",
    "resolve_model_settings",
    "run_agent_loop",
    "string_config_value",
    "tool_prompt_schemas",
]


async def run_agent_loop(
    *,
    goal: str,
    repository_root: str = ".",
    context: str = "",
    adapter_path: str | None = None,
    base_model: str | None = None,
    use_adapter: bool = True,
    available_tools: list[str] | None = None,
    required_tools: list[str] | None = None,
    max_turns: int = 4,
    max_tool_calls: int | None = 1,
    max_new_tokens: int = 350,
    max_tool_result_prompt_chars: int = 2500,
    model_timeout_seconds: float | None = None,
    run_profile: RunProfile | None = None,
    schema_prompt: bool = True,
    capture_prompts: bool = False,
    apply_patches: bool = False,
    allow_test_run: bool = False,
    test_command_name: str | None = None,
    test_command_args: list[str] | None = None,
    scripted_responses: list[str] | None = None,
    comparison_session_id: str | None = None,
    comparison_repository_root: str | None = None,
    offline: bool = True,
) -> dict[str, Any]:
    """Run the model-driven tool loop and return a JSON-serializable result."""

    profile_settings = run_profile_settings(run_profile)
    max_turns = profile_settings.get("max_turns", max_turns)
    max_tool_calls = profile_settings.get("max_tool_calls", max_tool_calls)
    max_new_tokens = profile_settings.get("max_new_tokens", max_new_tokens)
    max_tool_result_prompt_chars = profile_settings.get(
        "max_tool_result_prompt_chars",
        max_tool_result_prompt_chars,
    )
    model_timeout_seconds = profile_settings.get(
        "model_timeout_seconds",
        model_timeout_seconds,
    )

    repository = Path(repository_root)
    test_commands = allowed_test_commands(
        test_command_name,
        test_command_args,
        use_default_pytest=allow_test_run and not test_command_name,
    )
    # Tool availability is narrowed before the model sees it.
    loop_tool_names = allowed_tool_names(
        available_tools=available_tools,
        repository_root=repository,
        apply_patches=apply_patches,
        allow_test_run=bool(test_commands),
    )
    loop_required_tool_names = required_tool_names(required_tools)
    model_settings = resolve_model_settings(
        repository_root=repository,
        adapter_path=adapter_path,
        base_model=base_model,
        use_adapter=use_adapter,
        allow_missing_base_model=bool(scripted_responses),
    )
    model_provider = model_provider_for_loop(
        adapter_path=model_settings["adapter_path"],
        base_model=model_settings["base_model"],
        max_new_tokens=max_new_tokens,
        scripted_responses=scripted_responses,
        offline=offline,
    )
    # Wrap the normal executor so MCP-specific patch policy is enforced in one place.
    executor = PatchPolicyToolExecutor(
        build_builtin_tool_executor(repository, test_commands),
        apply_patches=apply_patches,
    )
    trace_root = Path(comparison_repository_root or repository)
    trace_store = mcp_workflow_trace_store(trace_root)
    agent = ToolLoopAgent(
        model_provider=model_provider,
        tool_executor=executor,
        trace_store=trace_store,
    )
    workflow = RunToolLoopWorkflow(agent)

    result = await workflow.run(
        RunToolLoopRequest(
            goal=goal,
            available_tools=loop_tool_names,
            required_tools=loop_required_tool_names,
            max_turns=max_turns,
            context=context,
            tool_schemas=tool_prompt_schemas(
                loop_tool_names,
                allowed_test_commands=test_commands,
            )
            if schema_prompt
            else {},
            max_tool_calls=max_tool_calls,
            max_tool_result_prompt_chars=max_tool_result_prompt_chars,
            model_timeout_seconds=model_timeout_seconds,
            capture_prompts=capture_prompts,
            run_metadata={
                "interface": "mcp",
                "schema_prompt": schema_prompt,
                "capture_prompts": capture_prompts,
                "apply_patches": apply_patches,
                "available_tools": list(loop_tool_names),
                "required_tools": list(loop_required_tool_names),
                "allowed_test_commands": list(test_commands),
                "model": model_settings,
                "model_timeout_seconds": model_timeout_seconds,
                "run_profile": run_profile,
            },
        )
    )
    if comparison_session_id:
        await append_comparison_event(
            repository_root=Path(comparison_repository_root or repository),
            session_id=comparison_session_id,
            event_type="local_model_run",
            actor="micro_model_agent",
            payload={
                "trace_id": str(result.trace_id),
                "ok": result.ok,
                "response": result.response,
                "turns_used": result.turns_used,
                "tool_calls_made": result.tool_calls_made,
                "model": model_settings,
            },
        )
    loop_budget: dict[str, int | float | str | None] = {
        "max_turns": max_turns,
        "max_tool_calls": max_tool_calls,
        "max_new_tokens": max_new_tokens,
        "max_tool_result_prompt_chars": max_tool_result_prompt_chars,
    }
    if model_timeout_seconds is not None:
        loop_budget["model_timeout_seconds"] = model_timeout_seconds
    if run_profile is not None:
        loop_budget["run_profile"] = run_profile
    # Return a compact summary rather than the full trace. The full trace can be
    # loaded by debug tooling when exposed.
    return {
        "ok": result.ok,
        "response": result.response,
        "trace_id": str(result.trace_id),
        "turns_used": result.turns_used,
        "tool_calls_made": result.tool_calls_made,
        "loop_budget": loop_budget,
        "model": model_settings,
        "steps": [
            {
                "name": step.name,
                "status": step.status.value,
                "tool_name": step.tool_call.tool_name if step.tool_call else None,
                "tool_ok": step.tool_result.ok if step.tool_result else None,
                "tool_error": step.tool_result.error if step.tool_result else None,
            }
            for step in result.trace.steps
        ],
    }


def model_provider_for_loop(
    *,
    adapter_path: str | None,
    base_model: str | None,
    max_new_tokens: int,
    scripted_responses: list[str] | None,
    offline: bool,
) -> ModelProvider:
    """Return the model provider used by the MCP run loop."""

    provider = build_model_provider(
        options=RuntimeModelOptions(
            model=None,
            base_model=base_model,
            adapter_path=Path(adapter_path) if adapter_path else None,
        ),
        max_new_tokens=max_new_tokens,
        scripted_responses=scripted_responses,
        offline=offline,
    )
    assert provider is not None
    return provider


def resolve_model_settings(
    *,
    repository_root: str | Path,
    adapter_path: str | None,
    base_model: str | None,
    use_adapter: bool,
    allow_missing_base_model: bool = False,
) -> dict[str, str | None]:
    """Resolve MCP model settings from explicit args, env, selected config, defaults."""

    options = resolve_model_options(
        repository_root=repository_root,
        base_model=base_model,
        adapter_path=adapter_path,
        use_adapter=use_adapter,
        default_adapter_path=DEFAULT_7B_ADAPTER_PATH,
    )
    resolved_base_model = options.base_model
    if resolved_base_model is None and not allow_missing_base_model:
        if options.adapter_path is None:
            raise ValueError("--base-model is required when use_adapter is false")
        resolved_base_model = base_model_from_adapter(options.adapter_path)

    return {
        "base_model": resolved_base_model,
        "adapter_path": str(options.adapter_path) if options.adapter_path else None,
        "selected_promotion_artifact_id": options.selected_promotion_artifact_id,
    }


def tool_prompt_schemas(
    allowed_tool_names: tuple[str, ...],
    *,
    allowed_test_commands: dict[str, AllowedTestCommand],
) -> dict[str, Any]:
    """Compatibility wrapper for shared tool schema enrichment."""

    return build_tool_prompt_schemas(
        allowed_tool_names,
        allowed_test_commands=allowed_test_commands,
    )
