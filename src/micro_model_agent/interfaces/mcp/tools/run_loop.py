"""MCP-facing local model run-loop helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.application.ports import ModelProvider
from micro_model_agent.application.tool_loop import normalized_tool_names
from micro_model_agent.infrastructure.composition import (
    ConfiguredToolLoopResult,
    RuntimeModelOptions,
    base_model_from_adapter,
    build_model_provider,
    resolve_model_options,
    run_configured_tool_loop,
    string_config_value,
)
from micro_model_agent.infrastructure.composition import (
    tool_prompt_schemas as build_tool_prompt_schemas,
)
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.interfaces.mcp.compat import (
    COMPAT_REQUIRED_TOOL_ALIASES,
    COMPAT_TOOL_ALIASES,
    DEFAULT_7B_ADAPTER_PATH,
    DEFAULT_MCP_AVAILABLE_TOOLS,
    RunProfile,
)
from micro_model_agent.interfaces.mcp.policy.patch_policy import PatchPolicyToolExecutor
from micro_model_agent.interfaces.mcp.policy.tool_names import (
    allowed_test_commands,
)
from micro_model_agent.interfaces.mcp.traces import append_comparison_event

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

    repository = Path(repository_root)
    test_commands = allowed_test_commands(
        test_command_name,
        test_command_args,
        use_default_pytest=allow_test_run and not test_command_name,
    )
    loop_tool_names = normalized_tool_names(
        available_tools,
        default_tools=DEFAULT_MCP_AVAILABLE_TOOLS,
        aliases=COMPAT_TOOL_ALIASES,
    )
    loop_required_tool_names = normalized_tool_names(
        required_tools,
        default_tools=(),
        aliases=COMPAT_REQUIRED_TOOL_ALIASES,
    )
    model_settings = resolve_model_settings(
        repository_root=repository,
        adapter_path=adapter_path,
        base_model=base_model,
        use_adapter=use_adapter,
        allow_missing_base_model=bool(scripted_responses),
    )
    configured = await run_configured_tool_loop(
        goal=goal,
        repository_root=repository,
        model_options=RuntimeModelOptions(
            model=None,
            base_model=model_settings["base_model"],
            adapter_path=Path(model_settings["adapter_path"])
            if model_settings["adapter_path"]
            else None,
            selected_promotion_artifact_id=model_settings["selected_promotion_artifact_id"],
        ),
        max_new_tokens=max_new_tokens,
        scripted_responses=scripted_responses,
        offline=offline,
        available_tools=loop_tool_names,
        required_tools=loop_required_tool_names,
        default_tools=DEFAULT_MCP_AVAILABLE_TOOLS,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_tool_result_prompt_chars=max_tool_result_prompt_chars,
        model_timeout_seconds=model_timeout_seconds,
        run_profile=run_profile,
        context=context,
        schema_prompt=schema_prompt,
        capture_prompts=capture_prompts,
        allowed_commands=test_commands,
        trace_repository_root=comparison_repository_root or repository,
        executor_wrapper=lambda executor: PatchPolicyToolExecutor(
            executor,
            apply_patches=apply_patches,
        ),
        run_metadata={
            "interface": "mcp",
            "apply_patches": apply_patches,
            "allowed_test_commands": list(test_commands),
            "model": model_settings,
        },
    )
    result = configured.result
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
                "model": configured.model,
            },
        )
    loop_budget = loop_budget_response(configured, run_profile=run_profile)
    # Return a compact summary rather than the full trace. The full trace can be
    # loaded by debug tooling when exposed.
    return {
        "ok": result.ok,
        "response": result.response,
        "trace_id": str(result.trace_id),
        "turns_used": result.turns_used,
        "tool_calls_made": result.tool_calls_made,
        "loop_budget": loop_budget,
        "model": configured.model,
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


def loop_budget_response(
    configured: ConfiguredToolLoopResult,
    *,
    run_profile: RunProfile | None,
) -> dict[str, int | float | str | None]:
    """Return the MCP response shape for the loop budget."""

    loop_budget: dict[str, int | float | str | None] = {
        "max_turns": configured.budget.max_turns,
        "max_tool_calls": configured.budget.max_tool_calls,
        "max_new_tokens": configured.budget.max_new_tokens,
        "max_tool_result_prompt_chars": configured.budget.max_tool_result_prompt_chars,
    }
    if configured.budget.model_timeout_seconds is not None:
        loop_budget["model_timeout_seconds"] = configured.budget.model_timeout_seconds
    if run_profile is not None:
        loop_budget["run_profile"] = run_profile
    return loop_budget


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
