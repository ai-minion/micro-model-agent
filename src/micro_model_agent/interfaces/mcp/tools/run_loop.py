"""MCP-facing local model run-loop helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.application.ports.contracts import ModelProvider
from micro_model_agent.infrastructure.composition import (
    ConfiguredToolLoopResult,
    base_model_from_adapter,
    build_loop_model_provider,
    resolve_mcp_model_settings,
    run_mcp_agent_loop,
    string_config_value,
)
from micro_model_agent.infrastructure.composition import (
    loop_budget_response as build_loop_budget_response,
)
from micro_model_agent.infrastructure.composition import (
    tool_prompt_schemas as build_tool_prompt_schemas,
)
from micro_model_agent.interfaces.mcp.compat import (
    COMPAT_REQUIRED_TOOL_ALIASES,
    COMPAT_TOOL_ALIASES,
    DEFAULT_7B_ADAPTER_PATH,
    DEFAULT_MCP_AVAILABLE_TOOLS,
    RunProfile,
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

    return await run_mcp_agent_loop(
        goal=goal,
        repository_root=repository_root,
        context=context,
        adapter_path=adapter_path,
        base_model=base_model,
        use_adapter=use_adapter,
        available_tools=available_tools,
        required_tools=required_tools,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_new_tokens=max_new_tokens,
        max_tool_result_prompt_chars=max_tool_result_prompt_chars,
        model_timeout_seconds=model_timeout_seconds,
        run_profile=run_profile,
        schema_prompt=schema_prompt,
        capture_prompts=capture_prompts,
        apply_patches=apply_patches,
        allow_test_run=allow_test_run,
        test_command_name=test_command_name,
        test_command_args=test_command_args,
        scripted_responses=scripted_responses,
        comparison_session_id=comparison_session_id,
        comparison_repository_root=comparison_repository_root,
        offline=offline,
        default_adapter_path=DEFAULT_7B_ADAPTER_PATH,
        default_available_tools=DEFAULT_MCP_AVAILABLE_TOOLS,
        tool_aliases=COMPAT_TOOL_ALIASES,
        required_tool_aliases=COMPAT_REQUIRED_TOOL_ALIASES,
    )


def loop_budget_response(
    configured: ConfiguredToolLoopResult,
    *,
    run_profile: RunProfile | None,
) -> dict[str, int | float | str | None]:
    """Return the MCP response shape for the loop budget."""

    return build_loop_budget_response(configured, run_profile=run_profile)


def model_provider_for_loop(
    *,
    adapter_path: str | None,
    base_model: str | None,
    max_new_tokens: int,
    scripted_responses: list[str] | None,
    offline: bool,
) -> ModelProvider:
    """Return the model provider used by the MCP run loop."""

    return build_loop_model_provider(
        adapter_path=adapter_path,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        scripted_responses=scripted_responses,
        offline=offline,
    )


def resolve_model_settings(
    *,
    repository_root: str | Path,
    adapter_path: str | None,
    base_model: str | None,
    use_adapter: bool,
    allow_missing_base_model: bool = False,
) -> dict[str, str | None]:
    """Resolve MCP model settings from explicit args, env, selected config, defaults."""

    return resolve_mcp_model_settings(
        repository_root=repository_root,
        adapter_path=adapter_path,
        base_model=base_model,
        use_adapter=use_adapter,
        default_adapter_path=DEFAULT_7B_ADAPTER_PATH,
        allow_missing_base_model=allow_missing_base_model,
    )


def tool_prompt_schemas(
    allowed_tool_names: tuple[str, ...],
    *,
    allowed_test_commands: dict[str, object],
) -> dict[str, Any]:
    """Compatibility wrapper for shared tool schema enrichment."""

    return build_tool_prompt_schemas(
        allowed_tool_names,
        allowed_test_commands=allowed_test_commands,
    )
