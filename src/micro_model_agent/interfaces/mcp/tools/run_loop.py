"""MCP-facing local model run-loop helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.execution.application.ports import ModelProvider
from micro_model_agent.interfaces.composition import (
    ConfiguredToolLoopResult,
    base_model_from_adapter,
    build_loop_model_provider,
    resolve_mcp_model_settings,
    run_mcp_agent_loop,
    string_config_value,
)
from micro_model_agent.interfaces.composition import (
    loop_budget_response as build_loop_budget_response,
)
from micro_model_agent.interfaces.composition import (
    tool_prompt_schemas as build_tool_prompt_schemas,
)
from micro_model_agent.interfaces.mcp.compat import (
    COMPAT_REQUIRED_TOOL_ALIASES,
    COMPAT_TOOL_ALIASES,
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


async def run_agent_loop(**kwargs: Any) -> dict[str, Any]:
    """Run the model-driven tool loop and return a JSON-serializable result."""
    return await run_mcp_agent_loop(
        **kwargs,
        apply_patches=True,
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
