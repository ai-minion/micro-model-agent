"""Tool-name normalization and runtime allowlist policy for MCP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.infrastructure.composition import (
    allowed_test_commands as build_allowed_test_commands,
)
from micro_model_agent.infrastructure.tools.catalog import BUILTIN_TOOL_SPECS
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.interfaces.mcp.compat import (
    COMPAT_REQUIRED_TOOL_ALIASES,
    COMPAT_TOOL_ALIASES,
    DEFAULT_MCP_AVAILABLE_TOOLS,
    RunProfile,
)


def allowed_tool_names(
    *,
    available_tools: list[str] | None,
    repository_root: Path,
    apply_patches: bool,
    allow_test_run: bool,
) -> tuple[str, ...]:
    """Filter requested tool names according to MCP safety options."""

    requested = normalized_tool_names(
        available_tools,
        default_tools=DEFAULT_MCP_AVAILABLE_TOOLS,
        aliases=COMPAT_TOOL_ALIASES,
    )
    allowed: list[str] = []
    for tool_name in requested:
        if tool_name not in BUILTIN_TOOL_SPECS:
            raise ValueError(f"unknown built-in tool: {tool_name}")
        if tool_name in {"repo.write_patch", "repo.write_files"} and not apply_patches:
            # The wrapper will force this tool into dry-run mode.
            allowed.append(tool_name)
            continue
        if tool_name == "git.diff" and not is_git_repository(repository_root):
            continue
        if tool_name == "test.run" and not allow_test_run:
            continue
        allowed.append(tool_name)
    return tuple(dict.fromkeys(allowed))


def required_tool_names(required_tools: list[str] | None) -> tuple[str, ...]:
    """Normalize required tool names from external MCP callers."""

    return normalized_tool_names(
        required_tools,
        default_tools=(),
        aliases=COMPAT_REQUIRED_TOOL_ALIASES,
    )


def normalized_tool_names(
    tool_names: list[str] | None,
    *,
    default_tools: tuple[str, ...],
    aliases: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Expand compatibility aliases and preserve first-seen order."""

    if not tool_names:
        return default_tools

    normalized: list[str] = []
    for tool_name in tool_names:
        normalized.extend(aliases.get(tool_name, (tool_name,)))
    return tuple(dict.fromkeys(normalized))


def allowed_test_commands(
    test_command_name: str | None,
    test_command_args: list[str] | None,
    *,
    use_default_pytest: bool = False,
) -> dict[str, AllowedTestCommand]:
    """Build the allowlist consumed by the test.run tool."""

    return build_allowed_test_commands(
        test_command_name,
        test_command_args,
        use_default_pytest=use_default_pytest,
    )


def run_profile_settings(profile: RunProfile | None) -> dict[str, Any]:
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


def is_git_repository(repository_root: Path) -> bool:
    """Return whether git.diff can operate in this repository root."""

    return (repository_root / ".git").exists()
