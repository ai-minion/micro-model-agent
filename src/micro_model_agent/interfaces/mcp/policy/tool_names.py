"""Tool-name normalization and runtime allowlist policy for MCP."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.application.tool_loop import (
    RunProfile,
    normalized_tool_names,
    select_loop_tool_names,
)
from micro_model_agent.application.tool_loop import (
    run_profile_settings as application_run_profile_settings,
)
from micro_model_agent.infrastructure.composition import (
    allowed_test_commands as build_allowed_test_commands,
)
from micro_model_agent.infrastructure.tools.catalog import BUILTIN_TOOL_SPECS
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.interfaces.mcp.compat import (
    COMPAT_REQUIRED_TOOL_ALIASES,
    COMPAT_TOOL_ALIASES,
    DEFAULT_MCP_AVAILABLE_TOOLS,
)


def allowed_tool_names(
    *,
    available_tools: list[str] | None,
    repository_root: Path,
    apply_patches: bool,
    allow_test_run: bool,
) -> tuple[str, ...]:
    """Compatibility wrapper for application tool-name selection."""

    requested = normalized_tool_names(
        available_tools,
        default_tools=DEFAULT_MCP_AVAILABLE_TOOLS,
        aliases=COMPAT_TOOL_ALIASES,
    )
    _ = apply_patches
    return select_loop_tool_names(
        requested,
        default_tools=(),
        known_tools=tuple(BUILTIN_TOOL_SPECS),
        repository_has_git=is_git_repository(repository_root),
        allow_test_run=allow_test_run,
    )


def required_tool_names(required_tools: list[str] | None) -> tuple[str, ...]:
    """Normalize required tool names from external MCP callers."""

    return normalized_tool_names(
        required_tools,
        default_tools=(),
        aliases=COMPAT_REQUIRED_TOOL_ALIASES,
    )


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


def run_profile_settings(profile: RunProfile | None) -> dict[str, int | float | None]:
    """Compatibility wrapper for application run-profile budgets."""

    return application_run_profile_settings(profile)


def is_git_repository(repository_root: Path) -> bool:
    """Return whether git.diff can operate in this repository root."""

    return (repository_root / ".git").exists()
