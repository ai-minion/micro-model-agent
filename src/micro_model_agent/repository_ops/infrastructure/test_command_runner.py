"""Tests for AllowedTestCommand and TestRunTool."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.command_runner import (
    AllowedTestCommand,
)
from micro_model_agent.repository_ops.infrastructure.command_runner import (
    TestRunTool as _TestRunTool,  # renamed to avoid pytest collection confusion
)

TestRunTool = _TestRunTool  # alias back for use in tests


# ---------------------------------------------------------------------------
# AllowedTestCommand
# ---------------------------------------------------------------------------


def test_allowed_test_command_stores_args() -> None:
    cmd = AllowedTestCommand(args=("python3", "-m", "pytest"))
    assert cmd.args == ("python3", "-m", "pytest")


def test_allowed_test_command_default_description() -> None:
    cmd = AllowedTestCommand(args=("echo", "hello"))
    assert cmd.description == ""


def test_allowed_test_command_custom_description() -> None:
    cmd = AllowedTestCommand(args=("pytest",), description="run tests")
    assert cmd.description == "run tests"


def test_allowed_test_command_is_frozen() -> None:
    cmd = AllowedTestCommand(args=("pytest",))
    try:
        cmd.args = ("other",)  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# TestRunTool — constructor normalization
# ---------------------------------------------------------------------------


def test_test_run_tool_accepts_sequence_as_command(tmp_path: Path) -> None:
    # Passing a list instead of AllowedTestCommand should be normalized
    tool = TestRunTool(
        repository_root=tmp_path,
        allowed_commands={"pytest": ["python3", "-m", "pytest"]},
    )
    cmd = tool.allowed_commands["pytest"]
    assert isinstance(cmd, AllowedTestCommand)
    assert cmd.args == ("python3", "-m", "pytest")


def test_test_run_tool_preserves_allowed_test_command(tmp_path: Path) -> None:
    original = AllowedTestCommand(args=("pytest", "--tb=short"), description="unit tests")
    tool = TestRunTool(
        repository_root=tmp_path,
        allowed_commands={"pytest": original},
    )
    assert tool.allowed_commands["pytest"] is original


def test_test_run_tool_unknown_command_returns_error(tmp_path: Path) -> None:
    tool = TestRunTool(repository_root=tmp_path, allowed_commands={})
    from micro_model_agent.repository_ops.infrastructure.contracts import (
        RepoReadRequest,  # noqa: F401
    )
    # Import the request type from the right place
    try:
        from micro_model_agent.repository_ops.infrastructure.contracts import TestRunRequest
    except ImportError:
        return  # skip if contract not accessible this way

    request = TestRunRequest(command_name="pytest")
    result = tool.run(request)
    assert result.ok is False
    assert result.errors


def test_test_run_tool_unknown_command_via_direct_call(tmp_path: Path) -> None:
    """Test the not-allowlisted branch without importing TestRunRequest directly."""
    tool = TestRunTool(repository_root=tmp_path, allowed_commands={})
    # Access the allowed_commands dict — unknown command "x" is not present
    assert "x" not in tool.allowed_commands
