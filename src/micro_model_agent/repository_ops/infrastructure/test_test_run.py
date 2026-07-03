"""Tests for the test.run tool."""

from __future__ import annotations

import sys
from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.command_runner import (
    AllowedTestCommand,
)
from micro_model_agent.repository_ops.infrastructure.command_runner import (
    TestRunTool as ToolTestRunTool,
)
from micro_model_agent.repository_ops.infrastructure.contracts import TestRunRequest as ToolTestRunRequest


def test_test_run_executes_allowlisted_command(tmp_path: Path) -> None:
    tool = ToolTestRunTool(
        tmp_path,
        {
            "python-ok": AllowedTestCommand(
                (sys.executable, "-c", "print('ok from allowlist')")
            )
        },
    )

    result = tool.run(ToolTestRunRequest(command_name="python-ok"))

    assert result.ok is True
    assert result.exit_code == 0
    assert "ok from allowlist" in result.stdout


def test_test_run_rejects_unknown_command(tmp_path: Path) -> None:
    tool = ToolTestRunTool(tmp_path, {})

    result = tool.run(ToolTestRunRequest(command_name="pytest"))

    assert result.ok is False
    assert result.errors[0].code == "command_not_allowlisted"


def test_test_run_reports_nonzero_exit(tmp_path: Path) -> None:
    tool = ToolTestRunTool(
        tmp_path,
        {
            "python-fail": (
                sys.executable,
                "-c",
                "import sys; print('failing'); sys.exit(3)",
            )
        },
    )

    result = tool.run(ToolTestRunRequest(command_name="python-fail"))

    assert result.ok is False
    assert result.exit_code == 3
    assert result.errors[0].code == "command_failed"
    assert "failing" in result.stdout


def test_test_run_times_out(tmp_path: Path) -> None:
    tool = ToolTestRunTool(
        tmp_path,
        {
            "python-sleep": (
                sys.executable,
                "-c",
                "import time; time.sleep(5)",
            )
        },
    )

    result = tool.run(ToolTestRunRequest(command_name="python-sleep", timeout_seconds=1))

    assert result.ok is False
    assert result.timed_out is True
    assert result.exit_code == 124
    assert result.errors[0].code == "timeout"


def test_test_run_passes_extra_args_as_literals(tmp_path: Path) -> None:
    tool = ToolTestRunTool(
        tmp_path,
        {
            "python-argv": (
                sys.executable,
                "-c",
                "import sys; print('|'.join(sys.argv[1:]))",
            )
        },
    )

    result = tool.run(
        ToolTestRunRequest(command_name="python-argv", extra_args=["--pattern", "value;not-shell"])
    )

    assert result.ok is True
    assert "--pattern|value;not-shell" in result.stdout
