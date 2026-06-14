"""Tests for the built-in tool executor."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.domain.contracts import ToolCall
from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor


def test_builtin_tool_executor_runs_repo_search(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text("class Example:\n    pass\n", encoding="utf-8")
    executor = BuiltinToolExecutor(tmp_path, allowed_test_commands={})

    result = asyncio.run(
        executor.execute(ToolCall(tool_name="repo.search", arguments={"query": "Example"}))
    )

    assert result.ok is True
    assert result.output["matches"][0]["path"] == "example.py"


def test_builtin_tool_executor_returns_unknown_tool_error(tmp_path: Path) -> None:
    executor = BuiltinToolExecutor(tmp_path, allowed_test_commands={})

    result = asyncio.run(
        executor.execute(ToolCall(tool_name="shell.exec", arguments={"command": "pytest"}))
    )

    assert result.ok is False
    assert result.error == "unknown tool: shell.exec"


def test_builtin_tool_executor_returns_validation_errors(tmp_path: Path) -> None:
    executor = BuiltinToolExecutor(tmp_path, allowed_test_commands={})

    result = asyncio.run(
        executor.execute(ToolCall(tool_name="repo.read", arguments={"files": []}))
    )

    assert result.ok is False
    assert result.error == "tool argument validation failed"
    assert result.output["validation_errors"]
