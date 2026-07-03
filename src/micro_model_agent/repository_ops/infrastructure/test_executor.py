"""Tests for the built-in tool executor."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.execution.domain.value_objects import ToolCall
from micro_model_agent.repository_ops.infrastructure.executor import BuiltinToolExecutor


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


def test_builtin_tool_executor_runs_repo_write_files(tmp_path: Path) -> None:
    executor = BuiltinToolExecutor(tmp_path, allowed_test_commands={})

    result = asyncio.run(
        executor.execute(
            ToolCall(
                tool_name="repo.write_files",
                arguments={
                    "files": [{"path": "app/main.py", "content": "VALUE = 1\n"}],
                    "dry_run": False,
                    "require_approval": False,
                },
            )
        )
    )

    assert result.ok is True
    assert result.output["changed_files"] == ["app/main.py"]
    assert (tmp_path / "app" / "main.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_builtin_tool_executor_validates_repo_write_files_paths(tmp_path: Path) -> None:
    executor = BuiltinToolExecutor(tmp_path, allowed_test_commands={})

    result = asyncio.run(
        executor.execute(
            ToolCall(
                tool_name="repo.write_files",
                arguments={
                    "files": [{"path": "../outside.py", "content": "bad"}],
                },
            )
        )
    )

    assert result.ok is False
    assert result.error == "tool argument validation failed"
    assert result.output["validation_errors"][0]["loc"] == ("files", 0, "path")
