"""Built-in tool executor registry.

The executor is the bridge between model-authored tool calls and concrete Python
tool classes. It validates arguments before running anything and always returns
a structured ToolResult.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from micro_model_agent.domain.contracts import ToolCall, ToolResult
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand, TestRunTool
from micro_model_agent.infrastructure.tools.contracts import (
    GitDiffRequest,
    RepoReadRequest,
    RepoSearchRequest,
    RepoWritePatchRequest,
    SemanticSearchRequest,
    TestRunRequest,
)
from micro_model_agent.infrastructure.tools.git_diff import GitDiffTool
from micro_model_agent.infrastructure.tools.repo_read import RepoReadTool
from micro_model_agent.infrastructure.tools.repo_search import RepoSearchTool
from micro_model_agent.infrastructure.tools.repo_semantic_search import RepoSemanticSearchTool
from micro_model_agent.infrastructure.tools.repo_write_patch import RepoWritePatchTool

ToolRunner = Callable[[dict[str, object]], BaseModel]


class BuiltinToolExecutor:
    """Execute the built-in tools through validated request contracts."""

    def __init__(
        self,
        repository_root: str | Path,
        allowed_test_commands: Mapping[str, AllowedTestCommand | Sequence[str]],
    ) -> None:
        self.repository_root = Path(repository_root)
        self.allowed_test_commands = allowed_test_commands
        # Each tool name maps to a small adapter method that validates the
        # arguments and calls the concrete tool implementation.
        self._tools: dict[str, ToolRunner] = {
            "repo.search": self._repo_search,
            "repo.read": self._repo_read,
            "repo.semantic_search": self._repo_semantic_search,
            "repo.write_patch": self._repo_write_patch,
            "test.run": self._test_run,
            "git.diff": self._git_diff,
        }

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        runner = self._tools.get(tool_call.tool_name)
        if runner is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                ok=False,
                error=f"unknown tool: {tool_call.tool_name}",
            )

        try:
            # The runner returns a Pydantic model. If validation fails, convert
            # the exception into a normal tool failure instead of crashing.
            result = runner(tool_call.arguments)
        except ValidationError as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                ok=False,
                output={"validation_errors": exc.errors(include_context=False)},
                error="tool argument validation failed",
            )

        output = result.model_dump(mode="json")
        # Tool models are converted to plain dictionaries so they can be stored
        # in traces and passed back into prompts.
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            ok=self._is_successful(output),
            output=output,
            error=self._error_summary(output),
        )

    def _repo_search(self, arguments: dict[str, object]) -> BaseModel:
        return RepoSearchTool(self.repository_root).run(RepoSearchRequest.model_validate(arguments))

    def _repo_read(self, arguments: dict[str, object]) -> BaseModel:
        return RepoReadTool(self.repository_root).run(RepoReadRequest.model_validate(arguments))

    def _repo_semantic_search(self, arguments: dict[str, object]) -> BaseModel:
        return RepoSemanticSearchTool(self.repository_root).run(
            SemanticSearchRequest.model_validate(arguments)
        )

    def _repo_write_patch(self, arguments: dict[str, object]) -> BaseModel:
        return RepoWritePatchTool(self.repository_root).run(
            RepoWritePatchRequest.model_validate(arguments)
        )

    def _test_run(self, arguments: dict[str, object]) -> BaseModel:
        return TestRunTool(self.repository_root, self.allowed_test_commands).run(
            TestRunRequest.model_validate(arguments)
        )

    def _git_diff(self, arguments: dict[str, object]) -> BaseModel:
        return GitDiffTool(self.repository_root).run(GitDiffRequest.model_validate(arguments))

    def _is_successful(self, output: dict[str, object]) -> bool:
        """Interpret common result shapes as a boolean success value."""

        if "ok" in output:
            return bool(output["ok"])
        errors = output.get("errors")
        if isinstance(errors, list):
            return len(errors) == 0
        return True

    def _error_summary(self, output: dict[str, object]) -> str | None:
        """Return a short error message from a tool output, if one exists."""

        errors = output.get("errors")
        if not isinstance(errors, list) or not errors:
            return None
        first_error = errors[0]
        if isinstance(first_error, dict):
            message = first_error.get("message")
            return str(message) if message is not None else str(first_error)
        return str(first_error)
