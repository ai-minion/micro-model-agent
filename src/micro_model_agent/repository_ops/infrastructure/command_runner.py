"""Implementation of the test.run tool.

The test runner only executes commands that were explicitly allowlisted by name.
It passes argument lists directly to subprocess.run with shell=False.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.contracts import (
    TestRunRequest,
    TestRunResult,
    ToolError,
)
from micro_model_agent.repository_ops.infrastructure.paths import RepositoryRoot


@dataclass(frozen=True, slots=True)
class AllowedTestCommand:
    """A shell-free command definition that can be selected by name."""

    args: tuple[str, ...]
    description: str = ""


class TestRunTool:
    """Run allowlisted verification commands with timeout and structured output."""

    def __init__(
        self,
        repository_root: str | Path,
        allowed_commands: Mapping[str, AllowedTestCommand | Sequence[str]],
    ) -> None:
        self.repository = RepositoryRoot(repository_root)
        self.allowed_commands = {
            name: command
            if isinstance(command, AllowedTestCommand)
            else AllowedTestCommand(tuple(command))
            for name, command in allowed_commands.items()
        }

    def run(self, request: TestRunRequest) -> TestRunResult:
        command = self.allowed_commands.get(request.command_name)
        if command is None:
            return TestRunResult(
                command_name=request.command_name,
                ok=False,
                duration_seconds=0.0,
                errors=[
                    ToolError(
                        code="command_not_allowlisted",
                        message="test command is not allowlisted",
                        details={"command_name": request.command_name},
                    )
                ],
            )

        args = [*command.args, *request.extra_args]
        started = time.perf_counter()
        try:
            # shell=False means subprocess runs exactly this argument list rather
            # than asking a shell to interpret a string.
            process = subprocess.run(
                args,
                cwd=self.repository.root,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            # The executable was not found on PATH or at the provided location.
            duration = time.perf_counter() - started
            return TestRunResult(
                command_name=request.command_name,
                ok=False,
                exit_code=127,
                stderr=str(exc),
                duration_seconds=duration,
                errors=[
                    ToolError(
                        code="command_not_found",
                        message=str(exc),
                        details={"command_name": request.command_name},
                    )
                ],
            )
        except subprocess.TimeoutExpired as exc:
            # Preserve any partial stdout/stderr captured before the timeout.
            duration = time.perf_counter() - started
            return TestRunResult(
                command_name=request.command_name,
                ok=False,
                exit_code=124,
                stdout=self._decode_timeout_output(exc.stdout),
                stderr=self._decode_timeout_output(exc.stderr),
                duration_seconds=duration,
                timed_out=True,
                errors=[
                    ToolError(
                        code="timeout",
                        message=f"test command timed out after {request.timeout_seconds} seconds",
                        details={"command_name": request.command_name},
                    )
                ],
            )

        duration = time.perf_counter() - started
        return TestRunResult(
            command_name=request.command_name,
            ok=process.returncode == 0,
            exit_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            duration_seconds=duration,
            timed_out=False,
            errors=[] if process.returncode == 0 else [self._failed_error(process.returncode)],
        )

    def _failed_error(self, exit_code: int) -> ToolError:
        """Build the standard error returned for nonzero command exits."""

        return ToolError(
            code="command_failed",
            message=f"test command exited with code {exit_code}",
            details={"exit_code": exit_code},
        )

    def _decode_timeout_output(self, output: str | bytes | None) -> str:
        """Normalize timeout output because subprocess may return bytes."""

        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output
