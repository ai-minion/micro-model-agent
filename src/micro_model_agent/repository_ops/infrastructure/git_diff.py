"""Implementation of the git.diff tool.

This tool shells out to git in read-only mode. It returns both the diff text and
the list of changed files while keeping paths constrained to the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.paths import RepositoryPathError, RepositoryRoot
from micro_model_agent.repository_ops.infrastructure.contracts import (
    GitDiffRequest,
    GitDiffResult,
    ToolError,
)


class GitDiffTool:
    """Return repository diffs without mutating the worktree."""

    def __init__(
        self,
        repository_root: str | Path,
        git_executable: str = "git",
        timeout_seconds: int = 30,
    ) -> None:
        self.repository = RepositoryRoot(repository_root)
        self.git_executable = git_executable
        self.timeout_seconds = timeout_seconds

    def run(self, request: GitDiffRequest) -> GitDiffResult:
        errors: list[ToolError] = []

        try:
            safe_paths = self._safe_paths(request.paths)
        except RepositoryPathError as exc:
            return GitDiffResult(
                diff="",
                errors=[
                    ToolError(
                        code="unsafe_path",
                        message=str(exc),
                        details={"paths": request.paths},
                    )
                ],
            )

        # Run git diff once for the patch text and once for the file list. The
        # file list is useful even when callers do not need the full diff.
        diff_args = self._base_args(request, name_only=False, paths=safe_paths)
        diff_process = self._run_git(diff_args)
        if diff_process.returncode != 0:
            errors.append(
                ToolError(
                    code="git_diff_failed",
                    message=diff_process.stderr.strip() or "git diff failed",
                    details={"returncode": diff_process.returncode},
                )
            )

        name_args = self._base_args(request, name_only=True, paths=safe_paths)
        name_process = self._run_git(name_args)
        changed_files = (
            [line.strip() for line in name_process.stdout.splitlines() if line.strip()]
            if name_process.returncode == 0
            else []
        )

        diff, truncated = self._truncate(diff_process.stdout, request.max_bytes)
        return GitDiffResult(
            diff=diff,
            changed_files=changed_files,
            truncated=truncated,
            errors=errors,
        )

    def _safe_paths(self, paths: list[str]) -> list[str]:
        """Resolve each requested path and convert it back to repo-relative text."""

        safe_paths: list[str] = []
        for path in paths:
            resolved = self.repository.resolve_file(path)
            safe_paths.append(self.repository.relative_path(resolved))
        return safe_paths

    def _base_args(self, request: GitDiffRequest, name_only: bool, paths: list[str]) -> list[str]:
        """Build the git diff argument list without using a shell."""

        args = [self.git_executable, "diff"]
        if name_only:
            args.append("--name-only")
        else:
            args.append(f"--unified={request.context_lines}")
        if request.cached:
            args.append("--cached")
        if paths:
            args.append("--")
            args.extend(paths)
        return args

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute git and normalize common subprocess failures."""

        try:
            return subprocess.run(
                args,
                cwd=self.repository.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(
                args=args,
                returncode=127,
                stdout="",
                stderr=str(exc),
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                args=args,
                returncode=124,
                stdout=_process_output_text(exc.stdout),
                stderr=f"git diff timed out after {self.timeout_seconds} seconds",
            )

    def _truncate(self, diff: str, max_bytes: int) -> tuple[str, bool]:
        """Trim large diffs by byte count so prompt/storage budgets are respected."""

        encoded = diff.encode("utf-8")
        if len(encoded) <= max_bytes:
            return diff, False
        return encoded[:max_bytes].decode("utf-8", errors="replace"), True


def _process_output_text(output: bytes | str | None) -> str:
    """Normalize subprocess timeout output to text."""

    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output
