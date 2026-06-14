"""Implementation of the repo.read tool.

This tool reads repository text files while enforcing a byte budget and rejecting
paths that escape the repository root.
"""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.infrastructure.repository_paths import (
    RepositoryPathError,
    RepositoryRoot,
    looks_binary,
)
from micro_model_agent.infrastructure.tools.contracts import (
    RepoReadFileRequest,
    RepoReadFileResult,
    RepoReadRequest,
    RepoReadResult,
    ToolError,
)


class RepoReadTool:
    """Read repository files through a root-constrained adapter."""

    def __init__(self, repository_root: str | Path) -> None:
        self.repository = RepositoryRoot(repository_root)

    def run(self, request: RepoReadRequest) -> RepoReadResult:
        results: list[RepoReadFileResult] = []
        errors: list[ToolError] = []
        total_bytes = 0
        remaining_bytes = request.max_bytes

        for file_request in request.files:
            # The byte budget applies across all requested files, not per file.
            if remaining_bytes <= 0:
                error = ToolError(
                    code="max_bytes_exceeded",
                    message="repo.read max_bytes budget exhausted",
                    details={"path": file_request.path},
                )
                errors.append(error)
                results.append(self._error_result(file_request, error))
                continue

            result = self._read_one(file_request, remaining_bytes)
            results.append(result)
            if result.error is not None:
                errors.append(result.error)
            total_bytes += len(result.content.encode("utf-8"))
            remaining_bytes = max(request.max_bytes - total_bytes, 0)

        return RepoReadResult(files=results, total_bytes=total_bytes, errors=errors)

    def _read_one(self, request: RepoReadFileRequest, max_bytes: int) -> RepoReadFileResult:
        """Read one requested file and return either content or a structured error."""

        try:
            path = self.repository.resolve_file(request.path)
        except RepositoryPathError as exc:
            return self._error_result(
                request,
                ToolError(code="unsafe_path", message=str(exc), details={"path": request.path}),
            )

        if not path.exists():
            return self._error_result(
                request,
                ToolError(
                    code="not_found",
                    message="file does not exist",
                    details={"path": request.path},
                ),
            )
        if not path.is_file():
            return self._error_result(
                request,
                ToolError(
                    code="not_file",
                    message="path is not a file",
                    details={"path": request.path},
                ),
            )

        raw = path.read_bytes()
        if looks_binary(raw[:4096]):
            return self._error_result(
                request,
                ToolError(
                    code="binary_file",
                    message="binary files cannot be read",
                    details={"path": request.path},
                ),
            )

        truncated = len(raw) > max_bytes
        # Slice bytes before decoding so max_bytes stays a true byte limit.
        raw = raw[:max_bytes]
        content = raw.decode("utf-8", errors="replace")
        ranged_content = self._apply_line_range(content, request)

        return RepoReadFileResult(
            path=self.repository.relative_path(path),
            content=ranged_content,
            start_line=request.start_line,
            end_line=request.end_line,
            truncated=truncated,
        )

    def _apply_line_range(self, content: str, request: RepoReadFileRequest) -> str:
        """Return the requested 1-based line range from decoded content."""

        if request.start_line is None:
            return content

        lines = content.splitlines(keepends=True)
        start_index = request.start_line - 1
        end_index = request.end_line if request.end_line is not None else len(lines)
        return "".join(lines[start_index:end_index])

    def _error_result(self, request: RepoReadFileRequest, error: ToolError) -> RepoReadFileResult:
        """Build the standard per-file error result."""

        return RepoReadFileResult(
            path=request.path,
            content="",
            start_line=request.start_line,
            end_line=request.end_line,
            truncated=False,
            error=error,
        )
