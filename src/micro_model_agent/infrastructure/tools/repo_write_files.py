"""Implementation of the repo.write_files tool."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.infrastructure.repositories.paths import RepositoryPathError, RepositoryRoot
from micro_model_agent.infrastructure.tools.contracts import (
    RepoWriteFileResult,
    RepoWriteFilesRequest,
    RepoWriteFilesResult,
    ToolError,
)


class RepoWriteFilesTool:
    """Safely create or replace repository-relative text files."""

    def __init__(self, repository_root: str | Path) -> None:
        self.repository = RepositoryRoot(repository_root)

    def run(self, request: RepoWriteFilesRequest) -> RepoWriteFilesResult:
        resolved_files: list[tuple[str, Path, str, bool]] = []
        errors: list[ToolError] = []
        seen_paths: set[str] = set()

        for file in request.files:
            try:
                resolved = self.repository.resolve_file(file.path)
                relative_path = self.repository.relative_path(resolved)
            except RepositoryPathError as exc:
                errors.append(
                    ToolError(
                        code="unsafe_file_path",
                        message=str(exc),
                        details={"path": file.path},
                    )
                )
                continue

            if relative_path in seen_paths:
                errors.append(
                    ToolError(
                        code="duplicate_file_path",
                        message="files must not contain duplicate paths",
                        details={"path": relative_path},
                    )
                )
                continue

            seen_paths.add(relative_path)
            resolved_files.append((relative_path, resolved, file.content, not resolved.exists()))

        changed_files = [relative_path for relative_path, *_ in resolved_files]
        file_results = [
            RepoWriteFileResult(
                path=relative_path,
                bytes=len(content.encode("utf-8")),
                created=created,
            )
            for relative_path, _, content, created in resolved_files
        ]

        if errors:
            return RepoWriteFilesResult(
                ok=False,
                dry_run=request.dry_run,
                applied=False,
                changed_files=changed_files,
                files=file_results,
                preview=self._preview(file_results, request.dry_run),
                errors=errors,
            )

        if request.dry_run:
            return RepoWriteFilesResult(
                ok=True,
                dry_run=True,
                applied=False,
                changed_files=changed_files,
                files=file_results,
                preview=self._preview(file_results, request.dry_run),
            )

        if request.require_approval:
            return RepoWriteFilesResult(
                ok=False,
                dry_run=False,
                applied=False,
                changed_files=changed_files,
                files=file_results,
                preview=self._preview(file_results, request.dry_run),
                errors=[
                    ToolError(
                        code="approval_required",
                        message="file writes require explicit approval",
                        details={"changed_files": changed_files},
                    )
                ],
            )

        for _, resolved, content, _ in resolved_files:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")

        return RepoWriteFilesResult(
            ok=True,
            dry_run=False,
            applied=True,
            changed_files=changed_files,
            files=file_results,
            preview=self._preview(file_results, request.dry_run),
        )

    def _preview(self, files: list[RepoWriteFileResult], dry_run: bool) -> str:
        action = "Would write" if dry_run else "Write"
        return "\n".join(
            f"{action} {file.path} ({file.bytes} bytes)" for file in files
        )
