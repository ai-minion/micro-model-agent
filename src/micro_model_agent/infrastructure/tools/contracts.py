"""Pydantic contracts for built-in repository and verification tools.

These models describe the shape of every tool request and response. Pydantic
validates incoming dictionaries before any tool touches the filesystem or runs a
command.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base contract that rejects accidental extra fields."""

    model_config = ConfigDict(extra="forbid")


class SearchKind(StrEnum):
    TEXT = "text"
    GLOB = "glob"
    SYMBOL = "symbol"


class ToolError(StrictBaseModel):
    """Machine-readable error shape returned by tool implementations."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RepoSearchRequest(StrictBaseModel):
    """Arguments accepted by the repo.search tool."""

    query: str | None = Field(default=None, min_length=1)
    glob: str | None = Field(default=None, min_length=1)
    kind: SearchKind = SearchKind.TEXT
    limit: int = Field(default=25, ge=1, le=200)

    @model_validator(mode="after")
    def require_query_or_glob(self) -> RepoSearchRequest:
        """Require at least one way to select what should be searched."""

        if self.query is None and self.glob is None:
            raise ValueError("repo.search requires query or glob")
        return self


class RepoSearchMatch(StrictBaseModel):
    """One match returned by repo.search."""

    path: str
    line_number: int | None = Field(default=None, ge=1)
    preview: str
    symbol: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepoSearchResult(StrictBaseModel):
    matches: list[RepoSearchMatch] = Field(default_factory=list)
    truncated: bool = False
    errors: list[ToolError] = Field(default_factory=list)


class RepoReadFileRequest(StrictBaseModel):
    """Request for one file, optionally limited to a line range."""

    path: str = Field(min_length=1)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @field_validator("path")
    @classmethod
    def reject_obviously_unsafe_path(cls, value: str) -> str:
        """Reject path strings that are clearly not repository-relative."""

        normalized = value.replace("\\", "/")
        if "\x00" in normalized:
            raise ValueError("path cannot contain null bytes")
        if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
            raise ValueError("path must be repository-relative")
        return value

    @model_validator(mode="after")
    def validate_line_range(self) -> RepoReadFileRequest:
        """Ensure optional line bounds form a valid range."""

        if self.start_line is None and self.end_line is not None:
            raise ValueError("end_line requires start_line")
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class RepoReadRequest(StrictBaseModel):
    """Arguments accepted by repo.read."""

    files: list[RepoReadFileRequest] = Field(min_length=1, max_length=50)
    max_bytes: int = Field(default=128_000, ge=1, le=2_000_000)


class RepoReadFileResult(StrictBaseModel):
    path: str
    content: str
    start_line: int | None = None
    end_line: int | None = None
    truncated: bool = False
    error: ToolError | None = None


class RepoReadResult(StrictBaseModel):
    files: list[RepoReadFileResult]
    total_bytes: int = Field(ge=0)
    errors: list[ToolError] = Field(default_factory=list)


class SemanticSearchRequest(StrictBaseModel):
    """Arguments accepted by repo.semantic_search."""

    query: str = Field(min_length=1)
    intent: str = Field(default="general", min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedItemContract(StrictBaseModel):
    source_type: str
    title: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticSearchResultContract(StrictBaseModel):
    query: str
    intent: str
    results: list[RetrievedItemContract] = Field(default_factory=list)


class RepoWritePatchRequest(StrictBaseModel):
    """Arguments accepted by repo.write_patch."""

    patch: str = Field(min_length=1)
    dry_run: bool = True
    require_approval: bool = True
    expected_changed_files: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("patch")
    @classmethod
    def require_unified_diff(cls, value: str) -> str:
        """Require the basic file headers used by unified diffs."""

        if "--- " not in value or "+++ " not in value:
            raise ValueError("patch must look like a unified diff")
        return value


class RepoWritePatchResult(StrictBaseModel):
    ok: bool
    dry_run: bool
    applied: bool = False
    preview: str
    changed_files: list[str] = Field(default_factory=list)
    errors: list[ToolError] = Field(default_factory=list)


class RepoWriteFileRequest(StrictBaseModel):
    """One file to create or replace through repo.write_files."""

    path: str = Field(min_length=1)
    content: str

    @field_validator("path")
    @classmethod
    def reject_obviously_unsafe_path(cls, value: str) -> str:
        """Reject path strings that are clearly not repository-relative."""

        normalized = value.replace("\\", "/")
        if "\x00" in normalized:
            raise ValueError("path cannot contain null bytes")
        if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
            raise ValueError("path must be repository-relative")
        return value


class RepoWriteFilesRequest(StrictBaseModel):
    """Arguments accepted by repo.write_files."""

    files: list[RepoWriteFileRequest] = Field(min_length=1, max_length=100)
    dry_run: bool = True
    require_approval: bool = True


class RepoWriteFileResult(StrictBaseModel):
    path: str
    bytes: int = Field(ge=0)
    created: bool


class RepoWriteFilesResult(StrictBaseModel):
    ok: bool
    dry_run: bool
    applied: bool = False
    changed_files: list[str] = Field(default_factory=list)
    files: list[RepoWriteFileResult] = Field(default_factory=list)
    preview: str = ""
    errors: list[ToolError] = Field(default_factory=list)


class TestRunRequest(StrictBaseModel):
    """Arguments accepted by test.run."""

    command_name: str = Field(min_length=1)
    extra_args: list[str] = Field(default_factory=list, max_length=20)
    timeout_seconds: int = Field(default=120, ge=1, le=3_600)

    @field_validator("command_name")
    @classmethod
    def reject_shell_syntax(cls, value: str) -> str:
        """Keep command_name as a lookup key, not as shell text."""

        blocked = ("|", "&", ";", "$", "`", ">", "<")
        if any(token in value for token in blocked):
            raise ValueError("command_name must refer to an allowlisted command key")
        return value


class TestRunResult(StrictBaseModel):
    command_name: str
    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = Field(ge=0.0)
    timed_out: bool = False
    errors: list[ToolError] = Field(default_factory=list)


class GitDiffRequest(StrictBaseModel):
    """Arguments accepted by git.diff."""

    paths: list[str] = Field(default_factory=list, max_length=100)
    cached: bool = False
    context_lines: int = Field(default=3, ge=0, le=100)
    max_bytes: int = Field(default=200_000, ge=1, le=5_000_000)

    @field_validator("paths")
    @classmethod
    def reject_unsafe_paths(cls, value: list[str]) -> list[str]:
        """Reject path traversal before git is invoked."""

        for path in value:
            normalized = path.replace("\\", "/")
            if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
                raise ValueError("paths must be repository-relative")
        return value


class GitDiffResult(StrictBaseModel):
    diff: str
    changed_files: list[str] = Field(default_factory=list)
    truncated: bool = False
    errors: list[ToolError] = Field(default_factory=list)
