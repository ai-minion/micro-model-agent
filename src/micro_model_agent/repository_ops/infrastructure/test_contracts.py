"""Tests for built-in Pydantic tool contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from micro_model_agent.repository_ops.infrastructure.contracts import (
    GitDiffRequest,
    RepoReadFileRequest,
    RepoSearchRequest,
    RepoWritePatchRequest,
    SemanticSearchRequest,
)
from micro_model_agent.repository_ops.infrastructure.contracts import (
    TestRunRequest as ToolTestRunRequest,
)


def test_repo_search_requires_query_or_glob() -> None:
    with pytest.raises(ValidationError):
        RepoSearchRequest()

    request = RepoSearchRequest(query="DatasetExample")

    assert request.query == "DatasetExample"


def test_repo_read_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        RepoReadFileRequest(path="../secret.txt")


def test_repo_read_validates_line_range() -> None:
    with pytest.raises(ValidationError):
        RepoReadFileRequest(path="src/example.py", start_line=10, end_line=2)


def test_semantic_search_defaults_to_general_intent() -> None:
    request = SemanticSearchRequest(query="tool schema")

    assert request.intent == "general"


def test_write_patch_requires_unified_diff_shape() -> None:
    with pytest.raises(ValidationError):
        RepoWritePatchRequest(patch="plain text")


def test_test_run_rejects_shell_syntax() -> None:
    with pytest.raises(ValidationError):
        ToolTestRunRequest(command_name="pytest; rm -rf .")


def test_git_diff_rejects_absolute_paths() -> None:
    with pytest.raises(ValidationError):
        GitDiffRequest(paths=["/etc/passwd"])
