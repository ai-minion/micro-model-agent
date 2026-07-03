"""Tests for the repo.search tool."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.contracts import (
    RepoSearchRequest,
    SearchKind,
)
from micro_model_agent.repository_ops.infrastructure.repo_search import RepoSearchTool


def _write_sample_repo(root: Path) -> None:
    src = root / "src"
    docs = root / "docs"
    src.mkdir()
    docs.mkdir()
    (src / "service.py").write_text(
        "class ServiceRunner:\n"
        "    def run_workflow(self) -> None:\n"
        "        print('WorkflowTrace created')\n",
        encoding="utf-8",
    )
    (docs / "architecture.md").write_text(
        "The domain layer owns WorkflowTrace concepts.\n",
        encoding="utf-8",
    )
    (root / "image.bin").write_bytes(b"abc\x00def")


def test_repo_search_finds_text_matches(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSearchTool(tmp_path).run(RepoSearchRequest(query="workflowtrace", limit=10))

    assert result.errors == []
    assert {match.path for match in result.matches} == {
        "docs/architecture.md",
        "src/service.py",
    }


def test_repo_search_lists_glob_matches(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSearchTool(tmp_path).run(
        RepoSearchRequest(glob="src/*.py", kind=SearchKind.GLOB)
    )

    assert [match.path for match in result.matches] == ["src/service.py"]
    assert result.matches[0].metadata["search_kind"] == "glob"


def test_repo_search_finds_python_symbols(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSearchTool(tmp_path).run(
        RepoSearchRequest(query="run_workflow", kind=SearchKind.SYMBOL)
    )

    assert result.errors == []
    assert result.matches[0].path == "src/service.py"
    assert result.matches[0].symbol == "run_workflow"
    assert result.matches[0].line_number == 2


def test_repo_search_rejects_unsafe_globs(tmp_path: Path) -> None:
    result = RepoSearchTool(tmp_path).run(
        RepoSearchRequest.model_validate({"glob": "../*.py", "kind": "glob"})
    )

    assert result.errors
    assert result.errors[0].code == "unsafe_glob"


def test_repo_search_truncates_at_limit(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSearchTool(tmp_path).run(RepoSearchRequest(query="workflow", limit=1))

    assert len(result.matches) == 1
    assert result.truncated is True
