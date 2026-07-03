"""Tests for repository path safety utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from micro_model_agent.repository_ops.infrastructure.paths import RepositoryPathError, RepositoryRoot


def test_repository_root_resolves_relative_paths(tmp_path: Path) -> None:
    repository = RepositoryRoot(tmp_path)
    file_path = tmp_path / "src" / "example.py"
    file_path.parent.mkdir()
    file_path.write_text("print('hello')", encoding="utf-8")

    resolved = repository.resolve_file("src/example.py")

    assert resolved == file_path.resolve()
    assert repository.relative_path(resolved) == "src/example.py"


def test_repository_root_rejects_parent_traversal(tmp_path: Path) -> None:
    repository = RepositoryRoot(tmp_path)

    with pytest.raises(RepositoryPathError):
        repository.resolve_file("../outside.txt")


def test_repository_root_iter_files_skips_excluded_directories(tmp_path: Path) -> None:
    repository = RepositoryRoot(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "example.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("private", encoding="utf-8")

    files = [repository.relative_path(path) for path in repository.iter_files("**/*")]

    assert files == ["src/example.py"]
