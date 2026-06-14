"""Tests for the git.diff tool."""

from __future__ import annotations

import subprocess
from pathlib import Path

from micro_model_agent.infrastructure.tools.contracts import GitDiffRequest
from micro_model_agent.infrastructure.tools.git_diff import GitDiffTool


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def _init_repo(root: Path) -> None:
    _run_git(root, "init")
    _run_git(root, "config", "core.autocrlf", "false")
    _write_text(root / "example.py", "def value():\n    return 1\n")
    _run_git(root, "add", "example.py")


def test_git_diff_returns_worktree_diff(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_text(tmp_path / "example.py", "def value():\n    return 2\n")

    result = GitDiffTool(tmp_path).run(GitDiffRequest(paths=["example.py"]))

    assert result.errors == []
    assert result.changed_files == ["example.py"]
    assert "-    return 1" in result.diff
    assert "+    return 2" in result.diff


def test_git_diff_returns_cached_diff(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_text(tmp_path / "example.py", "def value():\n    return 3\n")
    _run_git(tmp_path, "add", "example.py")

    result = GitDiffTool(tmp_path).run(GitDiffRequest(paths=["example.py"], cached=True))

    assert result.errors == []
    assert result.changed_files == ["example.py"]
    assert "+    return 3" in result.diff


def test_git_diff_rejects_unsafe_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    request = GitDiffRequest.model_construct(
        paths=["../outside.py"],
        cached=False,
        context_lines=3,
        max_bytes=200_000,
    )

    result = GitDiffTool(tmp_path).run(request)

    assert result.errors
    assert result.errors[0].code == "unsafe_path"


def test_git_diff_truncates_large_output(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_text(tmp_path / "example.py", "def value():\n    return 'changed value'\n")

    result = GitDiffTool(tmp_path).run(GitDiffRequest(paths=["example.py"], max_bytes=20))

    assert result.truncated is True
    assert len(result.diff.encode("utf-8")) <= 20
