"""Tests for the repo.write_files tool."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.contracts import (
    RepoWriteFileRequest,
    RepoWriteFilesRequest,
)
from micro_model_agent.repository_ops.infrastructure.repo_write_files import RepoWriteFilesTool


def test_repo_write_files_dry_run_previews_without_writing(tmp_path: Path) -> None:
    result = RepoWriteFilesTool(tmp_path).run(
        RepoWriteFilesRequest(
            files=[
                RepoWriteFileRequest(path="app/main.py", content="print('hi')\n"),
            ],
            dry_run=True,
        )
    )

    assert result.ok is True
    assert result.applied is False
    assert result.changed_files == ["app/main.py"]
    assert result.files[0].created is True
    assert not (tmp_path / "app" / "main.py").exists()


def test_repo_write_files_requires_approval_before_real_writes(tmp_path: Path) -> None:
    result = RepoWriteFilesTool(tmp_path).run(
        RepoWriteFilesRequest(
            files=[RepoWriteFileRequest(path="README.md", content="# Demo\n")],
            dry_run=False,
            require_approval=True,
        )
    )

    assert result.ok is False
    assert result.applied is False
    assert result.errors[0].code == "approval_required"
    assert not (tmp_path / "README.md").exists()


def test_repo_write_files_applies_when_approval_disabled(tmp_path: Path) -> None:
    result = RepoWriteFilesTool(tmp_path).run(
        RepoWriteFilesRequest(
            files=[
                RepoWriteFileRequest(path="app/__init__.py", content=""),
                RepoWriteFileRequest(path="app/main.py", content="VALUE = 1\n"),
            ],
            dry_run=False,
            require_approval=False,
        )
    )

    assert result.ok is True
    assert result.applied is True
    assert result.changed_files == ["app/__init__.py", "app/main.py"]
    assert (tmp_path / "app" / "main.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_repo_write_files_rejects_duplicate_paths(tmp_path: Path) -> None:
    result = RepoWriteFilesTool(tmp_path).run(
        RepoWriteFilesRequest(
            files=[
                RepoWriteFileRequest(path="app.py", content="one"),
                RepoWriteFileRequest(path="./app.py", content="two"),
            ],
        )
    )

    assert result.ok is False
    assert [error.code for error in result.errors] == ["duplicate_file_path"]
    assert not (tmp_path / "app.py").exists()
