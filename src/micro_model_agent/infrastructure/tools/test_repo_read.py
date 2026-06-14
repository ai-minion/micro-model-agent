"""Tests for the repo.read tool."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.infrastructure.tools.contracts import (
    RepoReadFileRequest,
    RepoReadRequest,
)
from micro_model_agent.infrastructure.tools.repo_read import RepoReadTool


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def test_repo_read_reads_single_file(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    _write_text(source, "line 1\nline 2\nline 3\n")

    result = RepoReadTool(tmp_path).run(
        RepoReadRequest(files=[RepoReadFileRequest(path="src/example.py")])
    )

    assert result.errors == []
    assert result.files[0].path == "src/example.py"
    assert result.files[0].content == "line 1\nline 2\nline 3\n"


def test_repo_read_applies_line_ranges(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    _write_text(source, "line 1\nline 2\nline 3\n")

    result = RepoReadTool(tmp_path).run(
        RepoReadRequest(
            files=[RepoReadFileRequest(path="src/example.py", start_line=2, end_line=3)]
        )
    )

    assert result.files[0].content == "line 2\nline 3\n"


def test_repo_read_rejects_unsafe_paths(tmp_path: Path) -> None:
    file_request = RepoReadFileRequest.model_construct(
        path="src/../../secret.txt",
        start_line=None,
        end_line=None,
    )
    request = RepoReadRequest.model_construct(files=[file_request], max_bytes=128_000)

    result = RepoReadTool(tmp_path).run(request)

    assert result.files[0].error is not None
    assert result.files[0].error.code == "unsafe_path"


def test_repo_read_rejects_binary_files(tmp_path: Path) -> None:
    binary = tmp_path / "image.bin"
    binary.write_bytes(b"abc\x00def")

    result = RepoReadTool(tmp_path).run(
        RepoReadRequest(files=[RepoReadFileRequest(path="image.bin")])
    )

    assert result.files[0].error is not None
    assert result.files[0].error.code == "binary_file"
