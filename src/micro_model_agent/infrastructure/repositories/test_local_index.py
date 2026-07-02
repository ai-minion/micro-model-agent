"""Tests for the local lexical repository index writer."""

from __future__ import annotations

import json
from pathlib import Path

from micro_model_agent.infrastructure.repositories.local_index import (
    LocalLexicalIndexReader,
    LocalLexicalIndexWriter,
)


def test_local_index_writes_files_terms_and_postings(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "class ServiceRunner:\n"
        "    def run_workflow(self) -> None:\n"
        "        return None\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "usage.md").write_text("Run the workflow.\n", encoding="utf-8")

    result = LocalLexicalIndexWriter(tmp_path).write()

    assert result.indexed_file_count == 2
    assert result.source_type_counts == {"documentation": 1, "source_code": 1}
    assert result.extension_counts == {".md": 1, ".py": 1}
    assert result.symbol_count == 2
    assert result.import_count == 0
    assert result.test_file_count == 0
    index = json.loads(Path(result.index_path).read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    assert [file["path"] for file in index["files"]] == [
        "docs/usage.md",
        "src/service.py",
    ]
    assert len(index["files"][0]["sha256"]) == 64
    service_file = next(file for file in index["files"] if file["path"] == "src/service.py")
    assert service_file["source_type"] == "source_code"
    assert service_file["extension"] == ".py"
    assert service_file["is_test"] is False
    assert service_file["symbols"] == [
        {"kind": "ClassDef", "line_number": 1, "name": "ServiceRunner"},
        {"kind": "FunctionDef", "line_number": 2, "name": "run_workflow"},
    ]
    assert index["summary"]["source_type_counts"] == {"documentation": 1, "source_code": 1}
    assert index["summary"]["extension_counts"] == {".md": 1, ".py": 1}
    assert index["summary"]["symbol_count"] == 2
    assert "workflow" in index["postings"]
    assert {posting["path"] for posting in index["postings"]["workflow"]} == {
        "docs/usage.md",
        "src/service.py",
    }


def test_local_index_skips_ignored_large_and_binary_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "small.py").write_text("print('index me')\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    (tmp_path / ".traces").mkdir(parents=True)
    (tmp_path / ".traces" / "workflows.jsonl").write_text(
        "private trace",
        encoding="utf-8",
    )
    (tmp_path / ".venv-wsl").mkdir()
    (tmp_path / ".venv-wsl" / "pyvenv.cfg").write_text("private env", encoding="utf-8")
    (tmp_path / "large.txt").write_text("x" * 200, encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"abc\x00def")

    result = LocalLexicalIndexWriter(tmp_path, max_file_bytes=100).write()

    assert result.indexed_file_count == 1
    assert result.skipped_large_count == 1
    assert result.skipped_binary_count == 1
    index = json.loads(Path(result.index_path).read_text(encoding="utf-8"))
    assert [file["path"] for file in index["files"]] == ["src/small.py"]


def test_local_index_reader_ranks_paths_by_term_overlap(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text(
        "Architecture architecture path safety.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "usage.md").write_text(
        "Usage path safety.\n",
        encoding="utf-8",
    )
    LocalLexicalIndexWriter(tmp_path).write()

    matches = LocalLexicalIndexReader(tmp_path).search_paths({"architecture", "safety"}, limit=2)

    assert [match.path for match in matches] == [
        "docs/architecture.md",
        "docs/usage.md",
    ]
    assert matches[0].score > matches[1].score


def test_local_index_reader_filters_ranked_paths_by_metadata(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "workflow workflow workflow workflow\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "workflow.md").write_text(
        "workflow\n",
        encoding="utf-8",
    )
    LocalLexicalIndexWriter(tmp_path).write()

    matches = LocalLexicalIndexReader(tmp_path).search_paths(
        {"workflow"},
        limit=1,
        source_types={"documentation"},
    )

    assert [match.path for match in matches] == ["docs/workflow.md"]


def test_local_index_reader_reports_fresh_index(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text("workflow\n", encoding="utf-8")
    LocalLexicalIndexWriter(tmp_path).write()

    freshness = LocalLexicalIndexReader(tmp_path).freshness()

    assert freshness is not None
    assert freshness.is_stale is False
    assert freshness.as_metadata() == {
        "is_stale": False,
        "indexed_file_count": 1,
        "changed_file_count": 0,
        "missing_file_count": 0,
        "extra_file_count": 0,
    }


def test_local_index_reader_reports_changed_missing_and_extra_files(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "changed.md").write_text("old workflow\n", encoding="utf-8")
    (tmp_path / "docs" / "missing.md").write_text("removed workflow\n", encoding="utf-8")
    LocalLexicalIndexWriter(tmp_path).write()

    (tmp_path / "docs" / "changed.md").write_text("new workflow\n", encoding="utf-8")
    (tmp_path / "docs" / "missing.md").unlink()
    (tmp_path / "docs" / "extra.md").write_text("extra workflow\n", encoding="utf-8")

    freshness = LocalLexicalIndexReader(tmp_path).freshness()

    assert freshness is not None
    assert freshness.is_stale is True
    assert freshness.changed_file_count == 1
    assert freshness.missing_file_count == 1
    assert freshness.extra_file_count == 1


def test_local_index_extracts_python_import_metadata(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "test_service.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "async def load_service() -> Path:\n"
        "    return Path(json.dumps({}))\n",
        encoding="utf-8",
    )
    LocalLexicalIndexWriter(tmp_path).write()

    metadata = LocalLexicalIndexReader(tmp_path).file_metadata("src/test_service.py")

    assert metadata is not None
    assert metadata["is_test"] is True
    assert metadata["imports"] == ["json", "pathlib"]
    assert metadata["symbols"] == [
        {"kind": "AsyncFunctionDef", "line_number": 4, "name": "load_service"}
    ]
