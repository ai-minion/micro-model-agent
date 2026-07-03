"""Tests for repository runtime composition helpers."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.local_index import LocalIndexResult
from micro_model_agent.repository_ops.infrastructure.runtime import (
    initialize_local_repository,
    local_repository_initialized,
    write_local_repository_index,
)


def test_repository_runtime_initializes_and_indexes_workspace(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def run() -> None:\n    pass\n", encoding="utf-8")

    assert local_repository_initialized(tmp_path) is False

    init_result = initialize_local_repository(tmp_path, default_model="qwen:test")
    index_result = write_local_repository_index(tmp_path, max_file_bytes=1_000_000)

    assert init_result.ok
    assert init_result.config["model"]["default_model"] == "qwen:test"
    assert local_repository_initialized(tmp_path) is True
    assert isinstance(index_result, LocalIndexResult)
    assert index_result.indexed_file_count >= 1
