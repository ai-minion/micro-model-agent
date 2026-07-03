"""Tests for workspace registry infrastructure."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from micro_model_agent.repository_ops.infrastructure.workspace_registry import (
    JsonlWorkspaceRegistry,
    WorkspaceRecord,
)


def _run(coro):  # type: ignore[return]
    return asyncio.run(coro)


def _record(path: str = "/tmp/project", name: str = "my-project") -> WorkspaceRecord:
    return WorkspaceRecord(path=path, name=name)


def test_save_and_get_by_id(tmp_path: Path) -> None:
    registry = JsonlWorkspaceRegistry(tmp_path / "workspaces.jsonl")
    record = _record()
    _run(registry.save(record))

    loaded = _run(registry.get(str(record.id)))
    assert loaded is not None
    assert loaded.id == record.id
    assert loaded.path == "/tmp/project"


def test_get_missing_id_returns_none(tmp_path: Path) -> None:
    registry = JsonlWorkspaceRegistry(tmp_path / "workspaces.jsonl")
    result = _run(registry.get(str(uuid4())))
    assert result is None


def test_list_empty_returns_empty(tmp_path: Path) -> None:
    registry = JsonlWorkspaceRegistry(tmp_path / "workspaces.jsonl")
    records = _run(registry.list())
    assert records == []


def test_list_after_saves(tmp_path: Path) -> None:
    registry = JsonlWorkspaceRegistry(tmp_path / "workspaces.jsonl")
    for i in range(3):
        _run(registry.save(_record(path=f"/tmp/project_{i}", name=f"project-{i}")))

    records = _run(registry.list())
    assert len(records) == 3


def test_save_twice_with_same_id_latest_wins(tmp_path: Path) -> None:
    registry = JsonlWorkspaceRegistry(tmp_path / "workspaces.jsonl")
    record = _record(name="v1")
    _run(registry.save(record))

    updated = WorkspaceRecord(id=record.id, path=record.path, name="v2")
    _run(registry.save(updated))

    loaded = _run(registry.get(str(record.id)))
    assert loaded is not None
    assert loaded.name == "v2"


def test_record_with_metadata(tmp_path: Path) -> None:
    registry = JsonlWorkspaceRegistry(tmp_path / "workspaces.jsonl")
    record = WorkspaceRecord(path="/tmp/r", metadata={"default_model": "tinyllama"})
    _run(registry.save(record))

    loaded = _run(registry.get(str(record.id)))
    assert loaded is not None
    assert loaded.metadata["default_model"] == "tinyllama"
