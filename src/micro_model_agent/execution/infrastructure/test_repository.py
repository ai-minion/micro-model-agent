"""Tests for JsonlWorkflowRepository."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.execution.domain.value_objects import WorkflowStatus
from micro_model_agent.execution.domain.aggregate import WorkflowExecution
from micro_model_agent.execution.infrastructure.repository import JsonlWorkflowRepository


def _run(coro):  # type: ignore[return]
    return asyncio.run(coro)


def test_add_and_get_round_trip(tmp_path: Path) -> None:
    repo = JsonlWorkflowRepository(tmp_path / "traces.jsonl")
    execution = WorkflowExecution(goal="fix the null pointer")
    execution.start()
    execution.complete({"ok": True})

    _run(repo.add(execution))
    loaded = _run(repo.get(execution.id))

    assert loaded is not None
    assert loaded.id == execution.id
    assert loaded.goal == "fix the null pointer"
    assert loaded.status == WorkflowStatus.SUCCEEDED


def test_get_missing_returns_none(tmp_path: Path) -> None:
    from uuid import uuid4

    repo = JsonlWorkflowRepository(tmp_path / "traces.jsonl")
    assert _run(repo.get(uuid4())) is None


def test_save_updates_existing(tmp_path: Path) -> None:
    repo = JsonlWorkflowRepository(tmp_path / "traces.jsonl")
    execution = WorkflowExecution(goal="refactor module")
    execution.start()
    _run(repo.add(execution))

    execution.complete({"ok": True})
    _run(repo.save(execution))

    loaded = _run(repo.get(execution.id))
    assert loaded is not None
    assert loaded.status == WorkflowStatus.SUCCEEDED


def test_find_by_status_filters_correctly(tmp_path: Path) -> None:
    repo = JsonlWorkflowRepository(tmp_path / "traces.jsonl")

    ok = WorkflowExecution(goal="passed run")
    ok.start()
    ok.complete({"ok": True})
    _run(repo.add(ok))

    failed = WorkflowExecution(goal="failed run")
    failed.start()
    failed.fail("model error")
    _run(repo.add(failed))

    succeeded = _run(repo.find_by_status(WorkflowStatus.SUCCEEDED))
    assert len(succeeded) == 1
    assert succeeded[0].goal == "passed run"

    failures = _run(repo.find_by_status(WorkflowStatus.FAILED))
    assert len(failures) == 1
    assert failures[0].goal == "failed run"
