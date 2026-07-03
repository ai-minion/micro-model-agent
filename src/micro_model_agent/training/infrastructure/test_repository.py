"""Integration tests for JsonlTrainingJobRepository."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.training.domain.aggregate import TrainingJob
from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from micro_model_agent.training.infrastructure.repository import JsonlTrainingJobRepository


def _run(coro):  # type: ignore[return]
    return asyncio.run(coro)


def _config(tmp: Path) -> TrainingConfig:
    return TrainingConfig(base_model="tinyllama", output_dir=str(tmp / "out"))


def _started_run() -> TrainingRun:
    return TrainingRun(
        kind=TrainingRunKind.SYNTHETIC,
        config=TrainingConfig(base_model="m", output_dir="/tmp"),
        status=TrainingRunStatus.RUNNING,
    )


def _succeeded_run() -> TrainingRun:
    return TrainingRun(
        kind=TrainingRunKind.SYNTHETIC,
        config=TrainingConfig(base_model="m", output_dir="/tmp"),
        status=TrainingRunStatus.SUCCEEDED,
    )


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        name="adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="/tmp/adapter",
        base_model="tinyllama",
    )


def test_add_and_get_round_trip(tmp_path: Path) -> None:
    repo = JsonlTrainingJobRepository(tmp_path)
    job = TrainingJob(config=_config(tmp_path))
    _run(repo.add(job))

    loaded = _run(repo.get(job.id))
    assert loaded is not None
    assert loaded.id == job.id
    assert loaded.config.base_model == "tinyllama"


def test_get_missing_returns_none(tmp_path: Path) -> None:
    from uuid import uuid4
    repo = JsonlTrainingJobRepository(tmp_path)
    assert _run(repo.get(uuid4())) is None


def test_save_updates_run_state(tmp_path: Path) -> None:
    repo = JsonlTrainingJobRepository(tmp_path)
    job = TrainingJob(config=_config(tmp_path))
    _run(repo.add(job))

    job.start(_started_run())
    artifact = _artifact()
    job.complete(_succeeded_run(), artifact)
    _run(repo.save(job))

    loaded = _run(repo.get(job.id))
    assert loaded is not None
    assert loaded.status == TrainingRunStatus.SUCCEEDED
    assert len(loaded.artifacts) == 1


def test_find_by_status_filters(tmp_path: Path) -> None:
    repo = JsonlTrainingJobRepository(tmp_path)

    pending = TrainingJob(config=_config(tmp_path))
    _run(repo.add(pending))

    running = TrainingJob(config=_config(tmp_path))
    running.start(_started_run())
    _run(repo.add(running))

    pending_jobs = _run(repo.find_by_status(TrainingRunStatus.PENDING))
    running_jobs = _run(repo.find_by_status(TrainingRunStatus.RUNNING))

    assert len(pending_jobs) == 1
    assert pending_jobs[0].id == pending.id
    assert len(running_jobs) == 1
    assert running_jobs[0].id == running.id
