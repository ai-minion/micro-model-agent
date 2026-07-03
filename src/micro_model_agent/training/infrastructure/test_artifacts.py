"""Tests for training infrastructure artifact stores and runners."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from micro_model_agent.training.domain.value_objects import (
    ModelArtifactKind,
    TrainingConfig,
    TrainingRunStatus,
)
from micro_model_agent.training.infrastructure.artifacts import (
    FakeTrainingRunner,
    JsonTrainingArtifactStore,
)


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _config(output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        base_model="tinyllama",
        output_dir=str(output_dir),
        dry_run=True,
        max_steps=5,
        parameters={"example_count": 10},
    )


# ---------------------------------------------------------------------------
# FakeTrainingRunner
# ---------------------------------------------------------------------------


def test_fake_runner_produces_succeeded_run(tmp_path: Path) -> None:
    runner = FakeTrainingRunner()
    config = _config(tmp_path)
    run = _run(runner.run(config))

    assert run.status == TrainingRunStatus.SUCCEEDED
    assert len(run.artifacts) >= 1


def test_fake_runner_writes_metadata_file(tmp_path: Path) -> None:
    runner = FakeTrainingRunner()
    config = _config(tmp_path)
    _run(runner.run(config))

    # FakeTrainingRunner writes run metadata
    assert tmp_path.exists()


def test_fake_runner_artifact_has_adapter_kind(tmp_path: Path) -> None:
    runner = FakeTrainingRunner()
    run = _run(runner.run(_config(tmp_path)))
    assert any(a.kind == ModelArtifactKind.ADAPTER for a in run.artifacts)


def test_fake_runner_dry_run_creates_dry_run_artifact(tmp_path: Path) -> None:
    runner = FakeTrainingRunner()
    config = TrainingConfig(
        base_model="tinyllama",
        output_dir=str(tmp_path),
        dry_run=True,
        parameters={"example_count": 5},
    )
    run = _run(runner.run(config))
    assert run.status == TrainingRunStatus.SUCCEEDED
    assert any("dry-run" in a.name for a in run.artifacts)


# ---------------------------------------------------------------------------
# JsonTrainingArtifactStore
# ---------------------------------------------------------------------------


def test_artifact_store_save_and_get(tmp_path: Path) -> None:
    store = JsonTrainingArtifactStore(tmp_path)
    runner = FakeTrainingRunner()
    run = _run(runner.run(_config(tmp_path / "out")))

    artifact = run.artifacts[0]
    _run(store.save(artifact))

    loaded = _run(store.get(str(artifact.id)))
    assert loaded is not None
    assert loaded.id == artifact.id
    assert loaded.kind == artifact.kind


def test_artifact_store_get_missing_returns_none(tmp_path: Path) -> None:
    store = JsonTrainingArtifactStore(tmp_path)
    result = _run(store.get("nonexistent-id"))
    assert result is None


def test_artifact_store_and_fake_runner_integration(tmp_path: Path) -> None:
    """FakeTrainingRunner → JsonTrainingArtifactStore round-trip."""
    runner = FakeTrainingRunner()
    store = JsonTrainingArtifactStore(tmp_path / "artifacts")
    config = _config(tmp_path / "run")

    run = _run(runner.run(config))
    for artifact in run.artifacts:
        _run(store.save(artifact))

    for artifact in run.artifacts:
        loaded = _run(store.get(str(artifact.id)))
        assert loaded is not None
        assert loaded.base_model == "tinyllama"
