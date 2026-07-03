"""Domain invariant and event tests for the training bounded context."""

from __future__ import annotations

from pathlib import Path

import pytest

from micro_model_agent.training.domain.aggregate import TrainingJob
from micro_model_agent.training.domain.events import (
    ArtifactProduced,
    TrainingJobCompleted,
    TrainingJobCreated,
    TrainingJobFailed,
    TrainingJobStarted,
)
from micro_model_agent.training.domain.exceptions import (
    InvalidConfigError,
    TrainingAlreadyStartedError,
)
from micro_model_agent.training.domain.services import TrainingConfigValidationService
from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from uuid import uuid4
from datetime import datetime, UTC


def _config(output_dir: str = "/tmp/run") -> TrainingConfig:
    return TrainingConfig(base_model="tinyllama", output_dir=output_dir)


def _run(status: TrainingRunStatus = TrainingRunStatus.SUCCEEDED) -> TrainingRun:
    return TrainingRun(
        kind=TrainingRunKind.SYNTHETIC,
        config=_config(),
        status=status,
    )


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        name="adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="/tmp/adapter",
        base_model="tinyllama",
    )


# ---------------------------------------------------------------------------
# TrainingJob aggregate
# ---------------------------------------------------------------------------


def test_new_job_emits_created_event() -> None:
    job = TrainingJob(config=_config())
    events = job.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], TrainingJobCreated)
    assert events[0].job_id == job.id


def test_start_emits_started_event() -> None:
    job = TrainingJob(config=_config())
    job.pull_events()  # drain created

    run = _run(TrainingRunStatus.RUNNING)
    job.start(run)

    events = job.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], TrainingJobStarted)
    assert events[0].job_id == job.id
    assert events[0].run_id == run.id


def test_start_twice_raises() -> None:
    job = TrainingJob(config=_config())
    job.start(_run(TrainingRunStatus.RUNNING))
    with pytest.raises(TrainingAlreadyStartedError):
        job.start(_run(TrainingRunStatus.RUNNING))


def test_complete_emits_completed_and_artifact_produced() -> None:
    job = TrainingJob(config=_config())
    run = _run(TrainingRunStatus.RUNNING)
    job.start(run)
    job.pull_events()  # drain created + started

    artifact = _artifact()
    final_run = _run(TrainingRunStatus.SUCCEEDED)
    job.complete(final_run, artifact)

    events = job.pull_events()
    assert len(events) == 2
    kinds = {type(e) for e in events}
    assert TrainingJobCompleted in kinds
    assert ArtifactProduced in kinds

    completed = next(e for e in events if isinstance(e, TrainingJobCompleted))
    assert completed.artifact_id == artifact.id

    produced = next(e for e in events if isinstance(e, ArtifactProduced))
    assert produced.artifact_id == artifact.id
    assert produced.kind == ModelArtifactKind.ADAPTER


def test_complete_before_start_raises() -> None:
    job = TrainingJob(config=_config())
    with pytest.raises(Exception):
        job.complete(_run(), _artifact())


def test_fail_emits_failed_event() -> None:
    job = TrainingJob(config=_config())
    job.pull_events()

    job.fail("GPU OOM")

    events = job.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], TrainingJobFailed)
    assert events[0].reason == "GPU OOM"


def test_artifacts_property_reflects_completion() -> None:
    job = TrainingJob(config=_config())
    job.start(_run(TrainingRunStatus.RUNNING))
    artifact = _artifact()
    job.complete(_run(), artifact)

    assert len(job.artifacts) == 1
    assert job.artifacts[0].id == artifact.id


def test_status_derived_from_run() -> None:
    job = TrainingJob(config=_config())
    assert job.status == TrainingRunStatus.PENDING

    job.start(_run(TrainingRunStatus.RUNNING))
    assert job.status == TrainingRunStatus.RUNNING


def test_pull_events_clears_list() -> None:
    job = TrainingJob(config=_config())
    first = job.pull_events()
    second = job.pull_events()
    assert len(first) == 1
    assert len(second) == 0


# ---------------------------------------------------------------------------
# TrainingConfigValidationService
# ---------------------------------------------------------------------------


def test_valid_config_passes() -> None:
    svc = TrainingConfigValidationService()
    svc.validate(_config())  # no exception


def test_blank_base_model_rejected() -> None:
    svc = TrainingConfigValidationService()
    with pytest.raises(InvalidConfigError, match="base_model"):
        svc.validate(TrainingConfig(base_model="", output_dir="/tmp"))


def test_negative_max_steps_rejected() -> None:
    svc = TrainingConfigValidationService()
    with pytest.raises(InvalidConfigError, match="max_steps"):
        svc.validate(TrainingConfig(base_model="m", output_dir="/tmp", max_steps=-1))
