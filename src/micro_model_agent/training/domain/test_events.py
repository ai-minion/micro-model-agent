"""Tests for training domain events and exceptions."""

from __future__ import annotations

from uuid import uuid4

import pytest

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
from micro_model_agent.training.domain.value_objects import ModelArtifactKind
from micro_model_agent.shared.domain.domain_event import DomainEvent


# ---------------------------------------------------------------------------
# Domain events
# ---------------------------------------------------------------------------


def test_training_job_created_fields() -> None:
    job_id = uuid4()
    event = TrainingJobCreated(job_id=job_id)
    assert event.job_id == job_id
    assert isinstance(event, DomainEvent)


def test_training_job_started_fields() -> None:
    job_id = uuid4()
    run_id = uuid4()
    event = TrainingJobStarted(job_id=job_id, run_id=run_id)
    assert event.run_id == run_id


def test_training_job_completed_fields() -> None:
    artifact_id = uuid4()
    event = TrainingJobCompleted(job_id=uuid4(), artifact_id=artifact_id)
    assert event.artifact_id == artifact_id


def test_training_job_failed_fields() -> None:
    event = TrainingJobFailed(job_id=uuid4(), reason="GPU OOM")
    assert event.reason == "GPU OOM"


def test_artifact_produced_fields() -> None:
    artifact_id = uuid4()
    event = ArtifactProduced(
        job_id=uuid4(),
        artifact_id=artifact_id,
        kind=ModelArtifactKind.ADAPTER,
    )
    assert event.artifact_id == artifact_id
    assert event.kind == ModelArtifactKind.ADAPTER


def test_all_training_events_are_domain_events() -> None:
    job_id = uuid4()
    events = [
        TrainingJobCreated(job_id=job_id),
        TrainingJobStarted(job_id=job_id, run_id=uuid4()),
        TrainingJobCompleted(job_id=job_id, artifact_id=uuid4()),
        TrainingJobFailed(job_id=job_id, reason="err"),
        ArtifactProduced(job_id=job_id, artifact_id=uuid4(), kind=ModelArtifactKind.ADAPTER),
    ]
    for event in events:
        assert isinstance(event, DomainEvent)


def test_events_get_unique_ids() -> None:
    e1 = TrainingJobCreated(job_id=uuid4())
    e2 = TrainingJobCreated(job_id=uuid4())
    assert e1.event_id != e2.event_id


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


def test_invalid_config_error_is_domain_exception() -> None:
    from micro_model_agent.shared.domain.exceptions import DomainException
    exc = InvalidConfigError("base_model must not be blank")
    assert isinstance(exc, DomainException)
    assert "base_model" in str(exc)


def test_training_already_started_error_includes_job_id() -> None:
    job_id = uuid4()
    exc = TrainingAlreadyStartedError(job_id)
    assert str(job_id) in str(exc)
    assert exc.job_id == job_id


def test_training_already_started_is_domain_exception() -> None:
    from micro_model_agent.shared.domain.exceptions import DomainException
    with pytest.raises(DomainException):
        raise TrainingAlreadyStartedError(uuid4())
