"""Domain events for the training bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from micro_model_agent.shared.domain.domain_event import DomainEvent
from micro_model_agent.training.domain.value_objects import ModelArtifactKind


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingJobCreated(DomainEvent):
    """Raised when a TrainingJob aggregate is first constructed."""

    job_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingJobStarted(DomainEvent):
    """Raised when a TrainingJob transitions to RUNNING."""

    job_id: UUID
    run_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingJobCompleted(DomainEvent):
    """Raised when a TrainingJob finishes successfully."""

    job_id: UUID
    artifact_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingJobFailed(DomainEvent):
    """Raised when a TrainingJob transitions to FAILED."""

    job_id: UUID
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactProduced(DomainEvent):
    """Raised alongside TrainingJobCompleted; carries artifact metadata.

    This event is the Published Language token consumed by the evaluation
    context to trigger an automated evaluation run.
    """

    job_id: UUID
    artifact_id: UUID
    kind: ModelArtifactKind
