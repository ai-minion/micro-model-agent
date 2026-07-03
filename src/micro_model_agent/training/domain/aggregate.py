"""TrainingJob — training context aggregate root."""

from __future__ import annotations

from uuid import UUID

from micro_model_agent.shared.domain.entity import Entity
from micro_model_agent.shared.domain.exceptions import DomainException
from micro_model_agent.training.domain.events import (
    ArtifactProduced,
    TrainingJobCompleted,
    TrainingJobCreated,
    TrainingJobFailed,
    TrainingJobStarted,
)
from micro_model_agent.training.domain.exceptions import (
    TrainingAlreadyStartedError,
)
from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    TrainingConfig,
    TrainingRun,
    TrainingRunStatus,
)


class _InvalidTransition(DomainException):
    """Internal: invalid state transition within TrainingJob."""


class TrainingJob(Entity):
    """Aggregate root for a single local fine-tuning job.

    Invariants enforced:
    - ``start()`` cannot be called more than once.
    - ``complete()`` requires a prior ``start()``.
    """

    def __init__(self, config: TrainingConfig, id: UUID | None = None) -> None:
        super().__init__(id)
        self.config = config
        self._run: TrainingRun | None = None
        self._artifacts: list[ModelArtifact] = []
        self._events.append(TrainingJobCreated(job_id=self.id))

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def start(self, run: TrainingRun) -> None:
        """Record that execution has begun; raises if already started."""

        if self._run is not None:
            raise TrainingAlreadyStartedError(self.id)
        self._run = run
        self._events.append(
            TrainingJobStarted(job_id=self.id, run_id=run.id)
        )

    def complete(self, run: TrainingRun, artifact: ModelArtifact) -> None:
        """Transition to SUCCEEDED, recording the final run state and artifact."""

        if self._run is None:
            raise _InvalidTransition("Cannot complete a job that has not been started")
        self._run = run
        self._artifacts.append(artifact)
        self._events.append(
            TrainingJobCompleted(job_id=self.id, artifact_id=artifact.id)
        )
        self._events.append(
            ArtifactProduced(
                job_id=self.id,
                artifact_id=artifact.id,
                kind=artifact.kind,
            )
        )

    def fail(self, reason: str) -> None:
        """Record a failure regardless of the current run state."""

        self._events.append(
            TrainingJobFailed(job_id=self.id, reason=reason)
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def run(self) -> TrainingRun | None:
        """The most recently recorded TrainingRun, or None if not started."""

        return self._run

    @property
    def artifacts(self) -> tuple[ModelArtifact, ...]:
        """All artifacts produced by this job."""

        return tuple(self._artifacts)

    @property
    def status(self) -> TrainingRunStatus:
        """Derive lifecycle status from the current run record."""

        if self._run is None:
            return TrainingRunStatus.PENDING
        return self._run.status
