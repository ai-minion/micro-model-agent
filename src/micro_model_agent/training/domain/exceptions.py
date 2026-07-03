"""Domain exceptions for the training bounded context."""

from __future__ import annotations

from uuid import UUID

from micro_model_agent.shared.domain.exceptions import DomainException


class InvalidConfigError(DomainException):
    """Raised when a TrainingConfig violates invariants."""


class TrainingAlreadyStartedError(DomainException):
    """Raised when start() is called on an already-running TrainingJob."""

    def __init__(self, job_id: UUID) -> None:
        super().__init__(f"Training job {job_id} has already been started")
        self.job_id = job_id
