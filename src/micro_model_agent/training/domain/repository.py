"""TrainingJobRepository — training context repository protocol."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from micro_model_agent.training.domain.aggregate import TrainingJob
from micro_model_agent.training.domain.value_objects import TrainingRunStatus


class TrainingJobRepository(Protocol):
    """Collection-semantics repository for TrainingJob aggregates."""

    async def add(self, job: TrainingJob) -> None:
        """Persist a newly created TrainingJob."""

    async def save(self, job: TrainingJob) -> None:
        """Update an already-persisted TrainingJob."""

    async def get(self, id: UUID) -> TrainingJob | None:
        """Load a TrainingJob by identity; return None if absent."""

    async def find_by_status(self, status: TrainingRunStatus) -> list[TrainingJob]:
        """Return all jobs currently in the given status."""
