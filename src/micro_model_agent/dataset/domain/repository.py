"""DatasetRepository — dataset context repository protocol."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from micro_model_agent.dataset.domain.aggregate import Dataset


class DatasetRepository(Protocol):
    """Collection-semantics repository for Dataset aggregates."""

    async def add(self, dataset: Dataset) -> None:
        """Persist a newly created Dataset for the first time."""

    async def save(self, dataset: Dataset) -> None:
        """Update an already-persisted Dataset."""

    async def get(self, id: UUID) -> Dataset | None:
        """Load a Dataset by identity; return None if absent."""

    async def find_by_name(self, name: str) -> Dataset | None:
        """Return the most recently saved Dataset with the given name."""
