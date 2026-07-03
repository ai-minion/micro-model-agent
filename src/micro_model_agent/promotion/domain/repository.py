"""ModelRegistryRepository — promotion context repository protocol."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from micro_model_agent.promotion.domain.aggregate import ModelRegistry


class ModelRegistryRepository(Protocol):
    """Collection-semantics repository for ModelRegistry aggregates.

    There is typically a single ModelRegistry per deployment.  ``get_or_create``
    is the primary entry point: it loads the existing registry or creates a
    fresh one if none exists.
    """

    async def get_or_create(self) -> ModelRegistry:
        """Return the existing ModelRegistry or create and persist a new one."""

    async def save(self, registry: ModelRegistry) -> None:
        """Update the persisted ModelRegistry."""

    async def get(self, id: UUID) -> ModelRegistry | None:
        """Load a ModelRegistry by identity; return None if absent."""
