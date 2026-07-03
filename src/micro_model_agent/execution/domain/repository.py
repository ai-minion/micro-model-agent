"""WorkflowRepository — execution context repository protocol."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from micro_model_agent.execution.domain.aggregate import WorkflowExecution
from micro_model_agent.execution.domain.value_objects import WorkflowStatus


class WorkflowRepository(Protocol):
    """Collection-semantics repository for WorkflowExecution aggregates.

    Each method operates on the aggregate root.  The repository is responsible
    for serializing and deserializing domain events and state.
    """

    async def add(self, execution: WorkflowExecution) -> None:
        """Persist a newly created WorkflowExecution for the first time."""

    async def save(self, execution: WorkflowExecution) -> None:
        """Update an already-persisted WorkflowExecution."""

    async def get(self, id: UUID) -> WorkflowExecution | None:
        """Load a WorkflowExecution by its identity; return None if absent."""

    async def find_by_status(
        self, status: WorkflowStatus
    ) -> list[WorkflowExecution]:
        """Return all executions that are currently in the given status."""
