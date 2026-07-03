"""JsonlWorkflowRepository — execution context DDD repository.

Implements ``WorkflowRepository`` by delegating to ``JsonlTraceStore`` for
persistence.  The aggregate's ``to_snapshot()`` and ``from_snapshot()`` methods
bridge the gap between the mutable aggregate and the immutable trace record.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from micro_model_agent.execution.domain.value_objects import WorkflowStatus
from micro_model_agent.execution.domain.aggregate import WorkflowExecution
from micro_model_agent.execution.infrastructure.trace_store import JsonlTraceStore


class JsonlWorkflowRepository:
    """DDD-style WorkflowRepository backed by ``JsonlTraceStore``.

    The store is append-only; ``add`` and ``save`` both delegate to
    ``JsonlTraceStore.save`` which writes the latest snapshot to both the
    flat JSONL log and the per-trace JSON artifact file.
    """

    def __init__(self, path: str | Path) -> None:
        self._store = JsonlTraceStore(path)

    async def add(self, execution: WorkflowExecution) -> None:
        """Persist a newly created WorkflowExecution."""

        await self._store.save(execution.to_snapshot())

    async def save(self, execution: WorkflowExecution) -> None:
        """Persist an updated WorkflowExecution snapshot."""

        await self._store.save(execution.to_snapshot())

    async def get(self, id: UUID) -> WorkflowExecution | None:
        """Load a WorkflowExecution by identity."""

        trace = await self._store.get(str(id))
        if trace is None:
            return None
        return WorkflowExecution.from_snapshot(trace)

    async def find_by_status(self, status: WorkflowStatus) -> list[WorkflowExecution]:
        """Return all executions whose current status matches."""

        traces = await self._store.list()
        return [
            WorkflowExecution.from_snapshot(t)
            for t in traces
            if t.status == status
        ]
