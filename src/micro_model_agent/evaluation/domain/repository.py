"""EvaluationReportRepository — evaluation context repository protocol."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from micro_model_agent.evaluation.domain.aggregate import EvaluationReport


class EvaluationReportRepository(Protocol):
    """Collection-semantics repository for EvaluationReport aggregates."""

    async def add(self, report: EvaluationReport) -> None:
        """Persist a newly created EvaluationReport."""

    async def save(self, report: EvaluationReport) -> None:
        """Update an already-persisted EvaluationReport."""

    async def get(self, id: UUID) -> EvaluationReport | None:
        """Load an EvaluationReport by identity; return None if absent."""

    async def find_by_run_id(self, run_id: str) -> list[EvaluationReport]:
        """Return all reports associated with the given training run id."""
