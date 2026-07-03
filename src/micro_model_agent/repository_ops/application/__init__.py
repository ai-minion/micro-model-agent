"""Repository Ops context application ports."""

from __future__ import annotations

from typing import Protocol

from micro_model_agent.repository_ops.domain.value_objects import (
    RetrievalQuery,
    SemanticSearchResult,
)


class SemanticRetriever(Protocol):
    """Search service for model-facing repository or documentation context."""

    async def search(self, query: RetrievalQuery) -> SemanticSearchResult:
        """Run structured semantic retrieval."""
