"""Local retrieval adapters for application ports."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.repository_ops.domain.value_objects import (
    RetrievalQuery,
    RetrievedItem,
    SemanticSearchResult,
)
from micro_model_agent.repository_ops.infrastructure.contracts import SemanticSearchRequest
from micro_model_agent.repository_ops.infrastructure.repo_semantic_search import RepoSemanticSearchTool


class LocalSemanticRetriever:
    """SemanticRetriever adapter backed by local lexical repository search."""

    def __init__(self, repository_root: str | Path, max_file_bytes: int = 1_000_000) -> None:
        self.tool = RepoSemanticSearchTool(repository_root, max_file_bytes=max_file_bytes)

    async def search(self, query: RetrievalQuery) -> SemanticSearchResult:
        """Run local semantic retrieval and return domain retrieval contracts."""

        result = self.tool.run(
            SemanticSearchRequest(
                query=query.query,
                intent=query.intent,
                limit=query.limit,
                filters=query.filters,
            )
        )
        return SemanticSearchResult(
            query=result.query,
            intent=result.intent,
            results=[
                RetrievedItem(
                    source_type=item.source_type,
                    title=item.title,
                    content=item.content,
                    relevance_score=item.relevance_score,
                    metadata=item.metadata,
                )
                for item in result.results
            ],
        )
