"""Repository Ops context value objects.

These are the canonical definitions for repository-retrieval types.
Canonical home: this module. ``domain/__init__.py`` re-exports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RepositoryProfile:
    """Basic information about the repository an agent is working in."""

    root_path: str
    name: str
    default_branch: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Question sent to a retriever when the agent needs repository context."""

    query: str
    intent: str = "general"
    limit: int = 10
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """One item returned from retrieval, with a score and extra metadata."""

    source_type: str
    title: str
    content: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Collection of retrieved items for a single query."""

    query: str
    intent: str
    results: list[RetrievedItem] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SemanticSearchResult(RetrievalResult):
    """Structured semantic retrieval result."""
