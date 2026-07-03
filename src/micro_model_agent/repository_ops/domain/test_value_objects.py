"""Tests for repository_ops domain value objects."""

from __future__ import annotations

from micro_model_agent.repository_ops.domain.value_objects import (
    RepositoryProfile,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    SemanticSearchResult,
)


def test_repository_profile_stores_root_path() -> None:
    profile = RepositoryProfile(root_path="/repo", name="my-repo")
    assert profile.root_path == "/repo"
    assert profile.name == "my-repo"
    assert profile.default_branch is None


def test_repository_profile_with_branch() -> None:
    profile = RepositoryProfile(root_path="/repo", name="my-repo", default_branch="main")
    assert profile.default_branch == "main"


def test_retrieval_query_defaults() -> None:
    q = RetrievalQuery(query="find auth code")
    assert q.query == "find auth code"
    assert q.intent == "general"
    assert q.limit == 10
    assert q.filters == {}


def test_retrieval_query_custom() -> None:
    q = RetrievalQuery(query="find auth", intent="code", limit=5, filters={"lang": "py"})
    assert q.intent == "code"
    assert q.limit == 5
    assert q.filters == {"lang": "py"}


def test_retrieved_item_fields() -> None:
    item = RetrievedItem(
        source_type="code",
        title="auth.py",
        content="def authenticate(): ...",
        relevance_score=0.9,
    )
    assert item.source_type == "code"
    assert item.relevance_score == 0.9
    assert item.metadata == {}


def test_retrieved_item_with_metadata() -> None:
    item = RetrievedItem(
        source_type="doc",
        title="README",
        content="Overview",
        relevance_score=0.7,
        metadata={"path": "README.md"},
    )
    assert item.metadata["path"] == "README.md"


def test_retrieval_result_empty_results() -> None:
    result = RetrievalResult(query="test", intent="code")
    assert result.results == []


def test_retrieval_result_with_items() -> None:
    items = [
        RetrievedItem("code", "a.py", "content a", 0.9),
        RetrievedItem("code", "b.py", "content b", 0.7),
    ]
    result = RetrievalResult(query="test", intent="code", results=items)
    assert len(result.results) == 2
    assert result.results[0].title == "a.py"


def test_semantic_search_result_is_retrieval_result() -> None:
    """SemanticSearchResult is a subtype of RetrievalResult."""
    result = SemanticSearchResult(query="find auth", intent="semantic")
    assert isinstance(result, RetrievalResult)
    assert result.results == []


def test_value_objects_are_frozen() -> None:
    """Domain value objects must be immutable."""
    profile = RepositoryProfile(root_path="/repo", name="repo")
    try:
        profile.name = "other"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass
