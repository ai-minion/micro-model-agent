"""Tests for local retrieval port adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.repository_ops.domain.value_objects import RetrievalQuery
from micro_model_agent.repository_ops.infrastructure.local_index import LocalLexicalIndexWriter
from micro_model_agent.repository_ops.infrastructure.local_retrieval import LocalSemanticRetriever


def test_local_semantic_retriever_returns_domain_results(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text(
        "# Architecture\n\nThe domain layer owns retrieval contracts.\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        LocalSemanticRetriever(tmp_path).search(
            RetrievalQuery(query="domain retrieval contracts", intent="architecture_rules")
        )
    )

    assert result.query == "domain retrieval contracts"
    assert result.intent == "architecture_rules"
    assert result.results[0].source_type == "architecture_doc"
    assert result.results[0].metadata["path"] == "docs/architecture.md"
    assert result.results[0].metadata["retrieval_backend"] == "direct_scan"


def test_local_semantic_retriever_uses_persisted_index(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "def repair_patch() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    LocalLexicalIndexWriter(tmp_path).write()

    result = asyncio.run(
        LocalSemanticRetriever(tmp_path).search(RetrievalQuery(query="repair patch", intent="code"))
    )

    assert result.results[0].metadata["path"] == "src/service.py"
    assert result.results[0].metadata["retrieval_backend"] == "local_lexical_index"
    assert result.results[0].metadata["indexed_score"] > 0
