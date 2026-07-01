"""Tests for the repo.semantic_search tool."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.infrastructure.repositories.local_index import LocalLexicalIndexWriter
from micro_model_agent.infrastructure.tools.contracts import SemanticSearchRequest
from micro_model_agent.infrastructure.tools.repo_semantic_search import RepoSemanticSearchTool


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def _write_sample_repo(root: Path) -> None:
    _write_text(
        root / "docs" / "architecture.md",
        "# Architecture\n\n"
        "The domain layer must remain framework independent. "
        "Repository tools enforce path safety before reading files.\n",
    )
    _write_text(
        root / "docs" / "usage.md",
        "# Usage\n\nRun synthetic training with dataset validation.\n",
    )
    _write_text(
        root / "src" / "service.py",
        "class DatasetService:\n"
        "    def validate_dataset(self) -> bool:\n"
        "        return True\n",
    )
    _write_text(
        root / "src" / "test_service.py",
        "def test_validate_dataset() -> None:\n"
        "    assert True\n",
    )
    _write_text(
        root / "examples" / "synthetic-data" / "tool-use.seed.jsonl",
        '{"kind":"tool_use","target":{"tool_name":"repo.search"}}\n',
    )


def test_semantic_search_retrieves_architecture_docs_first_for_architecture_intent(
    tmp_path: Path,
) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(
            query="domain framework independent path safety",
            intent="architecture_rules",
            limit=3,
        )
    )

    assert result.results
    assert result.results[0].source_type == "architecture_doc"
    assert result.results[0].metadata["path"] == "docs/architecture.md"
    assert "framework independent" in result.results[0].content


def test_semantic_search_filters_by_source_type(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(
            query="repo.search tool use",
            intent="synthetic_data",
            filters={"source_type": "synthetic_data"},
        )
    )

    assert [item.source_type for item in result.results] == ["synthetic_data"]
    assert result.results[0].metadata["path"] == "examples/synthetic-data/tool-use.seed.jsonl"


def test_semantic_search_filters_by_extension(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(
            query="validate_dataset",
            intent="source_code",
            filters={"extensions": ["py"]},
        )
    )

    assert result.results
    assert {item.metadata["extension"] for item in result.results} == {".py"}


def test_semantic_search_applies_filters_before_index_limit(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "src" / "service.py",
        "workflow workflow workflow workflow workflow workflow\n",
    )
    _write_text(
        tmp_path / "docs" / "workflow.md",
        "# Workflow\n\nworkflow\n",
    )
    LocalLexicalIndexWriter(tmp_path).write()

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(
            query="workflow",
            limit=1,
            filters={"source_type": "documentation"},
        )
    )

    assert [item.metadata["path"] for item in result.results] == ["docs/workflow.md"]
    assert result.results[0].metadata["retrieval_backend"] == "local_lexical_index"


def test_semantic_search_marks_test_metadata(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(query="validate_dataset", intent="tests", limit=1)
    )

    assert result.results[0].metadata["is_test"] is True
    assert result.results[0].metadata["path"] == "src/test_service.py"


def test_semantic_search_enforces_limit(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(query="dataset", intent="general", limit=1)
    )

    assert len(result.results) == 1


def test_semantic_search_returns_empty_results_for_unsafe_glob(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(
            query="domain",
            filters={"path_glob": "../*.md"},
        )
    )

    assert result.results == []


def test_semantic_search_uses_local_index_when_available(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    LocalLexicalIndexWriter(tmp_path).write()

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(query="framework independent", intent="architecture_rules")
    )

    assert result.results
    assert result.results[0].metadata["path"] == "docs/architecture.md"
    assert result.results[0].metadata["retrieval_backend"] == "local_lexical_index"
    assert result.results[0].metadata["indexed_score"] > 0


def test_semantic_search_includes_local_index_freshness_metadata(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    LocalLexicalIndexWriter(tmp_path).write()
    _write_text(
        tmp_path / "docs" / "architecture.md",
        "# Architecture\n\nThe domain layer changed.\n",
    )
    (tmp_path / "docs" / "new.md").write_text("framework independent\n", encoding="utf-8")

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(query="framework independent", intent="architecture_rules")
    )

    assert result.results
    assert result.results[0].metadata["retrieval_backend"] == "local_lexical_index"
    assert result.results[0].metadata["index_status"] == {
        "is_stale": True,
        "indexed_file_count": 5,
        "changed_file_count": 1,
        "missing_file_count": 0,
        "extra_file_count": 1,
    }


def test_semantic_search_includes_indexed_code_metadata(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    LocalLexicalIndexWriter(tmp_path).write()

    result = RepoSemanticSearchTool(tmp_path).run(
        SemanticSearchRequest(query="DatasetService validate_dataset", intent="code")
    )

    assert result.results[0].metadata["path"] == "src/service.py"
    assert result.results[0].metadata["retrieval_backend"] == "local_lexical_index"
    assert result.results[0].metadata["symbols"] == [
        {"kind": "ClassDef", "line_number": 1, "name": "DatasetService"},
        {"kind": "FunctionDef", "line_number": 2, "name": "validate_dataset"},
    ]
