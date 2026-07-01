"""Local structured semantic search implementation.

This is a lightweight lexical retriever: it classifies repository files, scores
them by query term overlap, and returns short excerpts for model prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from micro_model_agent.infrastructure.repositories.local_index import LocalLexicalIndexReader
from micro_model_agent.infrastructure.repositories.paths import (
    RepositoryPathError,
    RepositoryRoot,
    looks_binary,
)
from micro_model_agent.infrastructure.tools.contracts import (
    RetrievedItemContract,
    SemanticSearchRequest,
    SemanticSearchResultContract,
)

SOURCE_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
}
DOCUMENTATION_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
DATA_EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml", ".toml"}


@dataclass(frozen=True, slots=True)
class CandidateDocument:
    """Internal searchable representation of one repository file."""

    path: str
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any]


class RepoSemanticSearchTool:
    """Lexical structured retrieval over local repository files."""

    def __init__(self, repository_root: str | Path, max_file_bytes: int = 1_000_000) -> None:
        self.repository = RepositoryRoot(repository_root)
        self.max_file_bytes = max_file_bytes

    def run(self, request: SemanticSearchRequest) -> SemanticSearchResultContract:
        filters = request.filters
        # Combine the user query and intent so intent words can influence scores.
        query_terms = self._tokenize(f"{request.query} {request.intent}")
        source_types = self._source_type_filter(filters.get("source_type"))
        extensions = self._extensions_filter(filters.get("extensions"))
        path_glob = str(filters.get("path_glob", "**/*"))

        try:
            candidates = self._candidate_documents(
                path_glob,
                source_types,
                extensions,
                query_terms=query_terms,
                limit=request.limit,
            )
        except RepositoryPathError:
            return SemanticSearchResultContract(
                query=request.query,
                intent=request.intent,
                results=[],
            )

        scored = [
            (self._score(candidate, query_terms, request.intent), candidate)
            for candidate in candidates
        ]
        # Drop zero-score candidates and sort highest score first, then path for
        # deterministic ties.
        scored = [(score, candidate) for score, candidate in scored if score > 0.0]
        scored.sort(key=lambda item: (-item[0], item[1].path))

        results = [
            RetrievedItemContract(
                source_type=candidate.source_type,
                title=candidate.title,
                content=self._excerpt(candidate.content, query_terms),
                relevance_score=score,
                metadata=candidate.metadata,
            )
            for score, candidate in scored[: request.limit]
        ]
        return SemanticSearchResultContract(
            query=request.query,
            intent=request.intent,
            results=results,
        )

    def _candidate_documents(
        self,
        path_glob: str,
        source_types: set[str] | None,
        extensions: set[str] | None,
        *,
        query_terms: set[str],
        limit: int,
    ) -> list[CandidateDocument]:
        """Load candidate files and attach source-type metadata."""

        indexed_documents = self._indexed_candidate_documents(
            path_glob,
            source_types,
            extensions,
            query_terms=query_terms,
            limit=limit,
        )
        if indexed_documents is not None:
            return indexed_documents

        documents: list[CandidateDocument] = []
        for path in self.repository.iter_files(path_glob):
            document = self._candidate_document(
                path,
                source_types,
                extensions,
                retrieval_backend="direct_scan",
            )
            if document is not None:
                documents.append(document)
        return documents

    def _indexed_candidate_documents(
        self,
        path_glob: str,
        source_types: set[str] | None,
        extensions: set[str] | None,
        *,
        query_terms: set[str],
        limit: int,
    ) -> list[CandidateDocument] | None:
        """Load candidates from the persisted lexical index when available."""

        if path_glob != "**/*":
            return None

        index = LocalLexicalIndexReader(self.repository.root)
        if not index.exists():
            return None

        documents: list[CandidateDocument] = []
        freshness = index.freshness()
        index_status = freshness.as_metadata() if freshness is not None else None
        # Pull extra candidates because stale paths can drop some indexed
        # results before final scoring.
        for candidate in index.search_paths(
            query_terms,
            limit=max(limit * 10, limit),
            source_types=source_types,
            extensions=extensions,
        ):
            try:
                path = self.repository.resolve_file(candidate.path)
            except RepositoryPathError:
                continue
            document = self._candidate_document(
                path,
                source_types,
                extensions,
                retrieval_backend="local_lexical_index",
                indexed_score=candidate.score,
                indexed_metadata=index.file_metadata(candidate.path),
                index_status=index_status,
            )
            if document is not None:
                documents.append(document)
        if not documents:
            return None
        return documents

    def _candidate_document(
        self,
        path: Path,
        source_types: set[str] | None,
        extensions: set[str] | None,
        *,
        retrieval_backend: str,
        indexed_score: float | None = None,
        indexed_metadata: dict[str, Any] | None = None,
        index_status: dict[str, int | bool] | None = None,
    ) -> CandidateDocument | None:
        """Load one candidate document if it passes filters and text checks."""

        relative_path = self.repository.relative_path(path)
        source_type = self._source_type(relative_path)
        if source_types is not None and source_type not in source_types:
            return None
        if extensions is not None and path.suffix.casefold() not in extensions:
            return None

        content = self._read_text(path)
        if content is None:
            return None

        metadata: dict[str, Any] = {
            "path": relative_path,
            "extension": path.suffix.casefold(),
            "is_test": self._is_test_path(relative_path),
            "retrieval_backend": retrieval_backend,
        }
        if indexed_score is not None:
            metadata["indexed_score"] = indexed_score
        if index_status is not None:
            metadata["index_status"] = index_status
        if indexed_metadata is not None:
            metadata["symbols"] = indexed_metadata.get("symbols", [])
            metadata["imports"] = indexed_metadata.get("imports", [])
            metadata["sha256"] = indexed_metadata.get("sha256")
            metadata["size_bytes"] = indexed_metadata.get("size_bytes")
        return CandidateDocument(
            path=relative_path,
            source_type=source_type,
            title=self._title(relative_path, content),
            content=content,
            metadata=metadata,
        )

    def _read_text(self, path: Path) -> str | None:
        """Read a candidate file, skipping large or binary files."""

        try:
            if path.stat().st_size > self.max_file_bytes:
                return None
            raw = path.read_bytes()
        except OSError:
            return None
        if looks_binary(raw[:4096]):
            return None
        return raw.decode("utf-8", errors="replace")

    def _source_type(self, relative_path: str) -> str:
        """Classify a file so callers can filter and score by source type."""

        path = Path(relative_path)
        normalized = relative_path.casefold()
        suffix = path.suffix.casefold()

        if normalized.startswith("examples/synthetic-data/") or suffix == ".jsonl":
            return "synthetic_data"
        if (
            "architecture" in normalized
            or "/adr" in normalized
            or normalized.startswith("docs/adr")
        ):
            return "architecture_doc"
        if normalized.startswith("docs/") or suffix in DOCUMENTATION_EXTENSIONS:
            return "documentation"
        if suffix in SOURCE_CODE_EXTENSIONS:
            return "source_code"
        if suffix in DATA_EXTENSIONS:
            return "structured_data"
        return "repository_file"

    def _title(self, relative_path: str, content: str) -> str:
        """Use the first Markdown heading as a title when available."""

        suffix = Path(relative_path).suffix.casefold()
        if suffix in DOCUMENTATION_EXTENSIONS:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    return stripped.lstrip("#").strip() or relative_path
        return relative_path

    def _score(self, candidate: CandidateDocument, query_terms: set[str], intent: str) -> float:
        """Compute a simple relevance score between 0.0 and 1.0."""

        if not query_terms:
            return 0.0

        searchable = " ".join(
            [
                candidate.path,
                candidate.title,
                candidate.source_type,
                candidate.content,
            ]
        )
        candidate_terms = self._tokenize(searchable)
        matched_terms = query_terms.intersection(candidate_terms)
        if not matched_terms:
            return 0.0

        score = len(matched_terms) / len(query_terms)
        score += self._intent_bonus(candidate, intent)
        if matched_terms.intersection(self._tokenize(candidate.title)):
            score += 0.1
        if matched_terms.intersection(self._tokenize(candidate.path)):
            score += 0.05
        return min(score, 1.0)

    def _intent_bonus(self, candidate: CandidateDocument, intent: str) -> float:
        """Add a small score boost when the source type matches the intent."""

        normalized = intent.casefold()
        if normalized in {"architecture", "architecture_rules"}:
            return 0.5 if candidate.source_type == "architecture_doc" else 0.0
        if normalized in {"docs", "documentation"}:
            return 0.2 if candidate.source_type in {"documentation", "architecture_doc"} else 0.0
        if normalized in {"code", "source", "source_code"}:
            return 0.2 if candidate.source_type == "source_code" else 0.0
        if normalized in {"tests", "test_examples"}:
            return 0.2 if candidate.metadata.get("is_test") else 0.0
        if normalized in {"synthetic", "synthetic_data", "tool_schema"}:
            return 0.2 if candidate.source_type == "synthetic_data" else 0.0
        return 0.0

    def _excerpt(self, content: str, query_terms: set[str], max_chars: int = 1_200) -> str:
        """Return a short snippet near the first matched query term."""

        if len(content) <= max_chars:
            return content

        lower_content = content.casefold()
        first_match = min(
            (
                index
                for term in query_terms
                if (index := lower_content.find(term)) >= 0
            ),
            default=0,
        )
        start = max(first_match - max_chars // 4, 0)
        end = min(start + max_chars, len(content))
        return content[start:end]

    def _tokenize(self, value: str) -> set[str]:
        """Split text into lowercase search terms with small plural handling."""

        terms: set[str] = set()
        for raw_term in re.findall(r"[A-Za-z0-9_]+", value):
            term = raw_term.casefold()
            if len(term) > 1:
                terms.add(term)
                if term.endswith("s") and len(term) > 3:
                    terms.add(term[:-1])
            for part in term.split("_"):
                if len(part) > 1:
                    terms.add(part)
                    if part.endswith("s") and len(part) > 3:
                        terms.add(part[:-1])
        return terms

    def _source_type_filter(self, value: Any) -> set[str] | None:
        """Normalize a source_type filter from string/list/None."""

        if value is None:
            return None
        if isinstance(value, str):
            return {value}
        if isinstance(value, list):
            return {str(item) for item in value}
        return None

    def _extensions_filter(self, value: Any) -> set[str] | None:
        """Normalize an extension filter and ensure each extension starts with a dot."""

        if value is None:
            return None
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list):
            return None
        return {
            extension.casefold()
            if str(extension).startswith(".")
            else f".{str(extension).casefold()}"
            for extension in values
        }

    def _is_test_path(self, relative_path: str) -> bool:
        """Return true for common test-file path conventions."""

        path = relative_path.casefold()
        return "/test_" in f"/{path}" or path.endswith("_test.py") or "/tests/" in f"/{path}/"
