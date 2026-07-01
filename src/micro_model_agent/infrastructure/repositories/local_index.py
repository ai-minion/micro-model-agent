"""Local lexical repository index writer.

The index is intentionally simple JSON so future retrieval adapters can consume
it without committing this project to a specific vector database.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from micro_model_agent.infrastructure.repositories.metadata import METADATA_DIR_NAME
from micro_model_agent.infrastructure.repositories.paths import RepositoryRoot, looks_binary

INDEX_SCHEMA_VERSION = 1
DEFAULT_INDEX_RELATIVE_PATH = Path(METADATA_DIR_NAME) / "index" / "lexical-index.json"
DEFAULT_MAX_FILE_BYTES = 1_000_000
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
class IndexedSymbol:
    """One source symbol discovered while indexing."""

    name: str
    kind: str
    line_number: int


@dataclass(frozen=True, slots=True)
class IndexedFile:
    """Metadata and token counts for one indexed repository file."""

    path: str
    extension: str
    source_type: str
    is_test: bool
    size_bytes: int
    sha256: str
    term_count: int
    unique_term_count: int
    terms: dict[str, int]
    symbols: list[IndexedSymbol]
    imports: list[str]


@dataclass(frozen=True, slots=True)
class LocalIndexResult:
    """Summary of a local repository indexing run."""

    repository_root: str
    index_path: str
    indexed_file_count: int
    skipped_large_count: int
    skipped_binary_count: int
    total_bytes: int
    unique_term_count: int
    source_type_counts: dict[str, int]
    extension_counts: dict[str, int]
    test_file_count: int
    symbol_count: int
    import_count: int


@dataclass(frozen=True, slots=True)
class LocalIndexFreshness:
    """Current repository drift from the persisted lexical index."""

    indexed_file_count: int
    changed_file_count: int
    missing_file_count: int
    extra_file_count: int

    @property
    def is_stale(self) -> bool:
        """Return true when current repository files differ from the index."""

        return (
            self.changed_file_count > 0
            or self.missing_file_count > 0
            or self.extra_file_count > 0
        )

    def as_metadata(self) -> dict[str, int | bool]:
        """Return a JSON-ready freshness summary for tool metadata."""

        return {
            "is_stale": self.is_stale,
            "indexed_file_count": self.indexed_file_count,
            "changed_file_count": self.changed_file_count,
            "missing_file_count": self.missing_file_count,
            "extra_file_count": self.extra_file_count,
        }


@dataclass(frozen=True, slots=True)
class IndexedPathScore:
    """One path match loaded from a persisted lexical index."""

    path: str
    score: float


class LocalLexicalIndexWriter:
    """Write a local lexical index under ``.micro_model_agent/index``."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        index_path: str | Path | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self.repository = RepositoryRoot(repository_root)
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else self.repository.root / DEFAULT_INDEX_RELATIVE_PATH
        )
        self.max_file_bytes = max_file_bytes

    def write(self) -> LocalIndexResult:
        """Build and persist the repository index."""

        files: list[IndexedFile] = []
        postings: dict[str, list[dict[str, int | str]]] = defaultdict(list)
        skipped_large_count = 0
        skipped_binary_count = 0
        total_bytes = 0

        for path in self.repository.iter_files("**/*"):
            try:
                size_bytes = path.stat().st_size
            except OSError:
                continue
            if size_bytes > self.max_file_bytes:
                skipped_large_count += 1
                continue

            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if looks_binary(raw[:4096]):
                skipped_binary_count += 1
                continue

            relative_path = self.repository.relative_path(path)
            text = raw.decode("utf-8", errors="replace")
            term_counts = self._term_counts(relative_path, text)
            symbols, imports = self._python_metadata(relative_path, text)
            indexed_file = IndexedFile(
                path=relative_path,
                extension=path.suffix.casefold(),
                source_type=self._source_type(relative_path),
                is_test=self._is_test_path(relative_path),
                size_bytes=size_bytes,
                sha256=hashlib.sha256(raw).hexdigest(),
                term_count=sum(term_counts.values()),
                unique_term_count=len(term_counts),
                terms=dict(sorted(term_counts.items())),
                symbols=symbols,
                imports=imports,
            )
            files.append(indexed_file)
            total_bytes += size_bytes
            for term, count in indexed_file.terms.items():
                postings[term].append({"path": relative_path, "count": count})

        record = self._index_record(files, postings, skipped_large_count, skipped_binary_count)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return LocalIndexResult(
            repository_root=str(self.repository.root),
            index_path=str(self.index_path),
            indexed_file_count=len(files),
            skipped_large_count=skipped_large_count,
            skipped_binary_count=skipped_binary_count,
            total_bytes=total_bytes,
            unique_term_count=len(postings),
            source_type_counts=self._source_type_counts(files),
            extension_counts=self._extension_counts(files),
            test_file_count=sum(1 for file in files if file.is_test),
            symbol_count=sum(len(file.symbols) for file in files),
            import_count=sum(len(file.imports) for file in files),
        )

    def _index_record(
        self,
        files: list[IndexedFile],
        postings: Mapping[str, list[dict[str, int | str]]],
        skipped_large_count: int,
        skipped_binary_count: int,
    ) -> dict[str, Any]:
        """Build the serialized index document."""

        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "repository_root": str(self.repository.root),
            "index_kind": "local_lexical",
            "files": [asdict(file) for file in files],
            "postings": {term: postings[term] for term in sorted(postings)},
            "summary": {
                "indexed_file_count": len(files),
                "skipped_large_count": skipped_large_count,
                "skipped_binary_count": skipped_binary_count,
                "total_bytes": sum(file.size_bytes for file in files),
                "unique_term_count": len(postings),
                "source_type_counts": self._source_type_counts(files),
                "extension_counts": self._extension_counts(files),
                "test_file_count": sum(1 for file in files if file.is_test),
                "symbol_count": sum(len(file.symbols) for file in files),
                "import_count": sum(len(file.imports) for file in files),
            },
        }

    def _source_type_counts(self, files: list[IndexedFile]) -> dict[str, int]:
        """Count indexed files by source type."""

        return dict(sorted(Counter(file.source_type for file in files).items()))

    def _extension_counts(self, files: list[IndexedFile]) -> dict[str, int]:
        """Count indexed files by extension."""

        return dict(sorted(Counter(file.extension or "<none>" for file in files).items()))

    def _term_counts(self, relative_path: str, text: str) -> Counter[str]:
        """Tokenize a file path plus contents into lexical term counts."""

        counts: Counter[str] = Counter()
        for raw_term in re.findall(r"[A-Za-z0-9_]+", f"{relative_path}\n{text}"):
            term = raw_term.casefold()
            if len(term) <= 1:
                continue
            counts[term] += 1
            if term.endswith("s") and len(term) > 3:
                counts[term[:-1]] += 1
            for part in term.split("_"):
                if len(part) > 1 and part != term:
                    counts[part] += 1
        return counts

    def _python_metadata(
        self,
        relative_path: str,
        text: str,
    ) -> tuple[list[IndexedSymbol], list[str]]:
        """Extract Python symbols and imports without executing code."""

        if Path(relative_path).suffix.casefold() != ".py":
            return [], []
        try:
            tree = ast.parse(text, filename=relative_path)
        except SyntaxError:
            return [], []

        symbols: list[IndexedSymbol] = []
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(
                    IndexedSymbol(
                        name=node.name,
                        kind=type(node).__name__,
                        line_number=node.lineno,
                    )
                )
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

        symbols.sort(key=lambda symbol: (symbol.line_number, symbol.name))
        return symbols, sorted(imports)

    def _source_type(self, relative_path: str) -> str:
        """Classify a file so retrievers can filter and score by source type."""

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

    def _is_test_path(self, relative_path: str) -> bool:
        """Return true for common test-file path conventions."""

        path = relative_path.casefold()
        return "/test_" in f"/{path}" or path.endswith("_test.py") or "/tests/" in f"/{path}/"


class LocalLexicalIndexReader:
    """Read and query a persisted local lexical index."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        index_path: str | Path | None = None,
    ) -> None:
        self.repository = RepositoryRoot(repository_root)
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else self.repository.root / DEFAULT_INDEX_RELATIVE_PATH
        )

    def exists(self) -> bool:
        """Return true when the lexical index file exists."""

        return self.index_path.exists()

    def search_paths(
        self,
        query_terms: set[str],
        limit: int,
        *,
        source_types: set[str] | None = None,
        extensions: set[str] | None = None,
    ) -> list[IndexedPathScore]:
        """Return candidate paths ranked by indexed term overlap."""

        index = self._load_index()
        if index is None or not query_terms:
            return []

        path_scores: dict[str, float] = defaultdict(float)
        postings = index.get("postings", {})
        if not isinstance(postings, dict):
            return []
        metadata_by_path = self._metadata_by_path(index)

        for term in query_terms:
            raw_postings = postings.get(term, [])
            if not isinstance(raw_postings, list):
                continue
            for posting in raw_postings:
                if not isinstance(posting, dict):
                    continue
                path = posting.get("path")
                count = posting.get("count")
                if not isinstance(path, str) or not isinstance(count, (int, float)):
                    continue
                metadata = metadata_by_path.get(path)
                if not self._matches_filters(
                    metadata,
                    source_types=source_types,
                    extensions=extensions,
                ):
                    continue
                path_scores[path] += float(count)

        scored = [
            IndexedPathScore(path=path, score=score)
            for path, score in path_scores.items()
            if score > 0.0
        ]
        scored.sort(key=lambda item: (-item.score, item.path))
        return scored[:limit]

    def file_metadata(self, relative_path: str) -> dict[str, Any] | None:
        """Return stored index metadata for one repository-relative path."""

        index = self._load_index()
        if index is None:
            return None
        files = index.get("files", [])
        if not isinstance(files, list):
            return None
        for file_record in files:
            if isinstance(file_record, dict) and file_record.get("path") == relative_path:
                return {
                    "source_type": file_record.get("source_type"),
                    "extension": file_record.get("extension"),
                    "is_test": file_record.get("is_test"),
                    "symbols": file_record.get("symbols", []),
                    "imports": file_record.get("imports", []),
                    "sha256": file_record.get("sha256"),
                    "size_bytes": file_record.get("size_bytes"),
                }
        return None

    def freshness(self) -> LocalIndexFreshness | None:
        """Compare the persisted index with current repository files."""

        index = self._load_index()
        if index is None:
            return None

        metadata_by_path = self._metadata_by_path(index)
        current_hashes = self._current_file_hashes()
        changed_file_count = 0
        missing_file_count = 0

        for relative_path, metadata in metadata_by_path.items():
            current_sha256 = current_hashes.get(relative_path)
            if current_sha256 is None:
                missing_file_count += 1
                continue
            indexed_sha256 = metadata.get("sha256")
            if isinstance(indexed_sha256, str) and indexed_sha256 != current_sha256:
                changed_file_count += 1

        return LocalIndexFreshness(
            indexed_file_count=len(metadata_by_path),
            changed_file_count=changed_file_count,
            missing_file_count=missing_file_count,
            extra_file_count=len(set(current_hashes) - set(metadata_by_path)),
        )

    def _metadata_by_path(self, index: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Return valid file records keyed by repository-relative path."""

        files = index.get("files", [])
        if not isinstance(files, list):
            return {}

        records: dict[str, dict[str, Any]] = {}
        for file_record in files:
            if not isinstance(file_record, dict):
                continue
            path = file_record.get("path")
            if isinstance(path, str):
                records[path] = file_record
        return records

    def _matches_filters(
        self,
        metadata: dict[str, Any] | None,
        *,
        source_types: set[str] | None,
        extensions: set[str] | None,
    ) -> bool:
        """Return true when an indexed file metadata record matches filters."""

        if metadata is None:
            return False
        if source_types is not None and metadata.get("source_type") not in source_types:
            return False
        extension = metadata.get("extension")
        return not (extensions is not None and extension not in extensions)

    def _load_index(self) -> dict[str, Any] | None:
        """Load a valid index JSON document, returning None on stale/bad files."""

        try:
            loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(loaded, dict):
            return None
        if loaded.get("schema_version") != INDEX_SCHEMA_VERSION:
            return None
        return cast(dict[str, Any], loaded)

    def _current_file_hashes(self) -> dict[str, str]:
        """Return SHA-256 hashes for files currently eligible for indexing."""

        hashes: dict[str, str] = {}
        for path in self.repository.iter_files("**/*"):
            try:
                if path.stat().st_size > DEFAULT_MAX_FILE_BYTES:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if looks_binary(raw[:4096]):
                continue
            hashes[self.repository.relative_path(path)] = hashlib.sha256(raw).hexdigest()
        return hashes
