"""Implementation of the repo.search tool.

The search tool supports three simple modes: file globbing, text search, and
Python symbol search using the standard library ast module.
"""

from __future__ import annotations

import ast
from pathlib import Path

from micro_model_agent.infrastructure.repository_paths import (
    RepositoryPathError,
    RepositoryRoot,
    looks_binary,
)
from micro_model_agent.infrastructure.tools.contracts import (
    RepoSearchMatch,
    RepoSearchRequest,
    RepoSearchResult,
    SearchKind,
    ToolError,
)


class RepoSearchTool:
    """Search repository files through a root-constrained adapter."""

    def __init__(self, repository_root: str | Path, max_file_bytes: int = 1_000_000) -> None:
        self.repository = RepositoryRoot(repository_root)
        self.max_file_bytes = max_file_bytes

    def run(self, request: RepoSearchRequest) -> RepoSearchResult:
        """Dispatch to the search strategy requested by the caller."""

        if request.kind is SearchKind.SYMBOL:
            return self._symbol_search(request)
        if request.kind is SearchKind.GLOB or request.query is None:
            return self._glob_search(request)
        return self._text_search(request)

    def _glob_search(self, request: RepoSearchRequest) -> RepoSearchResult:
        """Return files whose paths match the requested glob pattern."""

        pattern = request.glob or request.query or "**/*"
        try:
            files = self.repository.iter_files(pattern)
            matches: list[RepoSearchMatch] = []
            truncated = False
            for path in files:
                if len(matches) >= request.limit:
                    truncated = True
                    break
                matches.append(
                    RepoSearchMatch(
                        path=self.repository.relative_path(path),
                        preview="file",
                        metadata={"search_kind": SearchKind.GLOB.value},
                    )
                )
            return RepoSearchResult(matches=matches, truncated=truncated)
        except RepositoryPathError as exc:
            return RepoSearchResult(
                errors=[ToolError(code="unsafe_glob", message=str(exc), details={"glob": pattern})]
            )

    def _text_search(self, request: RepoSearchRequest) -> RepoSearchResult:
        """Search text files line-by-line for a case-insensitive query."""

        assert request.query is not None
        pattern = request.glob or "**/*"
        query = request.query.casefold()
        matches: list[RepoSearchMatch] = []
        errors: list[ToolError] = []
        truncated = False

        try:
            files = self.repository.iter_files(pattern)
            for path in files:
                text = self._read_text_for_search(path)
                if text is None:
                    continue
                # enumerate(..., start=1) reports line numbers the way editors do.
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if query not in line.casefold():
                        continue
                    if len(matches) >= request.limit:
                        truncated = True
                        return RepoSearchResult(matches=matches, truncated=truncated, errors=errors)
                    matches.append(
                        RepoSearchMatch(
                            path=self.repository.relative_path(path),
                            line_number=line_number,
                            preview=line.strip(),
                            score=1.0,
                            metadata={"search_kind": SearchKind.TEXT.value},
                        )
                    )
        except RepositoryPathError as exc:
            errors.append(
                ToolError(code="unsafe_glob", message=str(exc), details={"glob": pattern})
            )

        return RepoSearchResult(matches=matches, truncated=truncated, errors=errors)

    def _symbol_search(self, request: RepoSearchRequest) -> RepoSearchResult:
        """Search Python functions/classes by parsing files into an AST."""

        if request.query is None:
            return RepoSearchResult(
                errors=[
                    ToolError(
                        code="query_required",
                        message="symbol search requires a query",
                        details={"kind": request.kind.value},
                    )
                ]
            )

        pattern = request.glob or "**/*.py"
        query = request.query.casefold()
        matches: list[RepoSearchMatch] = []
        errors: list[ToolError] = []
        truncated = False

        try:
            files = self.repository.iter_files(pattern)
            for path in files:
                text = self._read_text_for_search(path)
                if text is None:
                    continue
                try:
                    # ast.parse lets us find definitions without executing code.
                    tree = ast.parse(text, filename=self.repository.relative_path(path))
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    symbol_name = self._symbol_name(node)
                    if symbol_name is None or query not in symbol_name.casefold():
                        continue
                    if len(matches) >= request.limit:
                        truncated = True
                        return RepoSearchResult(matches=matches, truncated=truncated, errors=errors)
                    matches.append(
                        RepoSearchMatch(
                            path=self.repository.relative_path(path),
                            line_number=getattr(node, "lineno", None),
                            preview=f"{type(node).__name__} {symbol_name}",
                            symbol=symbol_name,
                            score=1.0,
                            metadata={
                                "search_kind": SearchKind.SYMBOL.value,
                                "symbol_type": type(node).__name__,
                            },
                        )
                    )
        except RepositoryPathError as exc:
            errors.append(
                ToolError(code="unsafe_glob", message=str(exc), details={"glob": pattern})
            )

        return RepoSearchResult(matches=matches, truncated=truncated, errors=errors)

    def _read_text_for_search(self, path: Path) -> str | None:
        """Read a file for searching, skipping large or binary files."""

        try:
            if path.stat().st_size > self.max_file_bytes:
                return None
            raw = path.read_bytes()
        except OSError:
            return None
        if looks_binary(raw[:4096]):
            return None
        return raw.decode("utf-8", errors="replace")

    def _symbol_name(self, node: ast.AST) -> str | None:
        """Return the name for AST nodes that define Python symbols."""

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return node.name
        return None
