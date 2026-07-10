"""Implementation of the repo.search tool.

The search tool supports three simple modes: file globbing, text search, and
Python symbol search using the standard library ast module.

For git repositories, glob and text search delegate to ``git ls-files`` and
``git grep`` respectively — both are dramatically faster than Python-level
filesystem walking on Windows/WSL paths and automatically respect .gitignore.
The pure-Python implementation is retained as a fallback for non-git repos.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.contracts import (
    RepoSearchMatch,
    RepoSearchRequest,
    RepoSearchResult,
    SearchKind,
    ToolError,
)
from micro_model_agent.repository_ops.infrastructure.paths import (
    RepositoryPathError,
    RepositoryRoot,
    looks_binary,
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
        # Auto-detect: if the query looks like a glob pattern (contains * or ?),
        # treat it as glob even when kind=text. Models sometimes emit kind=text
        # with a glob query like **/*.py.
        if request.query and ('*' in request.query or '?' in request.query):
            return self._glob_search(request)
        return self._text_search(request)

    # ------------------------------------------------------------------
    # Git-backed fast paths
    # ------------------------------------------------------------------

    def _is_git_repo(self) -> bool:
        return (self.repository.root / ".git").exists()

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run a git command under the repository root."""
        try:
            return subprocess.run(
                args,
                cwd=self.repository.root,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
                check=False,
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                args=args,
                returncode=127,
                stdout="",
                stderr="git not found",
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=args,
                returncode=124,
                stdout="",
                stderr="git timed out",
            )

    def _glob_search_git(self, request: RepoSearchRequest) -> RepoSearchResult | None:
        """List files via git ls-files. Returns None to signal fallback."""
        pattern = request.glob or request.query or "**/*"
        proc = self._run_git(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                f":(glob){pattern}",
            ]
        )
        if proc.returncode not in (0, 1):
            return None
        matches: list[RepoSearchMatch] = []
        truncated = False
        for line in proc.stdout.splitlines():
            path = line.strip()
            if not path:
                continue
            if len(matches) >= request.limit:
                truncated = True
                break
            matches.append(RepoSearchMatch(
                path=path,
                preview="file",
                metadata={"search_kind": SearchKind.GLOB.value},
            ))
        return RepoSearchResult(matches=matches, truncated=truncated)

    def _text_search_git(self, request: RepoSearchRequest) -> RepoSearchResult | None:
        """Search via git grep. Returns None to signal fallback."""
        assert request.query is not None
        pattern = request.glob or "**/*"
        proc = self._run_git(
            [
                "git",
                "grep",
                "--untracked",
                "--exclude-standard",
                "-i",
                "-n",
                "-e",
                request.query,
                "--",
                f":(glob){pattern}",
            ]
        )
        # exit 1 = no matches (not an error); anything else is a real failure
        if proc.returncode == 1:
            return RepoSearchResult(matches=[])
        if proc.returncode != 0:
            return None
        matches: list[RepoSearchMatch] = []
        truncated = False
        for line in proc.stdout.splitlines():
            if len(matches) >= request.limit:
                truncated = True
                break
            # git grep output: <file>:<line>:<content>
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            file_path, line_no_str, content = parts
            try:
                line_number = int(line_no_str)
            except ValueError:
                continue
            matches.append(RepoSearchMatch(
                path=file_path,
                line_number=line_number,
                preview=content.strip(),
                score=1.0,
                metadata={"search_kind": SearchKind.TEXT.value},
            ))
        return RepoSearchResult(matches=matches, truncated=truncated)

    # ------------------------------------------------------------------
    # Public search methods (git fast path → Python fallback)
    # ------------------------------------------------------------------

    def _glob_search(self, request: RepoSearchRequest) -> RepoSearchResult:
        """Return files whose paths match the requested glob pattern."""

        if self._is_git_repo():
            result = self._glob_search_git(request)
            if result is not None:
                return result

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

        if self._is_git_repo():
            result = self._text_search_git(request)
            if result is not None:
                return result

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
