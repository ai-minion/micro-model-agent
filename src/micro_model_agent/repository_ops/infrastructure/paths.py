"""Repository-root constrained path utilities.

Repository tools use this module before touching files. The goal is to accept
normal repository-relative paths while rejecting absolute paths and ``..``
traversal that could escape the project.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path


class RepositoryPathError(ValueError):
    """Raised when a path is not safe to resolve within a repository."""


DEFAULT_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".micro_model_agent",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".traces",
        ".venv",
        ".venv-wsl",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


class RepositoryRoot:
    """Resolve and iterate paths inside a repository root."""

    def __init__(
        self,
        root: str | Path,
        excluded_parts: Iterable[str] = DEFAULT_EXCLUDED_PARTS,
    ) -> None:
        self.root = Path(root).resolve()
        self.excluded_parts = frozenset(excluded_parts)

    def resolve_file(self, relative_path: str) -> Path:
        """Resolve a repository-relative file path."""

        normalized = self._normalize_relative_path(relative_path)
        # resolve(strict=False) normalizes the path even when the file does not
        # exist yet, which is useful for patch creation.
        candidate = (self.root / normalized).resolve(strict=False)
        self._ensure_inside_root(candidate)
        return candidate

    def relative_path(self, path: Path) -> str:
        """Return a POSIX-style repository-relative path."""

        resolved = path.resolve(strict=False)
        self._ensure_inside_root(resolved)
        return resolved.relative_to(self.root).as_posix()

    def iter_files(self, pattern: str = "**/*") -> Iterator[Path]:
        """Yield files under the repository root, skipping generated directories."""

        normalized = self._normalize_glob(pattern)
        for path in sorted(self.root.glob(normalized)):
            try:
                # Convert to a root-relative path so excluded directories can be
                # checked by path part instead of by fragile string matching.
                relative = path.resolve(strict=False).relative_to(self.root)
            except (OSError, ValueError):
                continue
            if self.excluded_parts.intersection(relative.parts):
                continue
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
            yield path

    def _ensure_inside_root(self, path: Path) -> None:
        """Raise if a resolved path points outside the repository root."""

        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise RepositoryPathError("path resolves outside repository root") from exc

    def _normalize_relative_path(self, path: str) -> str:
        """Reject unsafe file paths before they are joined with the root."""

        normalized = path.replace("\\", "/").strip()
        if not normalized:
            raise RepositoryPathError("path is required")
        if "\x00" in normalized:
            raise RepositoryPathError("path cannot contain null bytes")
        if re.match(r"^[A-Za-z]:", normalized):
            raise RepositoryPathError("path must be repository-relative")
        if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
            raise RepositoryPathError("path must be repository-relative")
        if "/../" in f"/{normalized}/":
            raise RepositoryPathError("path must not contain parent traversal")
        return normalized.removeprefix("./")

    def _normalize_glob(self, pattern: str) -> str:
        """Reject unsafe glob patterns while allowing normal repository globs."""

        normalized = pattern.replace("\\", "/").strip()
        if not normalized:
            return "**/*"
        if "\x00" in normalized:
            raise RepositoryPathError("glob cannot contain null bytes")
        if re.match(r"^[A-Za-z]:", normalized):
            raise RepositoryPathError("glob must be repository-relative")
        if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
            raise RepositoryPathError("glob must be repository-relative")
        if "/../" in f"/{normalized}/":
            raise RepositoryPathError("glob must not contain parent traversal")
        # A bare extension glob like *.py only matches root-level files in
        # Python's Path.glob(). Expand to **/*.py so models get the expected
        # recursive behaviour when they emit {"glob": "*.py"}.
        if "/" not in normalized and normalized.startswith("*") and not normalized.startswith("**"):
            normalized = "**/" + normalized
        return normalized.removeprefix("./")


def looks_binary(sample: bytes) -> bool:
    """Return true when a byte sample appears to be binary."""

    # Text files almost never contain null bytes; binary files often do.
    return b"\x00" in sample
