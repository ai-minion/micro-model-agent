"""Runtime repository composition helpers."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.local_index import (
    LocalIndexResult,
    LocalLexicalIndexWriter,
)
from micro_model_agent.repository_ops.infrastructure.metadata import (
    RepositoryInitializationResult,
    initialize_repository,
    is_repository_initialized,
)

__all__ = [
    "initialize_local_repository",
    "local_repository_initialized",
    "write_local_repository_index",
]


def initialize_local_repository(
    repository_root: str | Path,
    *,
    default_model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | Path | None = None,
) -> RepositoryInitializationResult:
    """Initialize local MicroModelAgent repository metadata."""

    return initialize_repository(
        repository_root,
        default_model=default_model,
        base_model=base_model,
        adapter_path=adapter_path,
    )


def local_repository_initialized(repository_root: str | Path) -> bool:
    """Return whether local MicroModelAgent metadata is initialized."""

    return is_repository_initialized(repository_root)


def write_local_repository_index(
    repository_root: str | Path,
    *,
    max_file_bytes: int,
) -> LocalIndexResult:
    """Write the standard local lexical repository index."""

    return LocalLexicalIndexWriter(
        repository_root,
        max_file_bytes=max_file_bytes,
    ).write()
