"""Compatibility imports for local repository indexing adapters."""

from micro_model_agent.infrastructure.repositories.local_index import (
    DATA_EXTENSIONS,
    DEFAULT_INDEX_RELATIVE_PATH,
    DEFAULT_MAX_FILE_BYTES,
    DOCUMENTATION_EXTENSIONS,
    INDEX_SCHEMA_VERSION,
    SOURCE_CODE_EXTENSIONS,
    IndexedFile,
    IndexedPathScore,
    IndexedSymbol,
    LocalIndexFreshness,
    LocalIndexResult,
    LocalLexicalIndexReader,
    LocalLexicalIndexWriter,
)

__all__ = [
    "DATA_EXTENSIONS",
    "DEFAULT_INDEX_RELATIVE_PATH",
    "DEFAULT_MAX_FILE_BYTES",
    "DOCUMENTATION_EXTENSIONS",
    "INDEX_SCHEMA_VERSION",
    "SOURCE_CODE_EXTENSIONS",
    "IndexedFile",
    "IndexedPathScore",
    "IndexedSymbol",
    "LocalIndexFreshness",
    "LocalIndexResult",
    "LocalLexicalIndexReader",
    "LocalLexicalIndexWriter",
]
