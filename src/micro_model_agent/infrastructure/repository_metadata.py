"""MicroModelAgent repository metadata initialization.

This module creates the hidden ``.micro_model_agent`` directory that stores
datasets, training runs, and configuration for one repository.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

METADATA_DIR_NAME = ".micro_model_agent"
CONFIG_FILE_NAME = "config.json"
CONFIG_SCHEMA_VERSION = 1
DEFAULT_METADATA_DIRECTORIES: tuple[str, ...] = (
    "datasets",
    "evaluations",
    "training",
    "training/artifacts",
    "training/runs",
)


@dataclass(frozen=True, slots=True)
class RepositoryInitializationResult:
    """Result of initializing local MicroModelAgent metadata."""

    ok: bool
    already_initialized: bool
    repository_root: str
    metadata_dir: str
    config_path: str
    created_directories: tuple[str, ...]
    config: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepositoryConfigUpdateResult:
    """Result of updating local MicroModelAgent repository configuration."""

    ok: bool
    repository_root: str
    metadata_dir: str
    config_path: str
    config: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def repository_config_path(repository_root: str | Path = ".") -> Path:
    """Return the MicroModelAgent repository config path."""

    # resolve() turns "." into an absolute path so later file writes are stable.
    return Path(repository_root).resolve() / METADATA_DIR_NAME / CONFIG_FILE_NAME


def is_repository_initialized(repository_root: str | Path = ".") -> bool:
    """Return true when the repository has a valid MicroModelAgent config."""

    config_path = repository_config_path(repository_root)
    if not config_path.exists():
        return False
    try:
        # Invalid JSON means the directory exists but is not usable yet.
        config = _read_config(config_path)
    except ValueError:
        return False
    return config.get("schema_version") == CONFIG_SCHEMA_VERSION


def load_repository_config(repository_root: str | Path = ".") -> dict[str, Any] | None:
    """Load repository-local MicroModelAgent config when it exists and is valid."""

    config_path = repository_config_path(repository_root)
    if not config_path.exists():
        return None
    config = _read_config(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        return None
    return config


def initialize_repository(
    repository_root: str | Path = ".",
    *,
    default_model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | Path | None = None,
) -> RepositoryInitializationResult:
    """Initialize MicroModelAgent metadata for a repository.

    The operation is idempotent. Existing valid config is preserved while the
    expected metadata directories are still ensured.
    """

    root = Path(repository_root).resolve()
    metadata_dir = root / METADATA_DIR_NAME
    config_path = metadata_dir / CONFIG_FILE_NAME
    created_directories = _ensure_metadata_directories(metadata_dir)

    if config_path.exists():
        # Existing valid config wins; initialization should be safe to run more
        # than once without overwriting user settings.
        try:
            config = _read_config(config_path)
        except ValueError as exc:
            return RepositoryInitializationResult(
                ok=False,
                already_initialized=False,
                repository_root=str(root),
                metadata_dir=str(metadata_dir),
                config_path=str(config_path),
                created_directories=created_directories,
                config={},
                error=str(exc),
            )
        return RepositoryInitializationResult(
            ok=True,
            already_initialized=True,
            repository_root=str(root),
            metadata_dir=str(metadata_dir),
            config_path=str(config_path),
            created_directories=created_directories,
            config=config,
        )

    config = _build_config(
        repository_root=root,
        default_model=default_model,
        base_model=base_model,
        adapter_path=adapter_path,
    )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RepositoryInitializationResult(
        ok=True,
        already_initialized=False,
        repository_root=str(root),
        metadata_dir=str(metadata_dir),
        config_path=str(config_path),
        created_directories=created_directories,
        config=config,
    )


def update_model_configuration(
    repository_root: str | Path = ".",
    *,
    base_model: str,
    adapter_path: str | Path,
    selected_promotion: dict[str, Any] | None = None,
) -> RepositoryConfigUpdateResult:
    """Update repository-local model defaults.

    This is used after a promoted artifact has already passed the manual gate
    and been recorded in the local promotion registry.
    """

    root = Path(repository_root).resolve()
    metadata_dir = root / METADATA_DIR_NAME
    config_path = metadata_dir / CONFIG_FILE_NAME
    _ensure_metadata_directories(metadata_dir)

    if config_path.exists():
        try:
            config = _read_config(config_path)
        except ValueError as exc:
            return RepositoryConfigUpdateResult(
                ok=False,
                repository_root=str(root),
                metadata_dir=str(metadata_dir),
                config_path=str(config_path),
                config={},
                error=str(exc),
            )
        if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
            return RepositoryConfigUpdateResult(
                ok=False,
                repository_root=str(root),
                metadata_dir=str(metadata_dir),
                config_path=str(config_path),
                config=config,
                error=f"unsupported MicroModelAgent config schema: {config_path}",
            )
    else:
        config = _build_config(
            repository_root=root,
            default_model=None,
            base_model=None,
            adapter_path=None,
        )

    model_config = dict(config.get("model", {}))
    model_config["base_model"] = base_model
    model_config["adapter_path"] = str(Path(adapter_path))
    if selected_promotion is not None:
        model_config["selected_promotion"] = selected_promotion
    config["model"] = model_config
    config["updated_at"] = datetime.now(UTC).isoformat()
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RepositoryConfigUpdateResult(
        ok=True,
        repository_root=str(root),
        metadata_dir=str(metadata_dir),
        config_path=str(config_path),
        config=config,
    )


def _ensure_metadata_directories(metadata_dir: Path) -> tuple[str, ...]:
    """Create expected metadata folders and return only the ones newly created."""

    created: list[str] = []
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for relative in DEFAULT_METADATA_DIRECTORIES:
        path = metadata_dir / relative
        if not path.exists():
            created.append(relative)
        path.mkdir(parents=True, exist_ok=True)
    return tuple(created)


def _read_config(config_path: Path) -> dict[str, Any]:
    """Read and type-check the local config JSON file."""

    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid MicroModelAgent config: {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"MicroModelAgent config must be a JSON object: {config_path}")
    return cast(dict[str, Any], loaded)


def _build_config(
    *,
    repository_root: Path,
    default_model: str | None,
    base_model: str | None,
    adapter_path: str | Path | None,
) -> dict[str, Any]:
    """Build the initial config dictionary written during first initialization."""

    model_config: dict[str, Any] = {}
    if default_model:
        model_config["default_model"] = default_model
    if base_model:
        model_config["base_model"] = base_model
    if adapter_path:
        model_config["adapter_path"] = str(Path(adapter_path))

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "repository_root": str(repository_root),
        "created_at": datetime.now(UTC).isoformat(),
        "model": model_config,
    }
