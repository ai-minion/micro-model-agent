"""JSON record helpers for local training metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert Python-only objects into JSON-friendly values."""

    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def training_run_to_record(run: TrainingRun) -> dict[str, Any]:
    """Convert a TrainingRun dataclass into a JSON-ready dictionary."""

    return cast(dict[str, Any], _to_jsonable(asdict(run)))


def model_artifact_to_record(artifact: ModelArtifact) -> dict[str, Any]:
    """Convert a ModelArtifact dataclass into a JSON-ready dictionary."""

    return cast(dict[str, Any], _to_jsonable(asdict(artifact)))


def model_artifact_from_record(record: dict[str, Any]) -> ModelArtifact:
    """Rebuild a ModelArtifact from a JSON dictionary."""

    created_at = record.get("created_at")
    return ModelArtifact(
        id=UUID(record["id"]) if record.get("id") else uuid4(),
        name=record["name"],
        kind=ModelArtifactKind(record["kind"]),
        path=record["path"],
        base_model=record["base_model"],
        metrics=dict(record.get("metrics", {})),
        metadata=dict(record.get("metadata", {})),
        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
    )


def load_artifact_from_training_run(run_dir: Path) -> ModelArtifact:
    """Load the artifact metadata written inside a training run directory."""

    artifact_path = run_dir / "artifact.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact metadata not found: {artifact_path}")
    return model_artifact_from_record(json.loads(artifact_path.read_text(encoding="utf-8")))


def training_dataset_version(config: TrainingConfig) -> str | None:
    """Use the source dataset hash as the training run dataset version."""

    value = config.parameters.get("source_dataset_sha256")
    return value if isinstance(value, str) and value else None
