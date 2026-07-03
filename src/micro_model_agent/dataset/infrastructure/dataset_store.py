"""Local JSONL dataset storage and serialization.

JSONL means "one JSON object per line". It is convenient for datasets because
large files can be streamed line-by-line instead of loaded as one giant list.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.metadata import metadata_with_tool_profile


def dataset_example_to_record(example: DatasetExample) -> dict[str, Any]:
    """Convert a domain dataset example into a JSON-serializable record."""

    # UUIDs, enums, and datetimes are Python objects, so they are converted to
    # strings before json.dumps writes them to disk.
    return {
        "id": str(example.id),
        "kind": example.kind.value,
        "source": example.source,
        "tool_schema_version": example.tool_schema_version,
        "input": example.input,
        "target": example.target,
        "label": {
            "outcome": example.label.outcome.value,
            "quality": example.label.quality.value,
            "failure_modes": [mode.value for mode in example.label.failure_modes],
            "reviewer_notes": example.label.reviewer_notes,
        },
        "metadata": metadata_with_tool_profile(example),
        "created_at": example.created_at.isoformat(),
    }


def dataset_example_from_record(record: dict[str, Any]) -> DatasetExample:
    """Convert a JSON record into a domain dataset example."""

    label_record = record["label"]
    created_at = record.get("created_at")
    # Rebuild the domain dataclass and enum values from their stored strings.
    return DatasetExample(
        id=UUID(record["id"]) if record.get("id") else uuid4(),
        kind=DatasetExampleKind(record["kind"]),
        source=record.get("source", "synthetic"),
        tool_schema_version=record.get("tool_schema_version"),
        input=dict(record["input"]),
        target=dict(record["target"]),
        label=DatasetLabel(
            outcome=OutcomeLabel(label_record["outcome"]),
            quality=QualityLabel(label_record["quality"]),
            failure_modes=tuple(
                FailureMode(mode) for mode in label_record.get("failure_modes", [])
            ),
            reviewer_notes=label_record.get("reviewer_notes"),
        ),
        metadata=dict(record.get("metadata", {})),
        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
    )


def load_dataset_examples(path: Path) -> list[DatasetExample]:
    """Load dataset examples from a JSONL file."""

    if not path.exists():
        return []
    examples: list[DatasetExample] = []
    # enumerate(..., start=1) gives human-friendly line numbers in error messages.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
        examples.append(dataset_example_from_record(raw))
    return examples


def write_dataset_examples(path: Path, examples: Iterable[DatasetExample]) -> None:
    """Write dataset examples to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Opening with "w" replaces the file; JsonlDatasetExampleStore.save appends.
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(dataset_example_to_record(example), sort_keys=True))
            file.write("\n")


class LocalDatasetExampleReader:
    """Filesystem adapter for loading dataset examples from JSONL files."""

    def load_dataset_examples(self, path: Path) -> list[DatasetExample]:
        """Load dataset examples from a JSONL path."""

        return load_dataset_examples(path)


class LocalDatasetExampleWriter:
    """Filesystem adapter for writing dataset examples to JSONL files."""

    async def save_dataset_examples(
        self,
        path: Path,
        examples: list[DatasetExample],
    ) -> None:
        """Persist dataset examples to a JSONL path."""

        write_dataset_examples(path, examples)


class JsonlDatasetExampleStore:
    """Local JSONL implementation of the dataset example store port."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def save(self, example: DatasetExample) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append one example per line so the file can grow over many workflow runs.
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(dataset_example_to_record(example), sort_keys=True))
            file.write("\n")

    async def save_many(self, examples: Iterable[DatasetExample]) -> None:
        write_dataset_examples(self.path, examples)

    async def list(self, kind: str | None = None) -> list[DatasetExample]:
        examples = load_dataset_examples(self.path)
        if kind is None:
            return examples
        return [example for example in examples if example.kind.value == kind]
