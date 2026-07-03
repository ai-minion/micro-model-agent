"""Dataset — dataset context aggregate root."""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from micro_model_agent.dataset.domain.events import (
    DatasetExported,
    DatasetMerged,
    ExampleAdded,
    ExampleLabelled,
)
from micro_model_agent.dataset.domain.exceptions import (
    DuplicateExampleError,
    ExampleNotFoundError,
)
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetLabel,
)
from micro_model_agent.shared.domain.entity import Entity


class Dataset(Entity):
    """Aggregate root that owns a collection of DatasetExample records.

    Invariants enforced:
    - Each ``DatasetExample.id`` must be unique within the Dataset.
    - Relabeling a non-existent example raises ``ExampleNotFoundError``.
    """

    def __init__(self, name: str, id: UUID | None = None) -> None:
        super().__init__(id)
        self.name = name
        self._examples: dict[UUID, DatasetExample] = {}

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def add_example(self, example: DatasetExample) -> None:
        """Append a new example; raises DuplicateExampleError if already present."""

        if example.id in self._examples:
            raise DuplicateExampleError(example.id)
        self._examples[example.id] = example
        self._events.append(
            ExampleAdded(dataset_id=self.id, example_id=example.id)
        )

    def relabel(self, example_id: UUID, new_label: DatasetLabel) -> None:
        """Update the label of an existing example."""

        if example_id not in self._examples:
            raise ExampleNotFoundError(example_id)
        updated = replace(self._examples[example_id], label=new_label)
        self._examples[example_id] = updated
        self._events.append(
            ExampleLabelled(dataset_id=self.id, example_id=example_id)
        )

    def record_export(self, output_path: str) -> None:
        """Record that this dataset was exported to the given path."""

        self._events.append(
            DatasetExported(
                dataset_id=self.id,
                output_path=output_path,
                example_count=len(self._examples),
            )
        )

    def record_merge(self, source_count: int, merged_count: int, skipped_count: int) -> None:
        """Record that this dataset was produced by merging other datasets."""

        self._events.append(
            DatasetMerged(
                dataset_id=self.id,
                source_count=source_count,
                merged_count=merged_count,
                skipped_count=skipped_count,
            )
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def examples(self) -> tuple[DatasetExample, ...]:
        """Immutable view of the current examples."""

        return tuple(self._examples.values())

    @property
    def size(self) -> int:
        """Number of examples currently in the dataset."""

        return len(self._examples)
