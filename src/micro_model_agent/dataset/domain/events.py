"""Domain events for the dataset bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from micro_model_agent.shared.domain.domain_event import DomainEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleAdded(DomainEvent):
    """Raised when a DatasetExample is added to a Dataset."""

    dataset_id: UUID
    example_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleLabelled(DomainEvent):
    """Raised when a DatasetExample's label is updated."""

    dataset_id: UUID
    example_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetExported(DomainEvent):
    """Raised when a Dataset is exported to a file."""

    dataset_id: UUID
    output_path: str
    example_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetMerged(DomainEvent):
    """Raised when datasets are merged into a new Dataset."""

    dataset_id: UUID
    source_count: int
    merged_count: int
    skipped_count: int
