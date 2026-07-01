"""Compatibility imports for dataset persistence adapters."""

from micro_model_agent.infrastructure.persistence.dataset_store import (
    JsonlDatasetExampleStore,
    LocalDatasetExampleReader,
    LocalDatasetExampleWriter,
    dataset_example_from_record,
    dataset_example_to_record,
    load_dataset_examples,
    write_dataset_examples,
)

__all__ = [
    "JsonlDatasetExampleStore",
    "LocalDatasetExampleReader",
    "LocalDatasetExampleWriter",
    "dataset_example_from_record",
    "dataset_example_to_record",
    "load_dataset_examples",
    "write_dataset_examples",
]
