"""Tests for dataset domain events and exceptions."""

from __future__ import annotations

from uuid import uuid4

import pytest

from micro_model_agent.dataset.domain.events import (
    DatasetExported,
    DatasetMerged,
    ExampleAdded,
    ExampleLabelled,
)
from micro_model_agent.dataset.domain.exceptions import (
    DuplicateExampleError,
    ExampleNotFoundError,
    InvalidLabelError,
)
from micro_model_agent.shared.domain.domain_event import DomainEvent


# ---------------------------------------------------------------------------
# Domain events
# ---------------------------------------------------------------------------


def test_example_added_fields() -> None:
    dataset_id = uuid4()
    example_id = uuid4()
    event = ExampleAdded(dataset_id=dataset_id, example_id=example_id)
    assert event.dataset_id == dataset_id
    assert event.example_id == example_id
    assert isinstance(event, DomainEvent)


def test_example_labelled_fields() -> None:
    event = ExampleLabelled(dataset_id=uuid4(), example_id=uuid4())
    assert isinstance(event, DomainEvent)


def test_dataset_exported_fields() -> None:
    dataset_id = uuid4()
    event = DatasetExported(
        dataset_id=dataset_id, output_path="/tmp/out.jsonl", example_count=42
    )
    assert event.output_path == "/tmp/out.jsonl"
    assert event.example_count == 42


def test_dataset_merged_fields() -> None:
    event = DatasetMerged(
        dataset_id=uuid4(), source_count=3, merged_count=25, skipped_count=2
    )
    assert event.source_count == 3
    assert event.merged_count == 25
    assert event.skipped_count == 2


def test_all_dataset_events_are_domain_events() -> None:
    ds_id = uuid4()
    events = [
        ExampleAdded(dataset_id=ds_id, example_id=uuid4()),
        ExampleLabelled(dataset_id=ds_id, example_id=uuid4()),
        DatasetExported(dataset_id=ds_id, output_path="/tmp", example_count=1),
        DatasetMerged(dataset_id=ds_id, source_count=1, merged_count=1, skipped_count=0),
    ]
    for event in events:
        assert isinstance(event, DomainEvent)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


def test_duplicate_example_error_includes_id() -> None:
    example_id = uuid4()
    exc = DuplicateExampleError(example_id)
    assert str(example_id) in str(exc)
    assert exc.example_id == example_id


def test_example_not_found_error_includes_id() -> None:
    example_id = uuid4()
    exc = ExampleNotFoundError(example_id)
    assert str(example_id) in str(exc)
    assert exc.example_id == example_id


def test_invalid_label_error_is_domain_exception() -> None:
    from micro_model_agent.shared.domain.exceptions import DomainException
    with pytest.raises(DomainException):
        raise InvalidLabelError("bad label")


def test_duplicate_and_not_found_are_domain_exceptions() -> None:
    from micro_model_agent.shared.domain.exceptions import DomainException
    example_id = uuid4()
    assert isinstance(DuplicateExampleError(example_id), DomainException)
    assert isinstance(ExampleNotFoundError(example_id), DomainException)
