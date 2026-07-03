"""Domain invariant and event tests for the dataset bounded context."""

from __future__ import annotations

from uuid import uuid4

import pytest

from micro_model_agent.dataset.domain.aggregate import Dataset
from micro_model_agent.dataset.domain.events import (
    DatasetExported,
    ExampleAdded,
    ExampleLabelled,
)
from micro_model_agent.dataset.domain.exceptions import (
    DuplicateExampleError,
    ExampleNotFoundError,
)
from micro_model_agent.dataset.domain.services import (
    DatasetMergeService,
    ExampleRelabelService,
)
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)


def _make_example(source: str = "test") -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "fix"},
        target={"patch": ""},
        label=DatasetLabel(
            outcome=OutcomeLabel.ACCEPTED,
            quality=QualityLabel.GOOD,
        ),
        source=source,
    )


# ---------------------------------------------------------------------------
# Dataset aggregate
# ---------------------------------------------------------------------------


def test_add_example_emits_event() -> None:
    ds = Dataset(name="seed")
    example = _make_example()
    ds.add_example(example)

    events = ds.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], ExampleAdded)
    assert events[0].example_id == example.id
    assert events[0].dataset_id == ds.id


def test_add_duplicate_raises() -> None:
    ds = Dataset(name="seed")
    example = _make_example()
    ds.add_example(example)
    with pytest.raises(DuplicateExampleError):
        ds.add_example(example)


def test_relabel_emits_event() -> None:
    ds = Dataset(name="seed")
    example = _make_example()
    ds.add_example(example)
    ds.pull_events()

    new_label = DatasetLabel(
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
    )
    ds.relabel(example.id, new_label)

    events = ds.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], ExampleLabelled)
    assert ds.examples[0].label.outcome == OutcomeLabel.REJECTED


def test_relabel_missing_raises() -> None:
    ds = Dataset(name="seed")
    with pytest.raises(ExampleNotFoundError):
        ds.relabel(uuid4(), DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD))


def test_record_export_emits_event() -> None:
    ds = Dataset(name="seed")
    ds.add_example(_make_example())
    ds.pull_events()

    ds.record_export("/tmp/out.jsonl")
    events = ds.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], DatasetExported)
    assert events[0].example_count == 1
    assert events[0].output_path == "/tmp/out.jsonl"


def test_examples_property_is_immutable_tuple() -> None:
    ds = Dataset(name="seed")
    ds.add_example(_make_example())
    examples = ds.examples
    assert isinstance(examples, tuple)
    assert len(examples) == 1


# ---------------------------------------------------------------------------
# DatasetMergeService
# ---------------------------------------------------------------------------


def test_merge_deduplicates_by_source() -> None:
    svc = DatasetMergeService()
    a = _make_example(source="trace:1")
    b = _make_example(source="trace:2")
    dup = _make_example(source="trace:1")  # same source, different id
    merged, skipped = svc.merge([[a, b], [dup]], deduplicate_by="source")
    assert len(merged) == 2
    assert skipped == 1


def test_merge_deduplicates_by_id() -> None:
    svc = DatasetMergeService()
    a = _make_example()
    merged, skipped = svc.merge([[a], [a]], deduplicate_by="id")
    assert len(merged) == 1
    assert skipped == 1


def test_merge_invalid_key_raises() -> None:
    svc = DatasetMergeService()
    with pytest.raises(ValueError):
        svc.merge([[]], deduplicate_by="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ExampleRelabelService
# ---------------------------------------------------------------------------


def test_relabel_updates_matching_example() -> None:
    svc = ExampleRelabelService()
    example = _make_example(source="trace:abc")
    updated, changed = svc.relabel(
        [example],
        source="trace:abc",
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
    )
    assert changed == 1
    assert updated[0].label.outcome == OutcomeLabel.REJECTED


def test_relabel_skips_non_matching() -> None:
    svc = ExampleRelabelService()
    example = _make_example(source="trace:abc")
    updated, changed = svc.relabel(
        [example],
        source="trace:xyz",
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
    )
    assert changed == 0
    assert updated[0].label.outcome == OutcomeLabel.ACCEPTED


def test_relabel_marks_curated_in_metadata() -> None:
    svc = ExampleRelabelService()
    example = _make_example()
    updated, _ = svc.relabel(
        [example],
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
    )
    assert updated[0].metadata.get("curated") is True
