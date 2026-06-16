"""Tests for dataset relabeling and merging helpers."""

from __future__ import annotations

from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.infrastructure.dataset_curation import merge_datasets, relabel_examples


def _example(source: str, quality: QualityLabel = QualityLabel.UNKNOWN) -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "Do the task"},
        target={"final_response": "Done."},
        source=source,
        label=DatasetLabel(outcome=OutcomeLabel.NEEDS_REVIEW, quality=quality),
        metadata={"trace_id": source.removeprefix("trace:")},
    )


def test_relabel_examples_updates_matching_trace_only() -> None:
    first = _example("trace:one")
    second = _example("trace:two")

    relabeled, changed = relabel_examples(
        [first, second],
        trace_id="one",
        outcome=OutcomeLabel.ACCEPTED,
        quality=QualityLabel.GOOD,
        reviewer_notes="Reviewed.",
    )

    assert changed == 1
    assert relabeled[0].label.outcome is OutcomeLabel.ACCEPTED
    assert relabeled[0].label.quality is QualityLabel.GOOD
    assert relabeled[0].metadata["curated"] is True
    assert relabeled[1].label.outcome is OutcomeLabel.NEEDS_REVIEW


def test_merge_datasets_deduplicates_by_source() -> None:
    first = _example("trace:one", QualityLabel.GOOD)
    duplicate = _example("trace:one", QualityLabel.BAD)
    second = _example("trace:two", QualityLabel.GOOD)

    merged, skipped = merge_datasets([[first], [duplicate, second]], deduplicate_by="source")

    assert skipped == 1
    assert [example.source for example in merged] == ["trace:one", "trace:two"]
    assert merged[0].label.quality is QualityLabel.GOOD
