"""Tests for dataset domain services: DatasetMergeService and ExampleRelabelService."""

from __future__ import annotations

import pytest

from micro_model_agent.dataset.domain.services import (
    DatasetMergeService,
    ExampleRelabelService,
)
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)


def _example(
    source: str = "trace:default",
    outcome: OutcomeLabel = OutcomeLabel.ACCEPTED,
    quality: QualityLabel = QualityLabel.GOOD,
) -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "fix"},
        target={"patch": ""},
        label=DatasetLabel(outcome=outcome, quality=quality),
        source=source,
    )


# ---------------------------------------------------------------------------
# DatasetMergeService
# ---------------------------------------------------------------------------


def test_merge_empty_datasets() -> None:
    svc = DatasetMergeService()
    merged, skipped = svc.merge([])
    assert merged == []
    assert skipped == 0


def test_merge_single_dataset_unchanged() -> None:
    svc = DatasetMergeService()
    examples = [_example("a"), _example("b")]
    merged, skipped = svc.merge([examples])
    assert len(merged) == 2
    assert skipped == 0


def test_merge_deduplicates_by_source() -> None:
    svc = DatasetMergeService()
    a = _example("trace:1")
    b = _example("trace:2")
    dup = _example("trace:1")  # duplicate source

    merged, skipped = svc.merge([[a, b], [dup]])
    assert len(merged) == 2
    assert skipped == 1


def test_merge_preserves_first_occurrence() -> None:
    """The first occurrence wins during deduplication."""
    svc = DatasetMergeService()
    first = _example("trace:1", outcome=OutcomeLabel.ACCEPTED)
    second = _example("trace:1", outcome=OutcomeLabel.REJECTED)

    merged, _ = svc.merge([[first], [second]])
    assert merged[0].label.outcome == OutcomeLabel.ACCEPTED


def test_merge_deduplicates_by_id() -> None:
    svc = DatasetMergeService()
    example = _example("trace:1")
    # Same example object → same id
    merged, skipped = svc.merge([[example], [example]], deduplicate_by="id")
    assert len(merged) == 1
    assert skipped == 1


def test_merge_rejects_invalid_deduplicate_key() -> None:
    svc = DatasetMergeService()
    with pytest.raises(ValueError, match="deduplicate_by"):
        svc.merge([[_example()]], deduplicate_by="invalid")  # type: ignore[arg-type]


def test_merge_multiple_datasets_order_preserved() -> None:
    svc = DatasetMergeService()
    batch1 = [_example("s1"), _example("s2")]
    batch2 = [_example("s3"), _example("s4")]
    merged, _ = svc.merge([batch1, batch2])
    assert [e.source for e in merged] == ["s1", "s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# ExampleRelabelService
# ---------------------------------------------------------------------------


def test_relabel_all_examples_when_no_filter() -> None:
    svc = ExampleRelabelService()
    examples = [_example("a"), _example("b")]
    updated, changed = svc.relabel(
        examples,
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
    )
    assert changed == 2
    assert all(e.label.outcome == OutcomeLabel.REJECTED for e in updated)


def test_relabel_filters_by_source() -> None:
    svc = ExampleRelabelService()
    target = _example("trace:abc")
    other = _example("trace:xyz")
    updated, changed = svc.relabel(
        [target, other],
        source="trace:abc",
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
    )
    assert changed == 1
    relabelled = next(e for e in updated if e.source == "trace:abc")
    assert relabelled.label.outcome == OutcomeLabel.REJECTED
    unchanged = next(e for e in updated if e.source == "trace:xyz")
    assert unchanged.label.outcome == OutcomeLabel.ACCEPTED


def test_relabel_filters_by_input_outcome() -> None:
    svc = ExampleRelabelService()
    good = _example("s1", outcome=OutcomeLabel.ACCEPTED)
    bad = _example("s2", outcome=OutcomeLabel.REJECTED)
    updated, changed = svc.relabel(
        [good, bad],
        input_outcome=OutcomeLabel.ACCEPTED,
        outcome=OutcomeLabel.NEEDS_REVIEW,
        quality=QualityLabel.MIXED,
    )
    assert changed == 1
    relabelled = next(e for e in updated if e.source == "s1")
    assert relabelled.label.outcome == OutcomeLabel.NEEDS_REVIEW


def test_relabel_updates_failure_modes() -> None:
    svc = ExampleRelabelService()
    example = _example()
    updated, _ = svc.relabel(
        [example],
        failure_modes=(FailureMode.BAD_PATCH,),
    )
    assert updated[0].label.failure_modes == (FailureMode.BAD_PATCH,)


def test_relabel_preserves_failure_modes_when_not_specified() -> None:
    svc = ExampleRelabelService()
    original = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={},
        target={},
        label=DatasetLabel(
            outcome=OutcomeLabel.REJECTED,
            quality=QualityLabel.BAD,
            failure_modes=(FailureMode.TEST_FAILED,),
        ),
    )
    updated, _ = svc.relabel([original], quality=QualityLabel.MIXED)
    assert updated[0].label.failure_modes == (FailureMode.TEST_FAILED,)


def test_relabel_marks_as_curated() -> None:
    svc = ExampleRelabelService()
    example = _example()
    updated, _ = svc.relabel([example], outcome=OutcomeLabel.REJECTED, quality=QualityLabel.BAD)
    assert updated[0].metadata.get("curated") is True
    assert updated[0].metadata.get("review_required") is False


def test_relabel_empty_list_returns_empty() -> None:
    svc = ExampleRelabelService()
    updated, changed = svc.relabel([], outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    assert updated == []
    assert changed == 0
