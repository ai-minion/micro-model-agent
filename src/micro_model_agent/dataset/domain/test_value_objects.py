"""Tests for dataset domain value objects."""

from __future__ import annotations

from uuid import uuid4

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    DatasetSplit,
    DatasetSplitName,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)

# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


def test_outcome_label_values() -> None:
    assert OutcomeLabel.ACCEPTED.value == "accepted"
    assert OutcomeLabel.REJECTED.value == "rejected"
    assert OutcomeLabel.NEEDS_REVIEW.value == "needs_review"
    assert OutcomeLabel.PARTIAL.value == "partial"
    assert OutcomeLabel.ERRORED.value == "errored"


def test_quality_label_values() -> None:
    assert QualityLabel.GOOD.value == "good"
    assert QualityLabel.BAD.value == "bad"
    assert QualityLabel.MIXED.value == "mixed"
    assert QualityLabel.UNKNOWN.value == "unknown"


def test_dataset_example_kind_values() -> None:
    assert DatasetExampleKind.TOOL_USE.value == "tool_use"
    assert DatasetExampleKind.REPAIR.value == "repair"
    assert DatasetExampleKind.EVALUATION.value == "evaluation"


def test_dataset_split_name_values() -> None:
    assert DatasetSplitName.TRAIN.value == "train"
    assert DatasetSplitName.VALIDATION.value == "validation"
    assert DatasetSplitName.TEST.value == "test"


# ---------------------------------------------------------------------------
# DatasetLabel
# ---------------------------------------------------------------------------


def test_dataset_label_required_fields() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    assert label.outcome == OutcomeLabel.ACCEPTED
    assert label.quality == QualityLabel.GOOD
    assert label.failure_modes == ()
    assert label.reviewer_notes is None


def test_dataset_label_with_failure_modes() -> None:
    label = DatasetLabel(
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
        failure_modes=(FailureMode.BAD_PATCH, FailureMode.TEST_FAILED),
        reviewer_notes="patch did not apply cleanly",
    )
    assert FailureMode.BAD_PATCH in label.failure_modes
    assert label.reviewer_notes == "patch did not apply cleanly"


def test_dataset_label_is_frozen() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    try:
        label.outcome = OutcomeLabel.REJECTED  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# DatasetExample
# ---------------------------------------------------------------------------


def test_dataset_example_required_fields() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "fix"},
        target={"patch": "diff"},
        label=label,
    )
    assert example.kind == DatasetExampleKind.TOOL_USE
    assert example.input["goal"] == "fix"
    assert example.target["patch"] == "diff"
    assert example.label is label
    assert example.id is not None
    assert example.source == "synthetic"
    assert example.metadata == {}
    assert example.created_at is not None


def test_dataset_example_custom_source() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={},
        target={},
        label=label,
        source="trace:abc-123",
    )
    assert example.source == "trace:abc-123"


def test_dataset_example_is_frozen() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    example = DatasetExample(kind=DatasetExampleKind.REPAIR, input={}, target={}, label=label)
    try:
        example.source = "other"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


def test_dataset_example_unique_ids() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    e1 = DatasetExample(kind=DatasetExampleKind.REPAIR, input={}, target={}, label=label)
    e2 = DatasetExample(kind=DatasetExampleKind.REPAIR, input={}, target={}, label=label)
    assert e1.id != e2.id


def test_dataset_example_with_metadata() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={},
        target={},
        label=label,
        metadata={"trace_id": "abc", "curated": True},
    )
    assert example.metadata["curated"] is True


# ---------------------------------------------------------------------------
# DatasetSplit
# ---------------------------------------------------------------------------


def test_dataset_split_fields() -> None:
    ids = (uuid4(), uuid4(), uuid4())
    split = DatasetSplit(name=DatasetSplitName.TRAIN, example_ids=ids)
    assert split.name == DatasetSplitName.TRAIN
    assert len(split.example_ids) == 3
    assert split.id is not None


def test_dataset_split_metadata_default_empty() -> None:
    split = DatasetSplit(name=DatasetSplitName.VALIDATION, example_ids=())
    assert split.metadata == {}


# ---------------------------------------------------------------------------
# FailureMode completeness
# ---------------------------------------------------------------------------


def test_all_failure_modes_are_defined() -> None:
    """Smoke test: all documented failure modes can be referenced."""
    modes = [
        FailureMode.INVALID_TOOL_SCHEMA,
        FailureMode.WRONG_TOOL_SELECTED,
        FailureMode.UNSAFE_PATH_REQUESTED,
        FailureMode.HALLUCINATED_FILE,
        FailureMode.RETRIEVAL_MISSED_CONTEXT,
        FailureMode.IGNORED_RETRIEVED_CONTEXT,
        FailureMode.BAD_PATCH,
        FailureMode.PATCH_FAILED_TO_APPLY,
        FailureMode.TEST_FAILED,
        FailureMode.VERIFICATION_SKIPPED,
        FailureMode.TOO_LARGE_CHANGE,
        FailureMode.ARCHITECTURE_VIOLATION,
        FailureMode.UNCLEAR_USER_GOAL,
        FailureMode.PROVIDER_ERROR,
    ]
    assert len(modes) == 14
