"""Domain contracts for fine-tuning dataset examples.

Dataset records are kept as small immutable dataclasses so the rest of the code
can pass training examples around without depending on a particular file format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class OutcomeLabel(StrEnum):
    """Human or automated outcome label for a workflow or dataset example."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    PARTIAL = "partial"
    ERRORED = "errored"


class QualityLabel(StrEnum):
    """Quality label used for later supervised and preference datasets."""

    GOOD = "good"
    BAD = "bad"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class FailureMode(StrEnum):
    """Common failure modes useful for evaluation and fine-tuning."""

    INVALID_TOOL_SCHEMA = "invalid_tool_schema"
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    UNSAFE_PATH_REQUESTED = "unsafe_path_requested"
    HALLUCINATED_FILE = "hallucinated_file"
    RETRIEVAL_MISSED_CONTEXT = "retrieval_missed_context"
    IGNORED_RETRIEVED_CONTEXT = "ignored_retrieved_context"
    BAD_PATCH = "bad_patch"
    PATCH_FAILED_TO_APPLY = "patch_failed_to_apply"
    TEST_FAILED = "test_failed"
    VERIFICATION_SKIPPED = "verification_skipped"
    TOO_LARGE_CHANGE = "too_large_change"
    ARCHITECTURE_VIOLATION = "architecture_violation"
    UNCLEAR_USER_GOAL = "unclear_user_goal"
    PROVIDER_ERROR = "provider_error"


class DatasetExampleKind(StrEnum):
    """Dataset views that can be derived from traces or synthetic templates."""

    TOOL_USE = "tool_use"
    DOCUMENTATION_GROUNDED = "documentation_grounded"
    CODEBASE_GROUNDED = "codebase_grounded"
    REPAIR = "repair"
    PREFERENCE_PAIR = "preference_pair"
    EVALUATION = "evaluation"


class DatasetSplitName(StrEnum):
    """Dataset split names used by training and evaluation workflows."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class DatasetLabel:
    """Labels attached to a trace-derived or synthetic training example."""

    outcome: OutcomeLabel
    quality: QualityLabel
    # A tuple is used here because tuples are immutable, which matches the
    # frozen dataclass and makes labels safe to share.
    failure_modes: tuple[FailureMode, ...] = ()
    reviewer_notes: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetExample:
    """A model-training example derived from a trace or synthetic template."""

    # "kind" describes how the example will be used during training/evaluation.
    kind: DatasetExampleKind
    # input and target intentionally stay dict-shaped because different example
    # kinds have different payloads.
    input: dict[str, Any]
    target: dict[str, Any]
    label: DatasetLabel
    # The fields below are metadata that help trace an exported record back to
    # where it came from.
    id: UUID = field(default_factory=uuid4)
    source: str = "synthetic"
    tool_schema_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A named split containing dataset example identifiers."""

    name: DatasetSplitName
    # Only IDs are stored in the split so the same example can live in one
    # central store and be referenced by different training/evaluation views.
    example_ids: tuple[UUID, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
