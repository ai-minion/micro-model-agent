"""Helpers for curating and combining dataset examples."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)

DeduplicateBy = Literal["id", "source"]


def relabel_examples(
    examples: list[DatasetExample],
    *,
    trace_id: str | None = None,
    source: str | None = None,
    input_outcome: OutcomeLabel | None = None,
    input_quality: QualityLabel | None = None,
    outcome: OutcomeLabel | None = None,
    quality: QualityLabel | None = None,
    failure_modes: tuple[FailureMode, ...] | None = None,
    reviewer_notes: str | None = None,
) -> tuple[list[DatasetExample], int]:
    """Return examples with matching records relabeled."""

    updated: list[DatasetExample] = []
    changed = 0
    for example in examples:
        if not _matches(
            example,
            trace_id=trace_id,
            source=source,
            outcome=input_outcome,
            quality=input_quality,
        ):
            updated.append(example)
            continue

        label = DatasetLabel(
            outcome=outcome or example.label.outcome,
            quality=quality or example.label.quality,
            failure_modes=(
                failure_modes if failure_modes is not None else example.label.failure_modes
            ),
            reviewer_notes=(
                reviewer_notes
                if reviewer_notes is not None
                else example.label.reviewer_notes
            ),
        )
        metadata = {
            **example.metadata,
            "review_required": False,
            "curated": True,
        }
        updated.append(replace(example, label=label, metadata=metadata))
        changed += 1
    return updated, changed


def merge_datasets(
    datasets: list[list[DatasetExample]],
    *,
    deduplicate_by: DeduplicateBy = "source",
) -> tuple[list[DatasetExample], int]:
    """Merge datasets while keeping the first example for each dedupe key."""

    merged: list[DatasetExample] = []
    seen: set[str] = set()
    skipped = 0
    for examples in datasets:
        for example in examples:
            key = str(example.id) if deduplicate_by == "id" else example.source
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            merged.append(example)
    return merged, skipped


def _matches(
    example: DatasetExample,
    *,
    trace_id: str | None,
    source: str | None,
    outcome: OutcomeLabel | None,
    quality: QualityLabel | None,
) -> bool:
    if trace_id is not None and example.metadata.get("trace_id") != trace_id:
        return False
    if source is not None and example.source != source:
        return False
    if outcome is not None and example.label.outcome is not outcome:
        return False
    return not (quality is not None and example.label.quality is not quality)
