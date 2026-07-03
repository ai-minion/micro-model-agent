"""Dataset context domain services.

Stateless pure-Python services that operate exclusively on dataset domain
value objects.  They have no infrastructure dependency and are trivial to
unit-test.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)

DeduplicateBy = Literal["id", "source"]


class DatasetMergeService:
    """Merge multiple lists of DatasetExample with deduplication.

    Deduplication keeps the *first* occurrence of each key so merge order
    matters: put the higher-quality dataset first.
    """

    SUPPORTED_KEYS: frozenset[str] = frozenset({"id", "source"})

    def merge(
        self,
        datasets: list[list[DatasetExample]],
        *,
        deduplicate_by: DeduplicateBy = "source",
    ) -> tuple[list[DatasetExample], int]:
        """Return (merged_examples, skipped_count).

        ``skipped_count`` is the number of examples dropped due to duplicate
        keys.
        """

        if deduplicate_by not in self.SUPPORTED_KEYS:
            raise ValueError(
                f"Unsupported deduplicate_by key: {deduplicate_by!r}."
                f" Choose from {sorted(self.SUPPORTED_KEYS)}."
            )

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


class ExampleRelabelService:
    """Apply label updates to matching DatasetExample records.

    Matching is additive — all supplied filters must hold simultaneously.
    Any filter left as ``None`` is treated as a wildcard.
    """

    def relabel(
        self,
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
        """Return (relabeled_examples, changed_count).

        Only examples that match the input filters are updated; others pass
        through unchanged.
        """

        updated: list[DatasetExample] = []
        changed = 0
        for example in examples:
            if not self._matches(
                example,
                trace_id=trace_id,
                source=source,
                outcome=input_outcome,
                quality=input_quality,
            ):
                updated.append(example)
                continue

            new_label = DatasetLabel(
                outcome=outcome or example.label.outcome,
                quality=quality or example.label.quality,
                failure_modes=(
                    failure_modes
                    if failure_modes is not None
                    else example.label.failure_modes
                ),
                reviewer_notes=(
                    reviewer_notes
                    if reviewer_notes is not None
                    else example.label.reviewer_notes
                ),
            )
            new_metadata = {
                **example.metadata,
                "review_required": False,
                "curated": True,
            }
            updated.append(replace(example, label=new_label, metadata=new_metadata))
            changed += 1
        return updated, changed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
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
