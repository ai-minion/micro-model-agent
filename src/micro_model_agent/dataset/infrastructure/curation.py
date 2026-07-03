"""Helpers for curating and combining dataset examples."""

from __future__ import annotations

from micro_model_agent.dataset.domain.services import (
    DatasetMergeService,
    ExampleRelabelService,
)
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)


class LocalDatasetMerger:
    """Adapter implementing the DatasetMerger port via DatasetMergeService."""

    def __init__(self) -> None:
        self._service = DatasetMergeService()

    def merge_datasets(
        self,
        datasets: list[list[DatasetExample]],
        *,
        deduplicate_by: str,
    ) -> tuple[list[DatasetExample], int]:
        """Merge datasets with the requested dedupe key."""
        return self._service.merge(datasets, deduplicate_by=deduplicate_by)  # type: ignore[arg-type]


class LocalDatasetRelabeler:
    """Adapter implementing the DatasetRelabeler port via ExampleRelabelService."""

    def __init__(self) -> None:
        self._service = ExampleRelabelService()

    def relabel_examples(
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
        """Return examples with matching records relabeled."""
        return self._service.relabel(
            examples,
            trace_id=trace_id,
            source=source,
            input_outcome=input_outcome,
            input_quality=input_quality,
            outcome=outcome,
            quality=quality,
            failure_modes=failure_modes,
            reviewer_notes=reviewer_notes,
        )


# ---------------------------------------------------------------------------
# Module-level functions kept for backward compatibility with existing tests
# that import them directly.  New code should use the service classes.
# ---------------------------------------------------------------------------

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
    """Backward-compatible shim — delegates to ExampleRelabelService."""
    return ExampleRelabelService().relabel(
        examples,
        trace_id=trace_id,
        source=source,
        input_outcome=input_outcome,
        input_quality=input_quality,
        outcome=outcome,
        quality=quality,
        failure_modes=failure_modes,
        reviewer_notes=reviewer_notes,
    )


def merge_datasets(
    datasets: list[list[DatasetExample]],
    *,
    deduplicate_by: str = "source",
) -> tuple[list[DatasetExample], int]:
    """Backward-compatible shim — delegates to DatasetMergeService."""
    return DatasetMergeService().merge(datasets, deduplicate_by=deduplicate_by)  # type: ignore[arg-type]
