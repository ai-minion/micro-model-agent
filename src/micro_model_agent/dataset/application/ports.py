"""Dataset context application ports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    DatasetSplit,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.execution.domain.value_objects import WorkflowStatus, WorkflowTrace
from micro_model_agent.shared.domain.value_objects import EvaluationResult


@dataclass(frozen=True, slots=True)
class TraceReviewRecord:
    """Application-owned record for one human trace review."""

    trace_id: str
    label: DatasetLabel
    corrected_target: dict[str, Any] | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class TraceReviewReader(Protocol):
    async def latest_trace_reviews_by_trace_id(self) -> Mapping[str, object]: ...


class TraceReviewWriter(Protocol):
    async def save_trace_review(self, path: Path, review: TraceReviewRecord) -> None: ...


class TraceDatasetExampleExporter(Protocol):
    def export_trace_dataset_examples(
        self,
        traces: list[WorkflowTrace],
        *,
        kind: DatasetExampleKind,
        label_mode: str,
        reviews_by_trace_id: Mapping[str, object],
        outcome: OutcomeLabel | None = None,
        quality: QualityLabel | None = None,
        workflow_status: WorkflowStatus | None = None,
        require_tool_call: bool = False,
        max_examples: int | None = None,
    ) -> list[DatasetExample]: ...


class TraceDatasetExportValidator(Protocol):
    def validate_trace_dataset_examples(self, examples: list[DatasetExample]) -> list[str]: ...


class DatasetExampleStore(Protocol):
    async def save(self, example: DatasetExample) -> None: ...
    async def list(self, kind: str | None = None) -> list[DatasetExample]: ...


class DatasetExampleReader(Protocol):
    def load_dataset_examples(self, path: Path) -> list[DatasetExample]: ...


class DatasetExampleWriter(Protocol):
    async def save_dataset_examples(self, path: Path, examples: list[DatasetExample]) -> None: ...


class DatasetMerger(Protocol):
    def merge_datasets(
        self,
        datasets: list[list[DatasetExample]],
        *,
        deduplicate_by: str,
    ) -> tuple[list[DatasetExample], int]: ...


class DatasetRelabeler(Protocol):
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
    ) -> tuple[list[DatasetExample], int]: ...


class DatasetExporter(Protocol):
    def export_dataset_examples(self, path: Path, examples: list[DatasetExample]) -> None: ...


class SyntheticDataGenerator(Protocol):
    async def generate(
        self,
        count: int,
        *,
        seed: int | None = None,
        balance_categories: bool = True,
        vary_scenarios: bool = True,
        include_categories: tuple[str, ...] = (),
        exclude_categories: tuple[str, ...] = (),
    ) -> list[DatasetExample]: ...


class DatasetValidator(Protocol):
    async def validate(self, examples: list[DatasetExample]) -> EvaluationResult: ...


class DatasetFileHasher(Protocol):
    def dataset_file_sha256(self, path: Path) -> str: ...


class DatasetToolProfileSummarizer(Protocol):
    def summarize_dataset_tool_profiles(
        self,
        examples: list[DatasetExample],
        *,
        default_available_tools: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...


class DatasetBuilder(Protocol):
    async def build_split(self, examples: list[DatasetExample], name: str) -> DatasetSplit: ...
