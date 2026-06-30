"""Application workflows for dataset operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from micro_model_agent.application.ports import (
    DatasetExampleReader,
    DatasetExampleWriter,
    DatasetExporter,
    DatasetMerger,
    DatasetRelabeler,
    DatasetValidator,
    SyntheticDataGenerator,
    TraceDatasetExampleExporter,
    TraceDatasetExportValidator,
    TraceReviewReader,
    WorkflowTraceReader,
)
from micro_model_agent.domain.contracts import EvaluationResult, WorkflowStatus
from micro_model_agent.domain.datasets import (
    DatasetExampleKind,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)


@dataclass(frozen=True, slots=True)
class RunDatasetValidationRequest:
    """Request for validating persisted dataset examples."""

    path: Path


@dataclass(frozen=True, slots=True)
class RunDatasetSynthesisRequest:
    """Request for generating and saving synthetic dataset examples."""

    count: int
    output_path: Path
    seed: int | None = None
    balance_categories: bool = True
    vary_scenarios: bool = True
    include_categories: tuple[str, ...] = ()
    exclude_categories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunDatasetSynthesisResult:
    """Result returned after synthetic dataset generation."""

    output_path: Path
    example_count: int
    evaluation: EvaluationResult
    saved: bool


@dataclass(frozen=True, slots=True)
class RunDatasetValidationResult:
    """Result returned after validating dataset examples."""

    path: Path
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class RunDatasetExportRequest:
    """Request for validating and exporting persisted dataset examples."""

    path: Path
    output_path: Path
    output_format: str = "sft-jsonl"


@dataclass(frozen=True, slots=True)
class RunDatasetExportResult:
    """Result returned after exporting dataset examples."""

    path: Path
    output_path: Path
    output_format: str
    example_count: int
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class RunDatasetMergeRequest:
    """Request for merging persisted dataset files."""

    input_paths: tuple[Path, ...]
    output_path: Path
    deduplicate_by: str = "source"
    validate_training_ready: bool = True


@dataclass(frozen=True, slots=True)
class RunDatasetMergeResult:
    """Result returned after merging dataset files."""

    input_paths: tuple[Path, ...]
    output_path: Path
    merged_count: int
    skipped_duplicates: int
    evaluation: EvaluationResult
    saved: bool


@dataclass(frozen=True, slots=True)
class RunDatasetRelabelRequest:
    """Request for relabeling matching dataset examples."""

    path: Path
    output_path: Path
    trace_id: str | None = None
    source: str | None = None
    input_outcome: OutcomeLabel | None = None
    input_quality: QualityLabel | None = None
    outcome: OutcomeLabel | None = None
    quality: QualityLabel | None = None
    failure_modes: tuple[FailureMode, ...] | None = None
    reviewer_notes: str | None = None


@dataclass(frozen=True, slots=True)
class RunDatasetRelabelResult:
    """Result returned after relabeling dataset examples."""

    path: Path
    output_path: Path
    original_count: int
    changed_count: int
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class RunTraceDatasetExportRequest:
    """Request for exporting stored traces as dataset examples."""

    output_path: Path
    label_mode: str = "review"
    kind: DatasetExampleKind = DatasetExampleKind.REPAIR
    outcome: OutcomeLabel | None = None
    quality: QualityLabel | None = None
    workflow_status: WorkflowStatus | None = None
    require_tool_call: bool = False
    max_examples: int | None = None


@dataclass(frozen=True, slots=True)
class RunTraceDatasetExportResult:
    """Result returned after exporting trace-derived dataset examples."""

    output_path: Path
    example_count: int
    traces_read: int
    reviews_read: int
    redacted_count: int
    outcome_counts: dict[str, int]
    validation_errors: tuple[str, ...]
    saved: bool


class RunDatasetSynthesisWorkflow:
    """Generate synthetic dataset examples, validate them, and save when valid."""

    def __init__(
        self,
        *,
        generator: SyntheticDataGenerator,
        validator: DatasetValidator,
        example_writer: DatasetExampleWriter,
    ) -> None:
        self.generator = generator
        self.validator = validator
        self.example_writer = example_writer

    async def run(self, request: RunDatasetSynthesisRequest) -> RunDatasetSynthesisResult:
        """Run synthetic data generation."""

        examples = await self.generator.generate(
            request.count,
            seed=request.seed,
            balance_categories=request.balance_categories,
            vary_scenarios=request.vary_scenarios,
            include_categories=request.include_categories,
            exclude_categories=request.exclude_categories,
        )
        evaluation = await self.validator.validate(examples)
        if evaluation.passed:
            await self.example_writer.save_dataset_examples(request.output_path, examples)
        return RunDatasetSynthesisResult(
            output_path=request.output_path,
            example_count=len(examples),
            evaluation=evaluation,
            saved=evaluation.passed,
        )


class RunDatasetValidationWorkflow:
    """Load dataset examples and validate them."""

    def __init__(
        self,
        *,
        example_reader: DatasetExampleReader,
        validator: DatasetValidator,
    ) -> None:
        self.example_reader = example_reader
        self.validator = validator

    async def run(self, request: RunDatasetValidationRequest) -> RunDatasetValidationResult:
        """Run dataset validation for a persisted dataset."""

        examples = self.example_reader.load_dataset_examples(request.path)
        evaluation = await self.validator.validate(examples)
        return RunDatasetValidationResult(path=request.path, evaluation=evaluation)


class RunDatasetExportWorkflow:
    """Validate persisted dataset examples and export them."""

    def __init__(
        self,
        *,
        example_reader: DatasetExampleReader,
        validator: DatasetValidator,
        exporter: DatasetExporter,
    ) -> None:
        self.example_reader = example_reader
        self.validator = validator
        self.exporter = exporter

    async def run(self, request: RunDatasetExportRequest) -> RunDatasetExportResult:
        """Run dataset export after validation succeeds."""

        if request.output_format != "sft-jsonl":
            raise ValueError(f"Unsupported dataset export format: {request.output_format}")

        examples = self.example_reader.load_dataset_examples(request.path)
        evaluation = await self.validator.validate(examples)
        if evaluation.passed:
            self.exporter.export_dataset_examples(request.output_path, examples)
        return RunDatasetExportResult(
            path=request.path,
            output_path=request.output_path,
            output_format=request.output_format,
            example_count=len(examples),
            evaluation=evaluation,
        )


class RunDatasetMergeWorkflow:
    """Merge persisted dataset files with validation and saving."""

    def __init__(
        self,
        *,
        example_reader: DatasetExampleReader,
        merger: DatasetMerger,
        validator: DatasetValidator,
        example_writer: DatasetExampleWriter,
    ) -> None:
        self.example_reader = example_reader
        self.merger = merger
        self.validator = validator
        self.example_writer = example_writer

    async def run(self, request: RunDatasetMergeRequest) -> RunDatasetMergeResult:
        """Merge dataset examples and write them when validation policy allows it."""

        if request.deduplicate_by not in {"source", "id"}:
            raise ValueError(f"Unsupported dedupe key: {request.deduplicate_by}")

        datasets = [
            self.example_reader.load_dataset_examples(path) for path in request.input_paths
        ]
        merged, skipped = self.merger.merge_datasets(
            datasets,
            deduplicate_by=request.deduplicate_by,
        )
        evaluation = await self.validator.validate(merged)
        saved = not (request.validate_training_ready and not evaluation.passed)
        if saved:
            await self.example_writer.save_dataset_examples(request.output_path, merged)
        return RunDatasetMergeResult(
            input_paths=request.input_paths,
            output_path=request.output_path,
            merged_count=len(merged),
            skipped_duplicates=skipped,
            evaluation=evaluation,
            saved=saved,
        )


class RunDatasetRelabelWorkflow:
    """Relabel matching dataset examples, save them, and validate the result."""

    def __init__(
        self,
        *,
        example_reader: DatasetExampleReader,
        relabeler: DatasetRelabeler,
        example_writer: DatasetExampleWriter,
        validator: DatasetValidator,
    ) -> None:
        self.example_reader = example_reader
        self.relabeler = relabeler
        self.example_writer = example_writer
        self.validator = validator

    async def run(self, request: RunDatasetRelabelRequest) -> RunDatasetRelabelResult:
        """Run dataset relabeling for matching records."""

        if not any(
            [
                request.outcome,
                request.quality,
                request.failure_modes,
                request.reviewer_notes is not None,
            ]
        ):
            raise ValueError("At least one label update is required")

        examples = self.example_reader.load_dataset_examples(request.path)
        relabeled, changed = self.relabeler.relabel_examples(
            examples,
            trace_id=request.trace_id,
            source=request.source,
            input_outcome=request.input_outcome,
            input_quality=request.input_quality,
            outcome=request.outcome,
            quality=request.quality,
            failure_modes=request.failure_modes,
            reviewer_notes=request.reviewer_notes,
        )
        await self.example_writer.save_dataset_examples(request.output_path, relabeled)
        evaluation = await self.validator.validate(relabeled)
        return RunDatasetRelabelResult(
            path=request.path,
            output_path=request.output_path,
            original_count=len(examples),
            changed_count=changed,
            evaluation=evaluation,
        )


class RunTraceDatasetExportWorkflow:
    """Export stored workflow traces as validated dataset examples."""

    def __init__(
        self,
        *,
        trace_reader: WorkflowTraceReader,
        review_reader: TraceReviewReader,
        trace_exporter: TraceDatasetExampleExporter,
        trace_export_validator: TraceDatasetExportValidator,
        example_writer: DatasetExampleWriter,
    ) -> None:
        self.trace_reader = trace_reader
        self.review_reader = review_reader
        self.trace_exporter = trace_exporter
        self.trace_export_validator = trace_export_validator
        self.example_writer = example_writer

    async def run(
        self,
        request: RunTraceDatasetExportRequest,
    ) -> RunTraceDatasetExportResult:
        """Run trace-to-dataset export."""

        traces = await self.trace_reader.list_workflow_traces()
        reviews_by_trace_id = await self.review_reader.latest_trace_reviews_by_trace_id()
        examples = self.trace_exporter.export_trace_dataset_examples(
            traces,
            kind=request.kind,
            label_mode=request.label_mode,
            reviews_by_trace_id=reviews_by_trace_id,
            outcome=request.outcome,
            quality=request.quality,
            workflow_status=request.workflow_status,
            require_tool_call=request.require_tool_call,
            max_examples=request.max_examples,
        )
        errors = self.trace_export_validator.validate_trace_dataset_examples(examples)
        saved = not errors
        if saved:
            await self.example_writer.save_dataset_examples(request.output_path, examples)

        return RunTraceDatasetExportResult(
            output_path=request.output_path,
            example_count=len(examples),
            traces_read=len(traces),
            reviews_read=len(reviews_by_trace_id),
            redacted_count=sum(1 for example in examples if example.metadata.get("redacted")),
            outcome_counts={
                outcome.value: sum(
                    1 for example in examples if example.label.outcome is outcome
                )
                for outcome in OutcomeLabel
            },
            validation_errors=tuple(errors),
            saved=saved,
        )
