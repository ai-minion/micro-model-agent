"""Tests for dataset application workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.application.datasets import (
    RunDatasetExportRequest,
    RunDatasetExportWorkflow,
    RunDatasetMergeRequest,
    RunDatasetMergeWorkflow,
    RunDatasetRelabelRequest,
    RunDatasetRelabelWorkflow,
    RunDatasetSynthesisRequest,
    RunDatasetSynthesisWorkflow,
    RunDatasetValidationRequest,
    RunDatasetValidationWorkflow,
    RunTraceDatasetExportRequest,
    RunTraceDatasetExportWorkflow,
)
from micro_model_agent.domain.contracts import EvaluationResult, WorkflowStatus, WorkflowTrace
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)


class FakeDatasetReader:
    """In-memory dataset reader for application tests."""

    def __init__(self, examples: list[DatasetExample]) -> None:
        self.examples = examples
        self.loaded_path: Path | None = None
        self.examples_by_path: dict[Path, list[DatasetExample]] = {}
        self.loaded_paths: list[Path] = []

    def load_dataset_examples(self, path: Path) -> list[DatasetExample]:
        self.loaded_path = path
        self.loaded_paths.append(path)
        return self.examples_by_path.get(path, self.examples)


class FakeDatasetValidator:
    """In-memory dataset validator for application tests."""

    def __init__(self, evaluation: EvaluationResult) -> None:
        self.evaluation = evaluation
        self.validated_examples: list[DatasetExample] | None = None

    async def validate(self, examples: list[DatasetExample]) -> EvaluationResult:
        self.validated_examples = examples
        return self.evaluation


class FakeDatasetExporter:
    """In-memory dataset exporter for application tests."""

    def __init__(self) -> None:
        self.exported: tuple[Path, list[DatasetExample]] | None = None

    def export_dataset_examples(self, path: Path, examples: list[DatasetExample]) -> None:
        self.exported = (path, examples)


class FakeDatasetMerger:
    """In-memory dataset merger for application tests."""

    def __init__(self) -> None:
        self.merged_inputs: tuple[list[list[DatasetExample]], str] | None = None

    def merge_datasets(
        self,
        datasets: list[list[DatasetExample]],
        *,
        deduplicate_by: str,
    ) -> tuple[list[DatasetExample], int]:
        self.merged_inputs = (datasets, deduplicate_by)
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


class FakeDatasetWriter:
    """In-memory dataset writer for application tests."""

    def __init__(self) -> None:
        self.saved: tuple[Path, list[DatasetExample]] | None = None

    async def save_dataset_examples(
        self,
        path: Path,
        examples: list[DatasetExample],
    ) -> None:
        self.saved = (path, examples)


class FakeSyntheticGenerator:
    """In-memory synthetic generator for application tests."""

    def __init__(self, examples: list[DatasetExample]) -> None:
        self.examples = examples
        self.request: dict[str, object] | None = None

    async def generate(
        self,
        count: int,
        *,
        seed: int | None = None,
        balance_categories: bool = True,
        vary_scenarios: bool = True,
        include_categories: tuple[str, ...] = (),
        exclude_categories: tuple[str, ...] = (),
    ) -> list[DatasetExample]:
        self.request = {
            "count": count,
            "seed": seed,
            "balance_categories": balance_categories,
            "vary_scenarios": vary_scenarios,
            "include_categories": include_categories,
            "exclude_categories": exclude_categories,
        }
        return self.examples


class FakeDatasetRelabeler:
    """In-memory dataset relabeler for application tests."""

    def __init__(self, relabeled: list[DatasetExample], changed: int) -> None:
        self.relabeled = relabeled
        self.changed = changed
        self.request: dict[str, object] | None = None

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
        self.request = {
            "examples": examples,
            "trace_id": trace_id,
            "source": source,
            "input_outcome": input_outcome,
            "input_quality": input_quality,
            "outcome": outcome,
            "quality": quality,
            "failure_modes": failure_modes,
            "reviewer_notes": reviewer_notes,
        }
        return self.relabeled, self.changed


class FakeWorkflowTraceReader:
    """In-memory workflow trace reader for application tests."""

    def __init__(self, traces: list[WorkflowTrace]) -> None:
        self.traces = traces
        self.called = False

    async def list_workflow_traces(self) -> list[WorkflowTrace]:
        self.called = True
        return self.traces


class FakeTraceReviewReader:
    """In-memory trace review reader for application tests."""

    def __init__(self, reviews_by_trace_id: dict[str, object]) -> None:
        self.reviews_by_trace_id = reviews_by_trace_id
        self.called = False

    async def latest_trace_reviews_by_trace_id(self) -> dict[str, object]:
        self.called = True
        return self.reviews_by_trace_id


class FakeTraceDatasetExporter:
    """In-memory trace dataset exporter for application tests."""

    def __init__(self, examples: list[DatasetExample]) -> None:
        self.examples = examples
        self.request: dict[str, object] | None = None

    def export_trace_dataset_examples(
        self,
        traces: list[WorkflowTrace],
        *,
        kind: DatasetExampleKind,
        label_mode: str,
        reviews_by_trace_id: dict[str, object],
        outcome: OutcomeLabel | None = None,
        quality: QualityLabel | None = None,
        workflow_status: WorkflowStatus | None = None,
        require_tool_call: bool = False,
        max_examples: int | None = None,
    ) -> list[DatasetExample]:
        self.request = {
            "traces": traces,
            "kind": kind,
            "label_mode": label_mode,
            "reviews_by_trace_id": reviews_by_trace_id,
            "outcome": outcome,
            "quality": quality,
            "workflow_status": workflow_status,
            "require_tool_call": require_tool_call,
            "max_examples": max_examples,
        }
        return self.examples


class FakeTraceDatasetExportValidator:
    """In-memory trace export validator for application tests."""

    def __init__(self, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        self.validated_examples: list[DatasetExample] | None = None

    def validate_trace_dataset_examples(self, examples: list[DatasetExample]) -> list[str]:
        self.validated_examples = examples
        return self.errors


def test_dataset_synthesis_workflow_generates_validates_and_saves() -> None:
    examples = [_example()]
    generator = FakeSyntheticGenerator(examples)
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=True,
            summary="validated 1 examples with 0 error(s)",
            score=1.0,
            details={},
        )
    )
    writer = FakeDatasetWriter()
    workflow = RunDatasetSynthesisWorkflow(
        generator=generator,
        validator=validator,
        example_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetSynthesisRequest(
                count=3,
                output_path=Path("synthetic.jsonl"),
                seed=41,
                balance_categories=False,
                vary_scenarios=False,
                include_categories=("trace_patch_training",),
                exclude_categories=("unsafe",),
            )
        )
    )

    assert result.example_count == 1
    assert result.saved is True
    assert result.evaluation.passed is True
    assert generator.request == {
        "count": 3,
        "seed": 41,
        "balance_categories": False,
        "vary_scenarios": False,
        "include_categories": ("trace_patch_training",),
        "exclude_categories": ("unsafe",),
    }
    assert validator.validated_examples == examples
    assert writer.saved == (Path("synthetic.jsonl"), examples)


def test_dataset_synthesis_workflow_skips_save_when_validation_fails() -> None:
    examples = [_example()]
    generator = FakeSyntheticGenerator(examples)
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=False,
            summary="validated 1 examples with 1 error(s)",
            score=0.0,
            details={"errors": ["bad example"]},
        )
    )
    writer = FakeDatasetWriter()
    workflow = RunDatasetSynthesisWorkflow(
        generator=generator,
        validator=validator,
        example_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetSynthesisRequest(
                count=1,
                output_path=Path("synthetic.jsonl"),
            )
        )
    )

    assert result.saved is False
    assert writer.saved is None


def test_dataset_validation_workflow_loads_and_validates_examples() -> None:
    examples = [_example()]
    reader = FakeDatasetReader(examples)
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=True,
            summary="validated 1 examples with 0 error(s)",
            score=1.0,
            details={"example_count": 1},
        )
    )
    workflow = RunDatasetValidationWorkflow(
        example_reader=reader,
        validator=validator,
    )

    result = asyncio.run(
        workflow.run(RunDatasetValidationRequest(path=Path("synthetic.jsonl")))
    )

    assert result.path == Path("synthetic.jsonl")
    assert result.evaluation.passed is True
    assert reader.loaded_path == Path("synthetic.jsonl")
    assert validator.validated_examples == examples


def test_dataset_validation_workflow_returns_failed_evaluation() -> None:
    reader = FakeDatasetReader([])
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=False,
            summary="validated 0 examples with 1 error(s)",
            score=0.0,
            details={"errors": ["dataset contains no examples"]},
        )
    )
    workflow = RunDatasetValidationWorkflow(
        example_reader=reader,
        validator=validator,
    )

    result = asyncio.run(
        workflow.run(RunDatasetValidationRequest(path=Path("empty.jsonl")))
    )

    assert result.evaluation.passed is False
    assert result.evaluation.details["errors"] == ["dataset contains no examples"]


def test_dataset_export_workflow_exports_when_validation_passes() -> None:
    examples = [_example()]
    reader = FakeDatasetReader(examples)
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=True,
            summary="validated 1 examples with 0 error(s)",
            score=1.0,
            details={},
        )
    )
    exporter = FakeDatasetExporter()
    workflow = RunDatasetExportWorkflow(
        example_reader=reader,
        validator=validator,
        exporter=exporter,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetExportRequest(
                path=Path("synthetic.jsonl"),
                output_path=Path("synthetic.sft.jsonl"),
            )
        )
    )

    assert result.example_count == 1
    assert result.output_format == "sft-jsonl"
    assert result.evaluation.passed is True
    assert exporter.exported == (Path("synthetic.sft.jsonl"), examples)


def test_dataset_export_workflow_skips_export_when_validation_fails() -> None:
    examples = [_example()]
    reader = FakeDatasetReader(examples)
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=False,
            summary="validated 1 examples with 1 error(s)",
            score=0.0,
            details={"errors": ["bad example"]},
        )
    )
    exporter = FakeDatasetExporter()
    workflow = RunDatasetExportWorkflow(
        example_reader=reader,
        validator=validator,
        exporter=exporter,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetExportRequest(
                path=Path("synthetic.jsonl"),
                output_path=Path("synthetic.sft.jsonl"),
            )
        )
    )

    assert result.evaluation.passed is False
    assert exporter.exported is None


def test_dataset_export_workflow_rejects_unsupported_format() -> None:
    workflow = RunDatasetExportWorkflow(
        example_reader=FakeDatasetReader([_example()]),
        validator=FakeDatasetValidator(
            EvaluationResult(passed=True, summary="ok", score=1.0)
        ),
        exporter=FakeDatasetExporter(),
    )

    try:
        asyncio.run(
            workflow.run(
                RunDatasetExportRequest(
                    path=Path("synthetic.jsonl"),
                    output_path=Path("synthetic.csv"),
                    output_format="csv",
                )
            )
        )
    except ValueError as exc:
        assert str(exc) == "Unsupported dataset export format: csv"
    else:
        raise AssertionError("expected ValueError")


def test_dataset_merge_workflow_merges_validates_and_saves() -> None:
    first = _example(source="synthetic:one")
    duplicate = _example(source="synthetic:one")
    second = _example(source="synthetic:two")
    reader = FakeDatasetReader([])
    reader.examples_by_path = {
        Path("one.jsonl"): [first],
        Path("two.jsonl"): [duplicate, second],
    }
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=True,
            summary="validated 2 examples with 0 error(s)",
            score=1.0,
            details={},
        )
    )
    writer = FakeDatasetWriter()
    merger = FakeDatasetMerger()
    workflow = RunDatasetMergeWorkflow(
        example_reader=reader,
        merger=merger,
        validator=validator,
        example_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetMergeRequest(
                input_paths=(Path("one.jsonl"), Path("two.jsonl")),
                output_path=Path("merged.jsonl"),
            )
        )
    )

    assert reader.loaded_paths == [Path("one.jsonl"), Path("two.jsonl")]
    assert merger.merged_inputs is not None
    _, deduplicate_by = merger.merged_inputs
    assert deduplicate_by == "source"
    assert result.merged_count == 2
    assert result.skipped_duplicates == 1
    assert result.saved is True
    assert writer.saved == (Path("merged.jsonl"), [first, second])
    assert validator.validated_examples == [first, second]


def test_dataset_merge_workflow_blocks_save_when_validation_fails() -> None:
    examples = [_example()]
    reader = FakeDatasetReader(examples)
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=False,
            summary="validated 1 examples with 1 error(s)",
            score=0.0,
            details={"errors": ["bad example"]},
        )
    )
    writer = FakeDatasetWriter()
    workflow = RunDatasetMergeWorkflow(
        example_reader=reader,
        merger=FakeDatasetMerger(),
        validator=validator,
        example_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetMergeRequest(
                input_paths=(Path("one.jsonl"),),
                output_path=Path("merged.jsonl"),
            )
        )
    )

    assert result.saved is False
    assert writer.saved is None


def test_dataset_merge_workflow_can_save_invalid_when_validation_disabled() -> None:
    examples = [_example()]
    reader = FakeDatasetReader(examples)
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=False,
            summary="validated 1 examples with 1 error(s)",
            score=0.0,
            details={"errors": ["bad example"]},
        )
    )
    writer = FakeDatasetWriter()
    workflow = RunDatasetMergeWorkflow(
        example_reader=reader,
        merger=FakeDatasetMerger(),
        validator=validator,
        example_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetMergeRequest(
                input_paths=(Path("one.jsonl"),),
                output_path=Path("merged.jsonl"),
                validate_training_ready=False,
            )
        )
    )

    assert result.saved is True
    assert writer.saved == (Path("merged.jsonl"), examples)


def test_dataset_merge_workflow_rejects_unsupported_dedupe_key() -> None:
    workflow = RunDatasetMergeWorkflow(
        example_reader=FakeDatasetReader([]),
        merger=FakeDatasetMerger(),
        validator=FakeDatasetValidator(EvaluationResult(passed=True, summary="ok")),
        example_writer=FakeDatasetWriter(),
    )

    try:
        asyncio.run(
            workflow.run(
                RunDatasetMergeRequest(
                    input_paths=(Path("one.jsonl"),),
                    output_path=Path("merged.jsonl"),
                    deduplicate_by="category",
                )
            )
        )
    except ValueError as exc:
        assert str(exc) == "Unsupported dedupe key: category"
    else:
        raise AssertionError("expected ValueError")


def test_dataset_relabel_workflow_relabels_saves_and_validates() -> None:
    original = [_example(source="trace:abc")]
    relabeled = [_example(source="trace:abc")]
    reader = FakeDatasetReader(original)
    relabeler = FakeDatasetRelabeler(relabeled, changed=1)
    writer = FakeDatasetWriter()
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=True,
            summary="validated 1 examples with 0 error(s)",
            score=1.0,
            details={},
        )
    )
    workflow = RunDatasetRelabelWorkflow(
        example_reader=reader,
        relabeler=relabeler,
        example_writer=writer,
        validator=validator,
    )

    result = asyncio.run(
        workflow.run(
            RunDatasetRelabelRequest(
                path=Path("trace.jsonl"),
                output_path=Path("curated.jsonl"),
                trace_id="abc",
                input_outcome=OutcomeLabel.NEEDS_REVIEW,
                input_quality=QualityLabel.UNKNOWN,
                outcome=OutcomeLabel.ACCEPTED,
                quality=QualityLabel.GOOD,
                failure_modes=(FailureMode.WRONG_TOOL_SELECTED,),
                reviewer_notes="Reviewed.",
            )
        )
    )

    assert result.original_count == 1
    assert result.changed_count == 1
    assert relabeler.request is not None
    assert relabeler.request["examples"] == original
    assert relabeler.request["trace_id"] == "abc"
    assert relabeler.request["outcome"] is OutcomeLabel.ACCEPTED
    assert relabeler.request["quality"] is QualityLabel.GOOD
    assert relabeler.request["failure_modes"] == (FailureMode.WRONG_TOOL_SELECTED,)
    assert relabeler.request["reviewer_notes"] == "Reviewed."
    assert writer.saved == (Path("curated.jsonl"), relabeled)
    assert validator.validated_examples == relabeled


def test_dataset_relabel_workflow_requires_label_update() -> None:
    workflow = RunDatasetRelabelWorkflow(
        example_reader=FakeDatasetReader([_example()]),
        relabeler=FakeDatasetRelabeler([], changed=0),
        example_writer=FakeDatasetWriter(),
        validator=FakeDatasetValidator(EvaluationResult(passed=True, summary="ok")),
    )

    try:
        asyncio.run(
            workflow.run(
                RunDatasetRelabelRequest(
                    path=Path("trace.jsonl"),
                    output_path=Path("curated.jsonl"),
                )
            )
        )
    except ValueError as exc:
        assert str(exc) == "At least one label update is required"
    else:
        raise AssertionError("expected ValueError")


def test_trace_dataset_export_workflow_exports_valid_examples() -> None:
    trace = WorkflowTrace(goal="Fix a test", status=WorkflowStatus.SUCCEEDED)
    example = _example(
        source=f"trace:{trace.id}",
        metadata={"trace_id": str(trace.id), "redacted": True},
    )
    review = object()
    trace_reader = FakeWorkflowTraceReader([trace])
    review_reader = FakeTraceReviewReader({str(trace.id): review})
    exporter = FakeTraceDatasetExporter([example])
    validator = FakeTraceDatasetExportValidator()
    writer = FakeDatasetWriter()
    workflow = RunTraceDatasetExportWorkflow(
        trace_reader=trace_reader,
        review_reader=review_reader,
        trace_exporter=exporter,
        trace_export_validator=validator,
        example_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunTraceDatasetExportRequest(
                output_path=Path("trace_examples.jsonl"),
                label_mode="reviewed",
                kind=DatasetExampleKind.REPAIR,
                outcome=OutcomeLabel.ACCEPTED,
                quality=QualityLabel.GOOD,
                workflow_status=WorkflowStatus.SUCCEEDED,
                require_tool_call=True,
                max_examples=5,
            )
        )
    )

    assert result.saved is True
    assert result.example_count == 1
    assert result.traces_read == 1
    assert result.reviews_read == 1
    assert result.redacted_count == 1
    assert result.outcome_counts["accepted"] == 1
    assert writer.saved == (Path("trace_examples.jsonl"), [example])
    assert validator.validated_examples == [example]
    assert exporter.request is not None
    assert exporter.request["traces"] == [trace]
    assert exporter.request["label_mode"] == "reviewed"
    assert exporter.request["reviews_by_trace_id"] == {str(trace.id): review}
    assert exporter.request["workflow_status"] is WorkflowStatus.SUCCEEDED
    assert exporter.request["require_tool_call"] is True
    assert exporter.request["max_examples"] == 5


def test_trace_dataset_export_workflow_skips_save_when_validation_fails() -> None:
    trace = WorkflowTrace(goal="Fix a test")
    example = _example(source=f"trace:{trace.id}", metadata={"trace_id": str(trace.id)})
    writer = FakeDatasetWriter()
    workflow = RunTraceDatasetExportWorkflow(
        trace_reader=FakeWorkflowTraceReader([trace]),
        review_reader=FakeTraceReviewReader({}),
        trace_exporter=FakeTraceDatasetExporter([example]),
        trace_export_validator=FakeTraceDatasetExportValidator(["bad trace export"]),
        example_writer=writer,
    )

    result = asyncio.run(
        workflow.run(RunTraceDatasetExportRequest(output_path=Path("trace_examples.jsonl")))
    )

    assert result.saved is False
    assert result.validation_errors == ("bad trace export",)
    assert writer.saved is None


def _example(
    source: str = "synthetic:test",
    metadata: dict[str, object] | None = None,
) -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "Summarize app.py"},
        target={"final_response": "app.py summarized."},
        source=source,
        metadata=metadata or {},
        label=DatasetLabel(
            outcome=OutcomeLabel.ACCEPTED,
            quality=QualityLabel.GOOD,
        ),
    )
