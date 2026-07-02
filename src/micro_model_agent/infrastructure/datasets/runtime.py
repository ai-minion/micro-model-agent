"""Runtime dataset and trace-dataset workflow composition helpers."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.application.datasets import (
    RunDatasetExportWorkflow,
    RunDatasetMergeWorkflow,
    RunDatasetRelabelWorkflow,
    RunDatasetSynthesisWorkflow,
    RunDatasetValidationWorkflow,
    RunTraceDatasetExportWorkflow,
    RunTraceReviewWorkflow,
)
from micro_model_agent.application.ports import DatasetExampleStore
from micro_model_agent.infrastructure.datasets.curation import (
    LocalDatasetMerger,
    LocalDatasetRelabeler,
)
from micro_model_agent.infrastructure.datasets.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.datasets.validation import (
    LocalDatasetValidator,
    SftJsonlDatasetExporter,
)
from micro_model_agent.infrastructure.persistence.dataset_store import (
    JsonlDatasetExampleStore,
    LocalDatasetExampleReader,
    LocalDatasetExampleWriter,
)
from micro_model_agent.infrastructure.persistence.trace_store import LocalWorkflowTraceReader
from micro_model_agent.infrastructure.traces.export import (
    LocalTraceDatasetExporter,
    LocalTraceDatasetExportValidator,
)
from micro_model_agent.infrastructure.traces.review import (
    LocalTraceReviewReader,
    LocalTraceReviewWriter,
)


def build_jsonl_dataset_example_store(path: str | Path) -> DatasetExampleStore:
    """Build the local JSONL dataset store used by coding task capture."""

    return JsonlDatasetExampleStore(path)


def build_dataset_synthesis_workflow(
    *,
    template_dir: str | Path,
) -> RunDatasetSynthesisWorkflow:
    """Build the standard synthetic dataset generation workflow."""

    return RunDatasetSynthesisWorkflow(
        generator=SyntheticTemplateGenerator(template_dir),
        validator=LocalDatasetValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )


def build_dataset_validation_workflow() -> RunDatasetValidationWorkflow:
    """Build the standard persisted dataset validation workflow."""

    return RunDatasetValidationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
    )


def build_dataset_export_workflow() -> RunDatasetExportWorkflow:
    """Build the standard dataset export workflow."""

    return RunDatasetExportWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
        exporter=SftJsonlDatasetExporter(),
    )


def build_trace_dataset_export_workflow(
    *,
    trace_path: str | Path,
    review_path: str | Path,
) -> RunTraceDatasetExportWorkflow:
    """Build the standard trace-to-dataset export workflow."""

    return RunTraceDatasetExportWorkflow(
        trace_reader=LocalWorkflowTraceReader(trace_path),
        review_reader=LocalTraceReviewReader(review_path),
        trace_exporter=LocalTraceDatasetExporter(),
        trace_export_validator=LocalTraceDatasetExportValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )


def build_trace_review_workflow() -> RunTraceReviewWorkflow:
    """Build the standard human trace review recording workflow."""

    return RunTraceReviewWorkflow(review_writer=LocalTraceReviewWriter())


def build_dataset_relabel_workflow() -> RunDatasetRelabelWorkflow:
    """Build the standard dataset relabeling workflow."""

    return RunDatasetRelabelWorkflow(
        example_reader=LocalDatasetExampleReader(),
        relabeler=LocalDatasetRelabeler(),
        example_writer=LocalDatasetExampleWriter(),
        validator=LocalDatasetValidator(),
    )


def build_dataset_merge_workflow() -> RunDatasetMergeWorkflow:
    """Build the standard dataset merge workflow."""

    return RunDatasetMergeWorkflow(
        example_reader=LocalDatasetExampleReader(),
        merger=LocalDatasetMerger(),
        validator=LocalDatasetValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )
