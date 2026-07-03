"""Tests for dataset runtime composition helpers."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.dataset.application.workflows import (
    RunDatasetExportWorkflow,
    RunDatasetMergeWorkflow,
    RunDatasetRelabelWorkflow,
    RunDatasetSynthesisWorkflow,
    RunDatasetValidationWorkflow,
    RunTraceDatasetExportWorkflow,
    RunTraceReviewWorkflow,
)
from micro_model_agent.dataset.infrastructure.dataset_store import JsonlDatasetExampleStore
from micro_model_agent.dataset.infrastructure.runtime import (
    build_dataset_export_workflow,
    build_dataset_merge_workflow,
    build_dataset_relabel_workflow,
    build_dataset_synthesis_workflow,
    build_dataset_validation_workflow,
    build_jsonl_dataset_example_store,
    build_trace_dataset_export_workflow,
    build_trace_review_workflow,
)


def test_dataset_runtime_builds_standard_store_and_workflows(tmp_path: Path) -> None:
    assert isinstance(
        build_jsonl_dataset_example_store(tmp_path / "datasets" / "tasks.jsonl"),
        JsonlDatasetExampleStore,
    )
    assert isinstance(
        build_dataset_synthesis_workflow(template_dir=tmp_path),
        RunDatasetSynthesisWorkflow,
    )
    assert isinstance(build_dataset_validation_workflow(), RunDatasetValidationWorkflow)
    assert isinstance(build_dataset_export_workflow(), RunDatasetExportWorkflow)
    assert isinstance(build_dataset_relabel_workflow(), RunDatasetRelabelWorkflow)
    assert isinstance(build_dataset_merge_workflow(), RunDatasetMergeWorkflow)
    assert isinstance(
        build_trace_dataset_export_workflow(
            trace_path=tmp_path / "workflows.jsonl",
            review_path=tmp_path / "reviews.jsonl",
        ),
        RunTraceDatasetExportWorkflow,
    )
    assert isinstance(build_trace_review_workflow(), RunTraceReviewWorkflow)
