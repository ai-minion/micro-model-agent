"""Tests for build_event_pipeline and execution context composition."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from micro_model_agent.execution.infrastructure.composition import build_event_pipeline
from micro_model_agent.execution.infrastructure.repository import JsonlWorkflowRepository
from micro_model_agent.execution.domain.aggregate import WorkflowExecution
from micro_model_agent.execution.domain.events import WorkflowCompleted


def _run(coro):  # type: ignore[return]
    return asyncio.run(coro)


def test_build_event_pipeline_wires_workflow_completed(tmp_path: Path) -> None:
    """WorkflowCompleted → OnWorkflowCompleted → DatasetExample in default dataset."""
    trace_dir = tmp_path / "traces"
    dataset_root = tmp_path / "datasets"
    evaluation_root = tmp_path / "evaluation"
    promotion_root = tmp_path / "promotion"

    bus = build_event_pipeline(
        trace_dir=trace_dir,
        dataset_root=dataset_root,
        evaluation_root=evaluation_root,
        promotion_root=promotion_root,
    )

    # Persist an execution so the handler can load it.
    workflow_repo = JsonlWorkflowRepository(trace_dir / "traces.jsonl")
    execution = WorkflowExecution(goal="refactor auth module")
    execution.start()
    execution.complete({"ok": True, "patch_applied": True})
    _run(workflow_repo.add(execution))

    # Publish the event via the bus.
    event = WorkflowCompleted(execution_id=execution.id, output={"ok": True})
    _run(bus.publish(event))

    # The "default" dataset should now have one example.
    from micro_model_agent.dataset.infrastructure.repository import JsonlDatasetRepository
    dataset_repo = JsonlDatasetRepository(dataset_root)
    dataset = _run(dataset_repo.find_by_name("default"))
    assert dataset is not None
    assert dataset.size == 1
    assert dataset.examples[0].input["goal"] == "refactor auth module"


def test_build_event_pipeline_wires_artifact_produced(tmp_path: Path) -> None:
    """ArtifactProduced → OnArtifactProduced → EvaluationReport created."""
    from micro_model_agent.training.domain.events import ArtifactProduced
    from micro_model_agent.training.domain.value_objects import ModelArtifactKind
    from micro_model_agent.evaluation.infrastructure.repository import (
        JsonlEvaluationReportRepository,
    )

    trace_dir = tmp_path / "traces"
    dataset_root = tmp_path / "datasets"
    evaluation_root = tmp_path / "evaluation"
    promotion_root = tmp_path / "promotion"

    bus = build_event_pipeline(
        trace_dir=trace_dir,
        dataset_root=dataset_root,
        evaluation_root=evaluation_root,
        promotion_root=promotion_root,
    )

    job_id = uuid4()
    artifact_id = uuid4()
    event = ArtifactProduced(
        job_id=job_id,
        artifact_id=artifact_id,
        kind=ModelArtifactKind.ADAPTER,
    )
    _run(bus.publish(event))

    eval_repo = JsonlEvaluationReportRepository(evaluation_root)
    reports = _run(eval_repo.find_by_run_id(str(job_id)))
    assert len(reports) == 1
    assert reports[0].run_id == str(job_id)
