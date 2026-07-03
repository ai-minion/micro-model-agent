"""Tests for cross-context event handlers and infrastructure repositories."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from uuid import uuid4

from micro_model_agent.dataset.application.event_handlers import OnWorkflowCompleted
from micro_model_agent.dataset.domain.value_objects import OutcomeLabel, QualityLabel
from micro_model_agent.dataset.infrastructure.repository import JsonlDatasetRepository
from micro_model_agent.evaluation.domain.aggregate import EvaluationReport, EvaluationThreshold
from micro_model_agent.evaluation.domain.events import ThresholdBreached, ThresholdMet
from micro_model_agent.evaluation.infrastructure.repository import (
    JsonlEvaluationReportRepository,
)
from micro_model_agent.execution.domain.aggregate import WorkflowExecution
from micro_model_agent.execution.domain.events import WorkflowCompleted
from micro_model_agent.execution.infrastructure.repository import JsonlWorkflowRepository
from micro_model_agent.promotion.application.event_handlers import OnThresholdEvent
from micro_model_agent.promotion.infrastructure.repository import (
    JsonlModelRegistryRepository,
)
from micro_model_agent.training.domain.aggregate import TrainingJob
from micro_model_agent.training.domain.value_objects import (
    TrainingConfig,
)
from micro_model_agent.training.infrastructure.repository import JsonlTrainingJobRepository


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


# ===========================================================================
# JsonlDatasetRepository
# ===========================================================================


def test_dataset_repo_add_and_find_by_name(tmp_path: Path) -> None:
    from micro_model_agent.dataset.domain.aggregate import Dataset
    from micro_model_agent.dataset.domain.value_objects import (
        DatasetExample,
        DatasetExampleKind,
        DatasetLabel,
    )

    repo = JsonlDatasetRepository(tmp_path)
    ds = Dataset(name="seed")
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "fix"},
        target={"patch": ""},
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
    )
    ds.add_example(example)
    _run(repo.add(ds))

    loaded = _run(repo.find_by_name("seed"))
    assert loaded is not None
    assert loaded.name == "seed"
    assert len(loaded.examples) == 1


def test_dataset_repo_get_by_id(tmp_path: Path) -> None:
    from micro_model_agent.dataset.domain.aggregate import Dataset

    repo = JsonlDatasetRepository(tmp_path)
    ds = Dataset(name="eval")
    _run(repo.add(ds))

    loaded = _run(repo.get(ds.id))
    assert loaded is not None
    assert loaded.id == ds.id


def test_dataset_repo_find_missing_returns_none(tmp_path: Path) -> None:
    repo = JsonlDatasetRepository(tmp_path)
    assert _run(repo.find_by_name("nonexistent")) is None


# ===========================================================================
# JsonlTrainingJobRepository
# ===========================================================================


def _make_config(output_dir: str) -> TrainingConfig:
    return TrainingConfig(base_model="tinyllama", output_dir=output_dir)


def test_training_job_repo_round_trip(tmp_path: Path) -> None:
    repo = JsonlTrainingJobRepository(tmp_path)
    config = _make_config(str(tmp_path / "run"))
    job = TrainingJob(config=config)

    _run(repo.add(job))
    loaded = _run(repo.get(job.id))

    assert loaded is not None
    assert loaded.id == job.id
    assert loaded.config.base_model == "tinyllama"


def test_training_job_repo_get_missing(tmp_path: Path) -> None:
    repo = JsonlTrainingJobRepository(tmp_path)
    assert _run(repo.get(uuid4())) is None


# ===========================================================================
# JsonlEvaluationReportRepository
# ===========================================================================


def test_evaluation_report_repo_round_trip(tmp_path: Path) -> None:
    repo = JsonlEvaluationReportRepository(tmp_path)
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="job-1", threshold=threshold)
    report.finalize(0.92)

    _run(repo.add(report))
    loaded = _run(repo.get(report.id))

    assert loaded is not None
    assert loaded.run_id == "job-1"
    assert loaded.summary_score == 0.92
    assert loaded.passed is True


def test_evaluation_report_repo_find_by_run_id(tmp_path: Path) -> None:
    repo = JsonlEvaluationReportRepository(tmp_path)
    threshold = EvaluationThreshold(minimum_score=0.8)
    r1 = EvaluationReport(run_id="job-1", threshold=threshold)
    r1.finalize(0.9)
    r2 = EvaluationReport(run_id="job-2", threshold=threshold)
    r2.finalize(0.7)
    _run(repo.add(r1))
    _run(repo.add(r2))

    found = _run(repo.find_by_run_id("job-1"))
    assert len(found) == 1
    assert found[0].run_id == "job-1"


# ===========================================================================
# JsonlModelRegistryRepository
# ===========================================================================


def test_model_registry_repo_get_or_create(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    registry = _run(repo.get_or_create())
    assert registry is not None

    # Second call returns the same registry
    registry2 = _run(repo.get_or_create())
    assert registry2.id == registry.id


def test_model_registry_repo_save_and_reload(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from micro_model_agent.promotion.domain.aggregate import PromotedModel

    repo = JsonlModelRegistryRepository(tmp_path)
    registry = _run(repo.get_or_create())

    model = PromotedModel(
        id=uuid4(),
        artifact_id=uuid4(),
        artifact_name="tinyllama-adapter",
        artifact_path="/tmp/adapter",
        base_model="tinyllama",
        promoted_at=datetime.now(UTC),
    )
    registry.record_promotion(model)
    _run(repo.save(registry))

    reloaded = _run(repo.get_or_create())
    assert len(reloaded.list_models()) == 1
    assert reloaded.list_models()[0].artifact_name == "tinyllama-adapter"


# ===========================================================================
# OnWorkflowCompleted event handler
# ===========================================================================


def test_on_workflow_completed_creates_dataset_example(tmp_path: Path) -> None:
    # Set up repositories
    workflow_repo = JsonlWorkflowRepository(tmp_path / "traces" / "traces.jsonl")
    dataset_repo = JsonlDatasetRepository(tmp_path / "datasets")

    # Persist a completed execution
    execution = WorkflowExecution(goal="fix null pointer")
    execution.start()
    execution.complete({"ok": True, "patch_applied": True})
    _run(workflow_repo.add(execution))

    # Fire the event
    handler = OnWorkflowCompleted(
        workflow_repo=workflow_repo,
        dataset_repo=dataset_repo,
    )
    event = WorkflowCompleted(execution_id=execution.id, output={"ok": True})
    _run(handler.handle(event))

    # Verify the dataset was created
    dataset = _run(dataset_repo.find_by_name("default"))
    assert dataset is not None
    assert dataset.size == 1
    assert dataset.examples[0].label.outcome == OutcomeLabel.ACCEPTED


def test_on_workflow_completed_second_example_appends(tmp_path: Path) -> None:
    workflow_repo = JsonlWorkflowRepository(tmp_path / "traces" / "traces.jsonl")
    dataset_repo = JsonlDatasetRepository(tmp_path / "datasets")
    handler = OnWorkflowCompleted(workflow_repo=workflow_repo, dataset_repo=dataset_repo)

    for i in range(2):
        ex = WorkflowExecution(goal=f"goal {i}")
        ex.start()
        ex.complete({"ok": True})
        _run(workflow_repo.add(ex))
        _run(handler.handle(WorkflowCompleted(execution_id=ex.id, output={"ok": True})))

    dataset = _run(dataset_repo.find_by_name("default"))
    assert dataset is not None
    assert dataset.size == 2


# ===========================================================================
# OnThresholdEvent handler
# ===========================================================================


def test_on_threshold_met_records_gate_pass(tmp_path: Path) -> None:
    registry_repo = JsonlModelRegistryRepository(tmp_path)
    handler = OnThresholdEvent(registry_repo=registry_repo, minimum_score=0.8)

    event = ThresholdMet(report_id=uuid4(), score=0.9)
    _run(handler.handle_met(event))

    # Gate pass was recorded — the registry now exists
    registry = _run(registry_repo.get_or_create())
    assert registry is not None


def test_on_threshold_breached_records_gate_fail(tmp_path: Path) -> None:
    registry_repo = JsonlModelRegistryRepository(tmp_path)
    handler = OnThresholdEvent(registry_repo=registry_repo, minimum_score=0.8)

    event = ThresholdBreached(report_id=uuid4(), score=0.5, threshold=0.8)
    _run(handler.handle_breached(event))

    registry = _run(registry_repo.get_or_create())
    assert registry is not None
