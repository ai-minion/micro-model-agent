"""Tests for training runtime composition helpers."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.training.infrastructure.artifacts import FakeTrainingRunner
from micro_model_agent.training.infrastructure.local_finetuning import LocalFineTuningRunner
from micro_model_agent.training.infrastructure.runtime import build_synthetic_training_workflow


def test_training_runtime_selects_runner_by_dry_run(tmp_path: Path) -> None:
    dry_run_workflow = build_synthetic_training_workflow(
        dry_run=True,
        artifact_store_root=tmp_path / "artifacts",
    )
    real_workflow = build_synthetic_training_workflow(
        dry_run=False,
        artifact_store_root=tmp_path / "artifacts",
    )

    assert isinstance(dry_run_workflow.runner, FakeTrainingRunner)
    assert isinstance(real_workflow.runner, LocalFineTuningRunner)
