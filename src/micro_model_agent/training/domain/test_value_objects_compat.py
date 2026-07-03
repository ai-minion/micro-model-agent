"""Tests for training domain contracts."""

from __future__ import annotations

from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)


def test_training_run_defaults_to_pending_dry_run() -> None:
    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        output_dir=".micro_model_agent/training/runs/example",
    )

    run = TrainingRun(kind=TrainingRunKind.SYNTHETIC, config=config)

    assert run.status is TrainingRunStatus.PENDING
    assert run.config.dry_run is True


def test_training_run_can_reference_adapter_artifact() -> None:
    config = TrainingConfig(base_model="base", output_dir="out", dry_run=False)
    artifact = ModelArtifact(
        name="synthetic-adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="out/adapter",
        base_model="base",
        metrics={"schema_valid_json_rate": 1.0},
    )

    run = TrainingRun(
        kind=TrainingRunKind.SYNTHETIC,
        config=config,
        status=TrainingRunStatus.SUCCEEDED,
        artifacts=(artifact,),
    )

    assert run.artifacts[0].kind is ModelArtifactKind.ADAPTER
    assert run.artifacts[0].metrics["schema_valid_json_rate"] == 1.0
