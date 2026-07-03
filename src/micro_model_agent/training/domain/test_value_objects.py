"""Tests for training domain value objects."""

from __future__ import annotations

from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)

# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_required_fields() -> None:
    config = TrainingConfig(base_model="tinyllama", output_dir="/tmp/run")
    assert config.base_model == "tinyllama"
    assert config.output_dir == "/tmp/run"
    assert config.dry_run is True  # default
    assert config.seed == 42


def test_training_config_optional_hyperparams_default_none() -> None:
    config = TrainingConfig(base_model="m", output_dir="/tmp")
    assert config.max_steps is None
    assert config.learning_rate is None
    assert config.batch_size is None
    assert config.gradient_accumulation_steps is None


def test_training_config_custom_hyperparams() -> None:
    config = TrainingConfig(
        base_model="m",
        output_dir="/tmp",
        max_steps=50,
        learning_rate=1e-4,
        batch_size=2,
        dry_run=False,
    )
    assert config.max_steps == 50
    assert config.learning_rate == 1e-4
    assert config.dry_run is False


def test_training_config_parameters_dict() -> None:
    config = TrainingConfig(
        base_model="m",
        output_dir="/tmp",
        parameters={"dataset_path": "/data/train.jsonl", "example_count": 100},
    )
    assert config.parameters["example_count"] == 100


def test_training_config_is_frozen() -> None:
    config = TrainingConfig(base_model="m", output_dir="/tmp")
    try:
        config.base_model = "other"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# ModelArtifact
# ---------------------------------------------------------------------------


def test_model_artifact_fields() -> None:
    artifact = ModelArtifact(
        name="synthetic-adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="/tmp/adapter",
        base_model="tinyllama",
    )
    assert artifact.name == "synthetic-adapter"
    assert artifact.kind == ModelArtifactKind.ADAPTER
    assert artifact.path == "/tmp/adapter"
    assert artifact.base_model == "tinyllama"
    assert artifact.id is not None
    assert artifact.metrics == {}
    assert artifact.metadata == {}


def test_model_artifact_kind_enum_values() -> None:
    assert ModelArtifactKind.ADAPTER.value == "adapter"
    assert ModelArtifactKind.MERGED_MODEL.value == "merged_model"
    assert ModelArtifactKind.OLLAMA_MODEL.value == "ollama_model"
    assert ModelArtifactKind.REPORT.value == "report"


def test_model_artifact_with_metrics() -> None:
    artifact = ModelArtifact(
        name="adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="/tmp",
        base_model="m",
        metrics={"loss": 0.42, "perplexity": 12.5},
    )
    assert artifact.metrics["loss"] == 0.42


def test_model_artifact_is_frozen() -> None:
    artifact = ModelArtifact(name="x", kind=ModelArtifactKind.ADAPTER, path="/p", base_model="m")
    try:
        artifact.name = "y"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# TrainingRun
# ---------------------------------------------------------------------------


def test_training_run_defaults() -> None:
    config = TrainingConfig(base_model="m", output_dir="/tmp")
    run = TrainingRun(kind=TrainingRunKind.SYNTHETIC, config=config)
    assert run.status == TrainingRunStatus.PENDING
    assert run.artifacts == ()
    assert run.metrics == {}
    assert run.error is None
    assert run.started_at is None
    assert run.finished_at is None


def test_training_run_status_enum_values() -> None:
    assert TrainingRunStatus.PENDING.value == "pending"
    assert TrainingRunStatus.RUNNING.value == "running"
    assert TrainingRunStatus.SUCCEEDED.value == "succeeded"
    assert TrainingRunStatus.FAILED.value == "failed"


def test_training_run_kind_enum_values() -> None:
    assert TrainingRunKind.SYNTHETIC.value == "synthetic"
    assert TrainingRunKind.TRACE_DERIVED.value == "trace_derived"
    assert TrainingRunKind.MIXED.value == "mixed"


def test_training_run_with_artifacts() -> None:
    config = TrainingConfig(base_model="m", output_dir="/tmp")
    artifact = ModelArtifact(name="a", kind=ModelArtifactKind.ADAPTER, path="/p", base_model="m")
    run = TrainingRun(
        kind=TrainingRunKind.SYNTHETIC,
        config=config,
        status=TrainingRunStatus.SUCCEEDED,
        artifacts=(artifact,),
        metrics={"loss": 0.5},
    )
    assert len(run.artifacts) == 1
    assert run.metrics["loss"] == 0.5


def test_training_run_is_frozen() -> None:
    config = TrainingConfig(base_model="m", output_dir="/tmp")
    run = TrainingRun(kind=TrainingRunKind.SYNTHETIC, config=config)
    try:
        run.status = TrainingRunStatus.RUNNING  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass
