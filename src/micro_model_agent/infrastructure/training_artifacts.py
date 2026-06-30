"""Local training artifact storage, fake training, and synthetic evaluation.

This module contains both lightweight dry-run training and the optional
Transformers/PEFT training path. The public runners return the same domain
objects so callers do not need to know which backend was used.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from micro_model_agent.domain.training import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from micro_model_agent.infrastructure.artifact_evaluation import SyntheticEvaluationSuite
from micro_model_agent.infrastructure.evaluation_reports import (
    LocalEvaluationResultReader,
    LocalEvaluationResultWriter,
    load_evaluation_result,
    write_evaluation_result,
)
from micro_model_agent.infrastructure.local_finetuning import (
    HuggingFacePeftFineTuningBackend,
    LocalFineTuningBackend,
    LocalFineTuningResult,
    LocalFineTuningRunner,
    _bool_parameter,
    _float_parameter,
    _int_parameter,
    _model_load_kwargs,
    _numeric_metrics,
    _target_modules,
    _training_text_from_record,
)
from micro_model_agent.infrastructure.promotion_gate import (
    LocalPromotionGateStore,
    MinimumScorePromotionPolicy,
    PromotionRegistryEntry,
    load_promotion_registry,
    promotion_registry_entry_from_record,
    promotion_registry_entry_to_record,
    record_promoted_artifact,
    write_promotion_gate_result,
)
from micro_model_agent.infrastructure.training_records import (
    load_artifact_from_training_run,
    model_artifact_from_record,
    model_artifact_to_record,
    training_dataset_version,
    training_run_to_record,
)

__all__ = [
    "FakeTrainingRunner",
    "HuggingFacePeftFineTuningBackend",
    "JsonTrainingArtifactStore",
    "LocalEvaluationResultReader",
    "LocalEvaluationResultWriter",
    "LocalFineTuningBackend",
    "LocalFineTuningResult",
    "LocalFineTuningRunner",
    "LocalPromotionGateStore",
    "MinimumScorePromotionPolicy",
    "PromotionRegistryEntry",
    "SyntheticEvaluationSuite",
    "load_artifact_from_training_run",
    "load_evaluation_result",
    "load_promotion_registry",
    "model_artifact_from_record",
    "model_artifact_to_record",
    "promotion_registry_entry_from_record",
    "promotion_registry_entry_to_record",
    "record_promoted_artifact",
    "training_run_to_record",
    "write_evaluation_result",
    "write_promotion_gate_result",
    "_bool_parameter",
    "_float_parameter",
    "_int_parameter",
    "_model_load_kwargs",
    "_numeric_metrics",
    "_target_modules",
    "_training_text_from_record",
]


class JsonTrainingArtifactStore:
    """Local JSON metadata store for training artifacts."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    async def save(self, artifact: ModelArtifact) -> None:
        path = self._artifact_path(str(artifact.id))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model_artifact_to_record(artifact), indent=2), encoding="utf-8")

    async def get(self, artifact_id: str) -> ModelArtifact | None:
        path = self._artifact_path(artifact_id)
        if not path.exists():
            return None
        return model_artifact_from_record(json.loads(path.read_text(encoding="utf-8")))

    def _artifact_path(self, artifact_id: str) -> Path:
        return self.root_dir / "artifacts" / f"{artifact_id}.json"


class FakeTrainingRunner:
    """Training runner that writes deterministic metadata without touching a GPU."""

    async def run(self, config: TrainingConfig) -> TrainingRun:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # The fake runner creates the same metadata shape as a real run, but it
        # does not load model weights or require a GPU.
        now = datetime.now(UTC)
        example_count = int(config.parameters.get("example_count", 0))
        artifact = ModelArtifact(
            name="synthetic-dry-run-adapter" if config.dry_run else "synthetic-fake-adapter",
            kind=ModelArtifactKind.ADAPTER,
            path=str(output_dir / "adapter"),
            base_model=config.base_model,
            metrics={
                "synthetic_example_count": float(example_count),
                "dry_run": 1.0 if config.dry_run else 0.0,
            },
            metadata={
                "dataset_path": config.parameters.get("dataset_path"),
                "source_dataset_path": config.parameters.get("source_dataset_path"),
                "source_dataset_sha256": config.parameters.get("source_dataset_sha256"),
                "training_dataset_sha256": config.parameters.get("training_dataset_sha256"),
                "dataset_tool_profile": config.parameters.get("dataset_tool_profile"),
                "runner": "fake",
            },
        )
        run = TrainingRun(
            kind=TrainingRunKind.SYNTHETIC,
            config=config,
            status=TrainingRunStatus.SUCCEEDED,
            dataset_version=training_dataset_version(config),
            artifacts=(artifact,),
            metrics=artifact.metrics,
            started_at=now,
            finished_at=datetime.now(UTC),
        )

        (output_dir / "artifact.json").write_text(
            json.dumps(model_artifact_to_record(artifact), indent=2),
            encoding="utf-8",
        )
        (output_dir / "run.json").write_text(
            json.dumps(training_run_to_record(run), indent=2),
            encoding="utf-8",
        )
        return run
