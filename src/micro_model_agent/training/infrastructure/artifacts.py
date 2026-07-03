"""Training artifact stores and lightweight training runners."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from micro_model_agent.training.infrastructure.training_records import (
    model_artifact_from_record,
    model_artifact_to_record,
    training_dataset_version,
    training_run_to_record,
)


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
