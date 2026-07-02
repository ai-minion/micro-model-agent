"""Runtime training workflow composition helpers."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.application.training import RunSyntheticTrainingWorkflow
from micro_model_agent.infrastructure.datasets.metadata import (
    LocalDatasetFileHasher,
    LocalDatasetToolProfileSummarizer,
)
from micro_model_agent.infrastructure.datasets.validation import (
    LocalDatasetValidator,
    SftJsonlDatasetExporter,
)
from micro_model_agent.infrastructure.persistence.dataset_store import LocalDatasetExampleReader
from micro_model_agent.infrastructure.training.artifacts import (
    FakeTrainingRunner,
    JsonTrainingArtifactStore,
)
from micro_model_agent.infrastructure.training.local_finetuning import LocalFineTuningRunner

__all__ = ["build_synthetic_training_workflow"]


def build_synthetic_training_workflow(
    *,
    dry_run: bool,
    artifact_store_root: str | Path = Path(".micro_model_agent/training"),
) -> RunSyntheticTrainingWorkflow:
    """Build the standard synthetic training workflow for CLI entrypoints."""

    runner = FakeTrainingRunner() if dry_run else LocalFineTuningRunner()
    return RunSyntheticTrainingWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
        exporter=SftJsonlDatasetExporter(),
        file_hasher=LocalDatasetFileHasher(),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        runner=runner,
        artifact_store=JsonTrainingArtifactStore(artifact_store_root),
    )
