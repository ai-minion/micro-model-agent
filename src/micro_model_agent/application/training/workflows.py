"""Application workflows for local training operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from micro_model_agent.application.ports import (
    ArtifactStore,
    DatasetExampleReader,
    DatasetExporter,
    DatasetFileHasher,
    DatasetToolProfileSummarizer,
    DatasetValidator,
    TrainingRunner,
)
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.training import TrainingConfig, TrainingRun


@dataclass(frozen=True, slots=True)
class RunSyntheticTrainingRequest:
    """Request for training or dry-running on a synthetic dataset."""

    base_model: str
    dataset_path: Path
    output_dir: Path
    dry_run: bool = True
    max_steps: int = 20
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass(frozen=True, slots=True)
class RunSyntheticTrainingResult:
    """Result returned after a synthetic training workflow."""

    output_dir: Path
    training_dataset_path: Path
    evaluation: EvaluationResult
    run: TrainingRun | None
    saved_artifact_count: int


class RunSyntheticTrainingWorkflow:
    """Validate, export, train, and record artifacts for synthetic data."""

    def __init__(
        self,
        *,
        example_reader: DatasetExampleReader,
        validator: DatasetValidator,
        exporter: DatasetExporter,
        file_hasher: DatasetFileHasher,
        tool_profile_summarizer: DatasetToolProfileSummarizer,
        runner: TrainingRunner,
        artifact_store: ArtifactStore,
    ) -> None:
        self.example_reader = example_reader
        self.validator = validator
        self.exporter = exporter
        self.file_hasher = file_hasher
        self.tool_profile_summarizer = tool_profile_summarizer
        self.runner = runner
        self.artifact_store = artifact_store

    async def run(
        self,
        request: RunSyntheticTrainingRequest,
    ) -> RunSyntheticTrainingResult:
        """Run synthetic training or dry-run training."""

        examples = self.example_reader.load_dataset_examples(request.dataset_path)
        evaluation = await self.validator.validate(examples)
        training_dataset_path = request.output_dir / "synthetic.sft.jsonl"
        if not evaluation.passed:
            return RunSyntheticTrainingResult(
                output_dir=request.output_dir,
                training_dataset_path=training_dataset_path,
                evaluation=evaluation,
                run=None,
                saved_artifact_count=0,
            )

        self.exporter.export_dataset_examples(training_dataset_path, examples)
        config = TrainingConfig(
            base_model=request.base_model,
            output_dir=str(request.output_dir),
            max_steps=request.max_steps,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            dry_run=request.dry_run,
            parameters={
                "dataset_path": str(training_dataset_path),
                "source_dataset_path": str(request.dataset_path),
                "source_dataset_sha256": self.file_hasher.dataset_file_sha256(
                    request.dataset_path
                ),
                "training_dataset_sha256": self.file_hasher.dataset_file_sha256(
                    training_dataset_path
                ),
                "example_count": len(examples),
                "dataset_tool_profile": (
                    self.tool_profile_summarizer.summarize_dataset_tool_profiles(examples)
                ),
                "max_seq_length": request.max_seq_length,
                "lora_r": request.lora_r,
                "lora_alpha": request.lora_alpha,
                "lora_dropout": request.lora_dropout,
            },
        )
        run = await self.runner.run(config)
        for artifact in run.artifacts:
            await self.artifact_store.save(artifact)
        return RunSyntheticTrainingResult(
            output_dir=request.output_dir,
            training_dataset_path=training_dataset_path,
            evaluation=evaluation,
            run=run,
            saved_artifact_count=len(run.artifacts),
        )
