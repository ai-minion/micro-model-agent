"""Tests for training application workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.application.training import (
    RunSyntheticTrainingRequest,
    RunSyntheticTrainingWorkflow,
)
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.domain.training import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)


class FakeDatasetReader:
    """In-memory dataset reader for training workflow tests."""

    def __init__(self, examples: list[DatasetExample]) -> None:
        self.examples = examples
        self.loaded_path: Path | None = None

    def load_dataset_examples(self, path: Path) -> list[DatasetExample]:
        self.loaded_path = path
        return self.examples


class FakeDatasetValidator:
    """In-memory validator for training workflow tests."""

    def __init__(self, evaluation: EvaluationResult) -> None:
        self.evaluation = evaluation
        self.validated_examples: list[DatasetExample] | None = None

    async def validate(self, examples: list[DatasetExample]) -> EvaluationResult:
        self.validated_examples = examples
        return self.evaluation


class FakeDatasetExporter:
    """In-memory exporter for training workflow tests."""

    def __init__(self) -> None:
        self.exported: tuple[Path, list[DatasetExample]] | None = None

    def export_dataset_examples(self, path: Path, examples: list[DatasetExample]) -> None:
        self.exported = (path, examples)


class FakeDatasetFileHasher:
    """Deterministic dataset file hasher for training workflow tests."""

    def __init__(self) -> None:
        self.hashed_paths: list[Path] = []

    def dataset_file_sha256(self, path: Path) -> str:
        self.hashed_paths.append(path)
        return f"sha256:{path.name}"


class FakeDatasetToolProfileSummarizer:
    """Deterministic tool-profile summarizer for training workflow tests."""

    def __init__(self) -> None:
        self.examples: list[DatasetExample] | None = None

    def summarize_dataset_tool_profiles(
        self,
        examples: list[DatasetExample],
        *,
        default_available_tools: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        self.examples = examples
        return {"example_count": len(examples), "tools_used": ["repo.read"]}


class FakeTrainingRunner:
    """In-memory training runner for workflow tests."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self.config: TrainingConfig | None = None

    async def run(self, config: TrainingConfig) -> TrainingRun:
        self.config = config
        return TrainingRun(
            kind=TrainingRunKind.SYNTHETIC,
            config=config,
            status=TrainingRunStatus.SUCCEEDED,
            artifacts=(self.artifact,),
        )


class FakeArtifactStore:
    """In-memory artifact store for training workflow tests."""

    def __init__(self) -> None:
        self.saved: list[ModelArtifact] = []

    async def save(self, artifact: ModelArtifact) -> None:
        self.saved.append(artifact)

    async def get(self, artifact_id: str) -> ModelArtifact | None:
        for artifact in self.saved:
            if str(artifact.id) == artifact_id:
                return artifact
        return None


def test_synthetic_training_workflow_validates_exports_runs_and_saves_artifacts() -> None:
    examples = [_example()]
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=True,
            summary="validated 1 examples with 0 error(s)",
            score=1.0,
        )
    )
    exporter = FakeDatasetExporter()
    hasher = FakeDatasetFileHasher()
    summarizer = FakeDatasetToolProfileSummarizer()
    artifact = ModelArtifact(
        name="adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="runs/latest/adapter",
        base_model="base-model",
    )
    runner = FakeTrainingRunner(artifact)
    artifact_store = FakeArtifactStore()
    workflow = RunSyntheticTrainingWorkflow(
        example_reader=FakeDatasetReader(examples),
        validator=validator,
        exporter=exporter,
        file_hasher=hasher,
        tool_profile_summarizer=summarizer,
        runner=runner,
        artifact_store=artifact_store,
    )

    result = asyncio.run(
        workflow.run(
            RunSyntheticTrainingRequest(
                base_model="base-model",
                dataset_path=Path("datasets/synthetic.jsonl"),
                output_dir=Path("runs/latest"),
                dry_run=False,
                max_steps=7,
                batch_size=2,
                gradient_accumulation_steps=3,
                learning_rate=1e-4,
                max_seq_length=2048,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
        )
    )

    training_dataset_path = Path("runs/latest/synthetic.sft.jsonl")
    assert result.evaluation.passed is True
    assert result.training_dataset_path == training_dataset_path
    assert result.run is not None
    assert result.saved_artifact_count == 1
    assert exporter.exported == (training_dataset_path, examples)
    assert hasher.hashed_paths == [Path("datasets/synthetic.jsonl"), training_dataset_path]
    assert summarizer.examples == examples
    assert artifact_store.saved == [artifact]
    assert runner.config is not None
    assert runner.config.base_model == "base-model"
    assert runner.config.output_dir == "runs/latest"
    assert runner.config.max_steps == 7
    assert runner.config.learning_rate == 1e-4
    assert runner.config.batch_size == 2
    assert runner.config.gradient_accumulation_steps == 3
    assert runner.config.dry_run is False
    assert runner.config.parameters == {
        "dataset_path": str(training_dataset_path),
        "source_dataset_path": "datasets/synthetic.jsonl",
        "source_dataset_sha256": "sha256:synthetic.jsonl",
        "training_dataset_sha256": "sha256:synthetic.sft.jsonl",
        "example_count": 1,
        "dataset_tool_profile": {"example_count": 1, "tools_used": ["repo.read"]},
        "max_seq_length": 2048,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }


def test_synthetic_training_workflow_stops_when_validation_fails() -> None:
    examples = [_example()]
    validator = FakeDatasetValidator(
        EvaluationResult(
            passed=False,
            summary="validated 1 examples with 1 error(s)",
            score=0.0,
        )
    )
    exporter = FakeDatasetExporter()
    hasher = FakeDatasetFileHasher()
    summarizer = FakeDatasetToolProfileSummarizer()
    runner = FakeTrainingRunner(
        ModelArtifact(
            name="adapter",
            kind=ModelArtifactKind.ADAPTER,
            path="runs/latest/adapter",
            base_model="base-model",
        )
    )
    artifact_store = FakeArtifactStore()
    workflow = RunSyntheticTrainingWorkflow(
        example_reader=FakeDatasetReader(examples),
        validator=validator,
        exporter=exporter,
        file_hasher=hasher,
        tool_profile_summarizer=summarizer,
        runner=runner,
        artifact_store=artifact_store,
    )

    result = asyncio.run(
        workflow.run(
            RunSyntheticTrainingRequest(
                base_model="base-model",
                dataset_path=Path("datasets/synthetic.jsonl"),
                output_dir=Path("runs/latest"),
            )
        )
    )

    assert result.evaluation.passed is False
    assert result.run is None
    assert result.saved_artifact_count == 0
    assert result.training_dataset_path == Path("runs/latest/synthetic.sft.jsonl")
    assert exporter.exported is None
    assert hasher.hashed_paths == []
    assert summarizer.examples is None
    assert runner.config is None
    assert artifact_store.saved == []


def _example() -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "Fix tests", "available_tools": ["repo.read"]},
        target={"final_response": "Tests fixed."},
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        source="synthetic:test",
    )
