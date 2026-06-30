"""Training CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from micro_model_agent.domain.training import TrainingConfig
from micro_model_agent.infrastructure.dataset_metadata import (
    dataset_file_sha256,
    summarize_tool_profiles,
)
from micro_model_agent.infrastructure.dataset_store import load_dataset_examples
from micro_model_agent.infrastructure.dataset_validation import (
    LocalDatasetValidator,
    export_sft_jsonl,
)
from micro_model_agent.infrastructure.training_artifacts import (
    FakeTrainingRunner,
    JsonTrainingArtifactStore,
    LocalFineTuningRunner,
)
from micro_model_agent.interfaces.cli.common import _load_dotenv, _run


def register_train_commands(train_app: typer.Typer) -> None:
    """Register training command group handlers."""

    train_app.command("synthetic")(train_synthetic)


def train_synthetic(
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Local or Hugging Face base model id.",
    ),
    dataset: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="Validated synthetic dataset path.",
    ),
    output_dir: Path = typer.Option(
        Path(".micro_model_agent/training/runs/latest"),
        help="Training run output directory.",
    ),
    dry_run: bool = typer.Option(True, help="Validate config and write a dry-run artifact."),
    max_steps: int = typer.Option(20, min=1, help="Maximum optimizer steps for real training."),
    batch_size: int = typer.Option(1, min=1, help="Per-device train batch size."),
    gradient_accumulation_steps: int = typer.Option(
        4,
        min=1,
        help="Gradient accumulation steps for real training.",
    ),
    learning_rate: float = typer.Option(2e-4, min=0.0, help="Learning rate for real training."),
    max_seq_length: int = typer.Option(1024, min=128, help="Maximum tokenized sequence length."),
    lora_r: int = typer.Option(16, min=1, help="LoRA rank for real training."),
    lora_alpha: int = typer.Option(32, min=1, help="LoRA alpha for real training."),
    lora_dropout: float = typer.Option(0.05, min=0.0, max=1.0, help="LoRA dropout."),
) -> None:
    """Run or dry-run local synthetic-data fine-tuning."""

    _load_dotenv()

    examples = load_dataset_examples(dataset)
    validation = _run(LocalDatasetValidator().validate(examples))
    if not validation.passed:
        typer.echo(validation.summary, err=True)
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_dataset = output_dir / "synthetic.sft.jsonl"
    # Training backends consume SFT JSONL, so export a run-local copy first.
    export_sft_jsonl(training_dataset, examples)
    source_dataset_sha256 = dataset_file_sha256(dataset)
    training_dataset_sha256 = dataset_file_sha256(training_dataset)

    config = TrainingConfig(
        base_model=base_model,
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dry_run=dry_run,
        parameters={
            "dataset_path": str(training_dataset),
            "source_dataset_path": str(dataset),
            "source_dataset_sha256": source_dataset_sha256,
            "training_dataset_sha256": training_dataset_sha256,
            "example_count": len(examples),
            "dataset_tool_profile": summarize_tool_profiles(examples),
            "max_seq_length": max_seq_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    )
    # dry_run uses the fake runner; real training uses the hardware-dependent runner.
    runner = FakeTrainingRunner() if dry_run else LocalFineTuningRunner()
    run = _run(runner.run(config))
    artifact_store = JsonTrainingArtifactStore(Path(".micro_model_agent/training"))
    for artifact in run.artifacts:
        _run(artifact_store.save(artifact))

    typer.echo(
        f"Training run {run.id} completed with status {run.status.value}; "
        f"metadata written to {output_dir}"
    )
    if run.status.value == "failed":
        raise typer.Exit(1)
