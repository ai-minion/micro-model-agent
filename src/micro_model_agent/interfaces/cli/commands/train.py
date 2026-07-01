"""Training CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from micro_model_agent.application.training import (
    RunSyntheticTrainingRequest,
)
from micro_model_agent.infrastructure.composition import build_synthetic_training_workflow
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

    workflow = build_synthetic_training_workflow(dry_run=dry_run)
    result = _run(
        workflow.run(
            RunSyntheticTrainingRequest(
                base_model=base_model,
                dataset_path=dataset,
                output_dir=output_dir,
                dry_run=dry_run,
                max_steps=max_steps,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        )
    )
    if not result.evaluation.passed:
        typer.echo(result.evaluation.summary, err=True)
        raise typer.Exit(1)

    run = result.run
    if run is None:
        raise typer.Exit(1)
    typer.echo(
        f"Training run {run.id} completed with status {run.status.value}; "
        f"metadata written to {output_dir}"
    )
    if run.status.value == "failed":
        raise typer.Exit(1)
