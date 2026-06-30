"""Promotion CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from micro_model_agent.infrastructure.ollama_packaging import (
    package_promoted_adapter_for_ollama,
)
from micro_model_agent.infrastructure.repository_metadata import update_model_configuration
from micro_model_agent.infrastructure.training_artifacts import (
    MinimumScorePromotionPolicy,
    load_artifact_from_training_run,
    load_evaluation_result,
    load_promotion_registry,
    record_promoted_artifact,
    write_promotion_gate_result,
)
from micro_model_agent.interfaces.cli.common import _fail, _run


def register_promote_commands(promote_app: typer.Typer) -> None:
    """Register promotion command group handlers."""

    promote_app.command("gate")(promote_gate)
    promote_app.command("record")(promote_record)
    promote_app.command("list")(promote_list)
    promote_app.command("select")(promote_select)
    promote_app.command("package-ollama")(promote_package_ollama)


def promote_gate(
    run_id: str = typer.Option("latest", help="Training run id or direct run directory path."),
    evaluation_report: list[Path] | None = typer.Option(
        None,
        "--evaluation-report",
        help="Evaluation JSON report to require. Can be passed more than once.",
    ),
    minimum_score: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum score each evaluation report must meet.",
    ),
) -> None:
    """Gate promotion on one or more persisted evaluation reports."""

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    artifact = load_artifact_from_training_run(run_dir)
    report_paths = list(evaluation_report or [run_dir / "evaluation.json"])
    if not report_paths:
        _fail("at least one evaluation report is required")

    policy = MinimumScorePromotionPolicy(minimum_score=minimum_score)
    evaluated_reports = []
    for report_path in report_paths:
        evaluation = load_evaluation_result(report_path)
        can_promote = _run(policy.can_promote(artifact, evaluation))
        evaluated_reports.append((report_path, evaluation, can_promote))

    promoted = all(can_promote for _, _, can_promote in evaluated_reports)
    output = write_promotion_gate_result(
        run_dir,
        promoted=promoted,
        minimum_score=minimum_score,
        evaluation_reports=evaluated_reports,
    )

    if promoted:
        typer.echo(f"Promotion gate passed for {artifact.name}; report written to {output}")
        return

    typer.echo(f"Promotion gate blocked for {artifact.name}; report written to {output}", err=True)
    for report_path, evaluation, can_promote in evaluated_reports:
        if not can_promote:
            score = "none" if evaluation.score is None else f"{evaluation.score:.2f}"
            typer.echo(
                f"{report_path}: passed={evaluation.passed} score={score} "
                f"summary={evaluation.summary}",
                err=True,
            )
    raise typer.Exit(1)


def promote_record(
    run_id: str = typer.Option("latest", help="Training run id or direct run directory path."),
    promotion_report: Path | None = typer.Option(
        None,
        "--promotion-report",
        help="Passing promotion report path. Defaults to <run>/promotion.json.",
    ),
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
    reviewer_notes: str | None = typer.Option(
        None,
        "--reviewer-notes",
        help="Human review note to store with the registry entry.",
    ),
    approved_by: str | None = typer.Option(
        None,
        "--approved-by",
        help="Reviewer or process that approved this artifact.",
    ),
) -> None:
    """Record a gate-passing artifact in the local promotion registry."""

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    artifact = load_artifact_from_training_run(run_dir)
    report_path = promotion_report or run_dir / "promotion.json"
    entry = record_promoted_artifact(
        registry,
        artifact=artifact,
        run_dir=run_dir,
        promotion_report_path=report_path,
        reviewer_notes=reviewer_notes,
        approved_by=approved_by,
    )
    typer.echo(
        f"Recorded promoted artifact {entry.artifact_name} "
        f"({entry.artifact_id}) in {registry}"
    )


def promote_list(
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
) -> None:
    """List locally recorded promoted artifacts."""

    entries = load_promotion_registry(registry)
    if not entries:
        typer.echo(f"No promoted artifacts recorded in {registry}")
        return

    for entry in entries:
        typer.echo(
            f"{entry.artifact_name} {entry.artifact_id} "
            f"score>={entry.minimum_score:.2f} path={entry.artifact_path}"
        )


def promote_select(
    artifact_id: str = typer.Option(
        ...,
        "--artifact-id",
        help="Promoted artifact id from the local registry.",
    ),
    repository_root: Path = typer.Option(
        Path("."),
        "--repository-root",
        help="Repository root whose local model defaults should be updated.",
    ),
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Required explicit confirmation before changing local defaults.",
    ),
) -> None:
    """Select a recorded promoted artifact as the repository-local default adapter."""

    if not confirm:
        _fail("Selecting a promoted adapter changes local defaults; rerun with --confirm")

    entries = load_promotion_registry(registry)
    entry = next((item for item in entries if str(item.artifact_id) == artifact_id), None)
    if entry is None:
        _fail(f"promoted artifact id not found in {registry}: {artifact_id}")

    result = update_model_configuration(
        repository_root,
        base_model=entry.base_model,
        adapter_path=entry.artifact_path,
        selected_promotion={
            "artifact_id": str(entry.artifact_id),
            "artifact_name": entry.artifact_name,
            "promotion_report_path": entry.promotion_report_path,
            "registry_path": str(registry),
            "minimum_score": entry.minimum_score,
            "approved_by": entry.approved_by,
            "created_at": entry.created_at.isoformat(),
        },
    )
    if not result.ok:
        _fail(result.error or "failed to update MicroModelAgent config")

    typer.echo(
        f"Selected promoted adapter {entry.artifact_name} "
        f"({entry.artifact_id}) in {result.config_path}"
    )


def promote_package_ollama(
    artifact_id: str = typer.Option(
        ...,
        "--artifact-id",
        help="Promoted artifact id from the local registry.",
    ),
    model_name: str = typer.Option(
        ...,
        "--model-name",
        help="Ollama model name to create, such as micro-agent-proof:qwen.",
    ),
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for the generated Modelfile and package manifest.",
    ),
    ollama_base_model: str | None = typer.Option(
        None,
        "--ollama-base-model",
        help="Ollama FROM value. Defaults to the artifact training base model.",
    ),
    create: bool = typer.Option(
        False,
        "--create",
        help="Run `ollama create` after writing the Modelfile.",
    ),
) -> None:
    """Package a recorded promoted adapter for Ollama with a generated Modelfile."""

    entries = load_promotion_registry(registry)
    entry = next((item for item in entries if str(item.artifact_id) == artifact_id), None)
    if entry is None:
        _fail(f"promoted artifact id not found in {registry}: {artifact_id}")

    package_dir = output_dir or Path(".micro_model_agent/training/ollama") / _path_safe_name(
        model_name
    )
    result = package_promoted_adapter_for_ollama(
        entry=entry,
        model_name=model_name,
        output_dir=package_dir,
        ollama_base_model=ollama_base_model,
        create=create,
    )

    typer.echo(f"Wrote Ollama Modelfile to {result.modelfile_path}")
    typer.echo(f"Wrote package manifest to {result.manifest_path}")
    typer.echo("Command: " + " ".join(result.command))
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if create:
        if result.created:
            typer.echo(f"Created Ollama model {result.model_name}")
            return
        typer.echo(f"ollama create failed with exit code {result.return_code}", err=True)
        if result.stderr:
            typer.echo(result.stderr, err=True)
        raise typer.Exit(1)


def _path_safe_name(value: str) -> str:
    """Return a conservative directory name for a model tag."""

    safe = "".join(character if character.isalnum() else "-" for character in value.lower())
    return "-".join(part for part in safe.split("-") if part) or "ollama-model"
