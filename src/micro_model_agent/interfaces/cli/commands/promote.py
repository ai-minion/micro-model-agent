"""Promotion CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from micro_model_agent.promotion.application.workflows import (
    RunPromotionGateRequest,
    RunPromotionListRequest,
    RunPromotionPackageOllamaRequest,
    RunPromotionRecordRequest,
    RunPromotionSelectRequest,
)
from micro_model_agent.infrastructure.composition import (
    build_promotion_gate_workflow,
    build_promotion_list_workflow,
    build_promotion_package_ollama_workflow,
    build_promotion_record_workflow,
    build_promotion_select_workflow,
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

    workflow = build_promotion_gate_workflow()
    try:
        result = _run(
            workflow.run(
                RunPromotionGateRequest(
                    run_id=run_id,
                    evaluation_report_paths=tuple(evaluation_report or ()),
                    minimum_score=minimum_score,
                )
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    if result.promoted:
        typer.echo(
            f"Promotion gate passed for {result.artifact.name}; "
            f"report written to {result.output_path}"
        )
        return

    typer.echo(
        f"Promotion gate blocked for {result.artifact.name}; "
        f"report written to {result.output_path}",
        err=True,
    )
    for evaluated in result.evaluations:
        if not evaluated.can_promote:
            score = (
                "none"
                if evaluated.evaluation.score is None
                else f"{evaluated.evaluation.score:.2f}"
            )
            typer.echo(
                f"{evaluated.report_path}: passed={evaluated.evaluation.passed} "
                f"score={score} summary={evaluated.evaluation.summary}",
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

    workflow = build_promotion_record_workflow()
    try:
        result = _run(
            workflow.run(
                RunPromotionRecordRequest(
                    run_id=run_id,
                    registry_path=registry,
                    promotion_report_path=promotion_report,
                    reviewer_notes=reviewer_notes,
                    approved_by=approved_by,
                )
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    typer.echo(
        f"Recorded promoted artifact {result.record.artifact_name} "
        f"({result.record.artifact_id}) in {registry}"
    )


def promote_list(
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
) -> None:
    """List locally recorded promoted artifacts."""

    workflow = build_promotion_list_workflow()
    result = _run(workflow.run(RunPromotionListRequest(registry_path=registry)))
    if not result.records:
        typer.echo(f"No promoted artifacts recorded in {registry}")
        return

    for entry in result.records:
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

    workflow = build_promotion_select_workflow()
    try:
        result = _run(
            workflow.run(
                RunPromotionSelectRequest(
                    artifact_id=artifact_id,
                    repository_root=repository_root,
                    registry_path=registry,
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    typer.echo(
        f"Selected promoted adapter {result.record.artifact_name} "
        f"({result.record.artifact_id}) in {result.config_path}"
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

    workflow = build_promotion_package_ollama_workflow()
    try:
        result = _run(
            workflow.run(
                RunPromotionPackageOllamaRequest(
                    artifact_id=artifact_id,
                    registry_path=registry,
                    model_name=model_name,
                    output_dir=output_dir,
                    ollama_base_model=ollama_base_model,
                    create=create,
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    package = result.package
    typer.echo(f"Wrote Ollama Modelfile to {package.modelfile_path}")
    typer.echo(f"Wrote package manifest to {package.manifest_path}")
    typer.echo("Command: " + " ".join(package.command))
    for warning in package.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if create:
        if package.created:
            typer.echo(f"Created Ollama model {package.model_name}")
            return
        typer.echo(
            f"ollama create failed with exit code {package.return_code}",
            err=True,
        )
        if package.stderr:
            typer.echo(package.stderr, err=True)
        raise typer.Exit(1)
