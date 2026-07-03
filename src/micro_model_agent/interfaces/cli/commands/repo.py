"""Root repository and task CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from micro_model_agent.dataset.domain.value_objects import FailureMode, OutcomeLabel, QualityLabel
from micro_model_agent.execution.application.ports import CodingAgentTask
from micro_model_agent.execution.application.workflows import label_from_workflow_result
from micro_model_agent.interfaces.cli.common import (
    _fail,
    _format_count_distribution,
    _run,
)
from micro_model_agent.interfaces.composition import (
    build_coding_workflow,
    build_event_pipeline,
    build_jsonl_dataset_example_store,
    initialize_local_repository,
    trace_dir,
    write_local_repository_index,
)


def register_repo_commands(app: typer.Typer) -> None:
    """Register root repository and coding-task commands."""

    app.command()(init)
    app.command()(index)
    app.command()(task)


def init(
    repository_root: Path = typer.Option(Path("."), help="Repository root to initialize."),
    default_model: str | None = typer.Option(
        None,
        "--default-model",
        help="Optional default Ollama model name to record in local metadata.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Optional Transformers base model to record in local metadata.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Optional local PEFT adapter path to record in local metadata.",
    ),
) -> None:
    """Initialize MicroModelAgent metadata for the current repository."""

    result = initialize_local_repository(
        repository_root,
        default_model=default_model,
        base_model=base_model,
        adapter_path=adapter_path,
    )
    if not result.ok:
        _fail(result.error or "MicroModelAgent repository initialization failed")

    action = "Already initialized" if result.already_initialized else "Initialized"
    typer.echo(f"{action} MicroModelAgent metadata at {result.metadata_dir}")
    typer.echo(f"Config: {result.config_path}")


def index(
    repository_root: Path = typer.Option(Path("."), help="Repository root to index."),
    max_file_bytes: int = typer.Option(
        1_000_000,
        help="Maximum file size to index. Larger files are skipped.",
    ),
) -> None:
    """Index the current repository."""

    result = write_local_repository_index(
        repository_root,
        max_file_bytes=max_file_bytes,
    )
    typer.echo(f"Indexed {result.indexed_file_count} files into {result.index_path}")
    typer.echo(
        "Terms: "
        f"{result.unique_term_count}; bytes: {result.total_bytes}; "
        f"skipped large: {result.skipped_large_count}; "
        f"skipped binary: {result.skipped_binary_count}"
    )
    typer.echo(f"Source types: {_format_count_distribution(result.source_type_counts)}")
    typer.echo(
        "Code metadata: "
        f"symbols={result.symbol_count}; imports={result.import_count}; "
        f"test_files={result.test_file_count}"
    )


def task(
    prompt: str,
    patch: str = typer.Option("", help="Unified diff to use with the static local model provider."),
    repository_root: Path = typer.Option(Path("."), help="Repository root to operate on."),
    dry_run: bool = typer.Option(True, help="Validate the patch without applying it."),
    require_approval: bool = typer.Option(True, help="Require explicit approval before applying."),
    expected_changed_file: list[str] | None = typer.Option(
        None,
        "--expected-changed-file",
        help="File that the generated patch is expected to change.",
    ),
    verification_command: str | None = typer.Option(
        None, help="Allowed test command name to run after applying changes."
    ),
    test_command: list[str] | None = typer.Option(
        None,
        "--test-command",
        help="Shell-free command tokens for the verification command name.",
    ),
    label_outcome: OutcomeLabel | None = typer.Option(None, "--label"),
    label_quality: QualityLabel | None = typer.Option(None, "--quality"),
    failure_mode: list[FailureMode] | None = typer.Option(None, "--failure-mode"),
    reviewer_notes: str | None = typer.Option(None, "--reviewer-notes"),
    dataset_output: Path | None = typer.Option(
        None, help="Optional JSONL path for storing a trace-derived labeled example."
    ),
) -> None:
    """Run a coding-agent task with local fake dependencies and trace capture."""

    allowed_commands = {}
    if verification_command and test_command:
        # test.run selects this command by name; it never receives shell text.
        allowed_commands[verification_command] = test_command
    elif verification_command:
        _fail("--test-command is required when --verification-command is set")

    dataset_store = (
        build_jsonl_dataset_example_store(dataset_output) if dataset_output else None
    )
    # Build an event pipeline so WorkflowCompleted events automatically
    # populate the default dataset under .micro_model_agent/datasets/.
    _trace_dir = trace_dir(repository_root)
    bus = build_event_pipeline(
        trace_dir=_trace_dir,
        dataset_root=repository_root / ".micro_model_agent" / "datasets",
        evaluation_root=repository_root / ".micro_model_agent" / "evaluation",
        promotion_root=repository_root / ".micro_model_agent" / "promotion",
    )
    workflow = build_coding_workflow(
        repository_root=repository_root,
        patch=patch,
        allowed_commands=allowed_commands,
        dataset_store=dataset_store,
        event_bus=bus,
    )
    task_input = CodingAgentTask(
        goal=prompt,
        dry_run=dry_run,
        require_approval=require_approval,
        expected_changed_files=expected_changed_file or [],
        verification_command_name=verification_command,
    )
    result = _run(workflow.run(task_input))

    if label_outcome or label_quality or failure_mode or reviewer_notes:
        # Labels are only useful if the caller also asked to store the resulting
        # trace-derived dataset example.
        if dataset_store is None:
            _fail("--dataset-output is required when labels are provided")
        assert dataset_store is not None
        label = label_from_workflow_result(
            result,
            outcome=label_outcome,
            quality=label_quality,
            failure_modes=tuple(failure_mode) if failure_mode else None,
            reviewer_notes=reviewer_notes,
        )
        _run(dataset_store.save(workflow.dataset_builder.build_example(result.trace, label)))

    typer.echo(result.summary)
    typer.echo(f"Trace: {result.trace_id}")
    if result.changed_files:
        typer.echo("Changed files: " + ", ".join(result.changed_files))
    if not result.ok:
        raise typer.Exit(1)
