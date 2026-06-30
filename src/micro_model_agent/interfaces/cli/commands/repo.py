"""Root repository and task CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from micro_model_agent.domain.datasets import FailureMode, OutcomeLabel, QualityLabel
from micro_model_agent.infrastructure.dataset_store import JsonlDatasetExampleStore
from micro_model_agent.infrastructure.local_index import LocalLexicalIndexWriter
from micro_model_agent.infrastructure.repository_metadata import initialize_repository
from micro_model_agent.infrastructure.trace_store import JsonlTraceStore
from micro_model_agent.interfaces.cli.common import (
    DEFAULT_TRACE_DIR,
    _fail,
    _format_count_distribution,
    _run,
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

    result = initialize_repository(
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

    result = LocalLexicalIndexWriter(
        repository_root,
        max_file_bytes=max_file_bytes,
    ).write()
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

    # Imports stay inside the command so lightweight commands start quickly and
    # optional dependencies are only loaded by commands that need them.
    from micro_model_agent.agents.coding_agent import CodingAgent, CodingAgentTask
    from micro_model_agent.application.workflows import RunAgentWorkflow, label_from_workflow_result
    from micro_model_agent.infrastructure.fake_model_provider import StaticModelProvider
    from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor

    allowed_commands = {}
    if verification_command and test_command:
        # test.run selects this command by name; it never receives shell text.
        allowed_commands[verification_command] = test_command
    elif verification_command:
        _fail("--test-command is required when --verification-command is set")

    trace_store = JsonlTraceStore(repository_root / DEFAULT_TRACE_DIR / "workflows.jsonl")
    # StaticModelProvider lets this command exercise the workflow using a patch
    # supplied on the command line instead of calling a real model.
    executor = BuiltinToolExecutor(repository_root, allowed_commands)
    agent = CodingAgent(
        model_provider=StaticModelProvider(patch),
        tool_executor=executor,
        trace_store=trace_store,
    )
    dataset_store = JsonlDatasetExampleStore(dataset_output) if dataset_output else None
    workflow = RunAgentWorkflow(agent=agent, dataset_store=dataset_store)
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
