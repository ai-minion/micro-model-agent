"""Dataset CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from micro_model_agent.application.datasets import (
    RunDatasetExportRequest,
    RunDatasetExportWorkflow,
    RunDatasetMergeRequest,
    RunDatasetMergeWorkflow,
    RunDatasetRelabelRequest,
    RunDatasetRelabelWorkflow,
    RunDatasetSynthesisRequest,
    RunDatasetSynthesisWorkflow,
    RunDatasetValidationRequest,
    RunDatasetValidationWorkflow,
    RunTraceDatasetExportRequest,
    RunTraceDatasetExportWorkflow,
    RunTraceReviewRequest,
    RunTraceReviewWorkflow,
)
from micro_model_agent.domain.contracts import WorkflowStatus
from micro_model_agent.domain.datasets import (
    DatasetExampleKind,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.infrastructure.dataset_curation import (
    LocalDatasetMerger,
    LocalDatasetRelabeler,
)
from micro_model_agent.infrastructure.dataset_store import (
    LocalDatasetExampleReader,
    LocalDatasetExampleWriter,
)
from micro_model_agent.infrastructure.dataset_validation import (
    LocalDatasetValidator,
    SftJsonlDatasetExporter,
)
from micro_model_agent.infrastructure.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.trace_export import (
    LocalTraceDatasetExporter,
    LocalTraceDatasetExportValidator,
)
from micro_model_agent.infrastructure.trace_review import (
    LocalTraceReviewReader,
    LocalTraceReviewWriter,
)
from micro_model_agent.infrastructure.trace_store import LocalWorkflowTraceReader
from micro_model_agent.interfaces.cli.common import (
    DEFAULT_TRACE_DIR,
    _fail,
    _format_count_distribution,
    _format_tool_profile,
    _run,
)


def register_dataset_commands(dataset_app: typer.Typer) -> None:
    """Register dataset command group handlers."""

    dataset_app.command()(synthesize)
    dataset_app.command()(validate)
    dataset_app.command("export")(export_dataset)
    dataset_app.command("export-traces")(export_traces)
    dataset_app.command("review-trace")(review_trace)
    dataset_app.command("relabel")(relabel_dataset)
    dataset_app.command("merge")(merge_dataset)


def synthesize(
    count: int = typer.Option(100, min=1, help="Number of synthetic examples to generate."),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="Output JSONL path for generated synthetic examples.",
    ),
    template_dir: Path = typer.Option(
        Path("examples/synthetic-data"),
        help="Directory containing committed *.seed.jsonl templates.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Deterministic generation seed for reproducible IDs and variants.",
    ),
    balance_categories: bool = typer.Option(
        True,
        "--balance-categories/--no-balance-categories",
        help="Cycle categories evenly instead of cycling raw templates.",
    ),
    vary_scenarios: bool = typer.Option(
        True,
        "--vary-scenarios/--no-vary-scenarios",
        help="Create deterministic prompt variants while preserving validated targets.",
    ),
    include_category: list[str] | None = typer.Option(
        None,
        "--include-category",
        help="Only synthesize templates with this metadata category. Can be repeated.",
    ),
    exclude_category: list[str] | None = typer.Option(
        None,
        "--exclude-category",
        help="Skip templates with this metadata category. Can be repeated.",
    ),
) -> None:
    """Generate synthetic tool-use and workflow examples."""

    workflow = RunDatasetSynthesisWorkflow(
        generator=SyntheticTemplateGenerator(template_dir),
        validator=LocalDatasetValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )
    try:
        result = _run(
            workflow.run(
                RunDatasetSynthesisRequest(
                    count=count,
                    output_path=output,
                    seed=seed,
                    balance_categories=balance_categories,
                    vary_scenarios=vary_scenarios,
                    include_categories=tuple(include_category or ()),
                    exclude_categories=tuple(exclude_category or ()),
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    validation = result.evaluation
    if not validation.passed:
        typer.echo(validation.summary, err=True)
        for error in validation.details.get("errors", [])[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Wrote {result.example_count} synthetic examples to {output}")
    typer.echo(
        "Categories: " + _format_count_distribution(validation.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(validation.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(validation.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(validation.details.get("tool_profile")))


def validate(
    path: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="JSONL dataset path to validate.",
    ),
) -> None:
    """Validate dataset records before training."""

    workflow = RunDatasetValidationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
    )
    try:
        validation = _run(workflow.run(RunDatasetValidationRequest(path=path)))
    except ValueError as exc:
        _fail(str(exc))

    result = validation.evaluation
    typer.echo(result.summary)
    typer.echo(
        "Categories: " + _format_count_distribution(result.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(result.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(result.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(result.details.get("tool_profile")))
    for error in result.details.get("errors", [])[:10]:
        typer.echo(f"- {error}")
    if not result.passed:
        raise typer.Exit(1)


def export_dataset(
    path: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="Validated dataset path to export.",
    ),
    output_format: str = typer.Option("sft-jsonl", "--format", help="Dataset export format."),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.sft.jsonl"),
        help="Output path for exported dataset.",
    ),
) -> None:
    """Export dataset records for a training backend."""

    workflow = RunDatasetExportWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
        exporter=SftJsonlDatasetExporter(),
    )
    try:
        result = _run(
            workflow.run(
                RunDatasetExportRequest(
                    path=path,
                    output_path=output,
                    output_format=output_format,
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    if not result.evaluation.passed:
        typer.echo(result.evaluation.summary, err=True)
        raise typer.Exit(1)

    typer.echo(f"Exported {result.example_count} examples to {output}")


def export_traces(
    trace_path: Path = typer.Option(
        DEFAULT_TRACE_DIR / "workflows.jsonl",
        help="Stored workflow trace JSONL path.",
    ),
    review_path: Path = typer.Option(
        DEFAULT_TRACE_DIR / "reviews.jsonl",
        help="Stored human trace review JSONL path for --label-mode reviewed.",
    ),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/trace_examples.jsonl"),
        help="Output JSONL path for trace-derived examples.",
    ),
    label_mode: str = typer.Option(
        "review",
        "--label-mode",
        help="Label mode: review or evaluation.",
    ),
    kind: DatasetExampleKind = typer.Option(
        DatasetExampleKind.REPAIR,
        "--kind",
        help="Dataset example kind to export.",
    ),
    outcome: OutcomeLabel | None = typer.Option(
        None,
        "--outcome",
        help="Only export examples with this outcome after label assignment.",
    ),
    quality: QualityLabel | None = typer.Option(
        None,
        "--quality",
        help="Only export examples with this quality after label assignment.",
    ),
    workflow_status: WorkflowStatus | None = typer.Option(
        None,
        "--workflow-status",
        help="Only export traces with this workflow status.",
    ),
    require_tool_call: bool = typer.Option(
        False,
        "--require-tool-call",
        help="Only export traces that include at least one tool call.",
    ),
    max_examples: int | None = typer.Option(
        None,
        "--max-examples",
        min=1,
        help="Optional maximum number of examples to export.",
    ),
) -> None:
    """Export stored workflow traces as redacted dataset examples."""

    workflow = RunTraceDatasetExportWorkflow(
        trace_reader=LocalWorkflowTraceReader(trace_path),
        review_reader=LocalTraceReviewReader(review_path),
        trace_exporter=LocalTraceDatasetExporter(),
        trace_export_validator=LocalTraceDatasetExportValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )
    try:
        result = _run(
            workflow.run(
                RunTraceDatasetExportRequest(
                    output_path=output,
                    label_mode=label_mode,
                    kind=kind,
                    outcome=outcome,
                    quality=quality,
                    workflow_status=workflow_status,
                    require_tool_call=require_tool_call,
                    max_examples=max_examples,
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    if result.validation_errors:
        typer.echo(
            f"trace export found {len(result.validation_errors)} error(s)",
            err=True,
        )
        for error in result.validation_errors[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Exported {result.example_count} trace-derived examples to {output}")
    typer.echo(f"Traces read: {result.traces_read}")
    typer.echo(f"Reviews read: {result.reviews_read}")
    typer.echo(f"Redacted examples: {result.redacted_count}")
    typer.echo("Outcomes: " + _format_count_distribution(result.outcome_counts))


def review_trace(
    trace_id: str = typer.Option(..., "--trace-id", help="Stored workflow trace id to review."),
    outcome: OutcomeLabel = typer.Option(..., "--outcome", help="Human outcome label."),
    quality: QualityLabel = typer.Option(..., "--quality", help="Human quality label."),
    failure_mode: list[FailureMode] | None = typer.Option(
        None,
        "--failure-mode",
        help="Failure mode label. Can be passed more than once.",
    ),
    reviewer_notes: str | None = typer.Option(
        None,
        "--reviewer-notes",
        help="Human review notes for this trace.",
    ),
    corrected_target_json: str | None = typer.Option(
        None,
        "--corrected-target-json",
        help="Optional corrected dataset target JSON object for this trace.",
    ),
    corrected_target_file: Path | None = typer.Option(
        None,
        "--corrected-target-file",
        help="Optional file containing a corrected dataset target JSON object.",
    ),
    output: Path = typer.Option(
        DEFAULT_TRACE_DIR / "reviews.jsonl",
        "--output",
        help="Append-only human trace review JSONL path.",
    ),
) -> None:
    """Record a human review label for one stored workflow trace."""

    if corrected_target_json and corrected_target_file:
        _fail("pass only one of --corrected-target-json or --corrected-target-file")

    raw_corrected_target: str | None = corrected_target_json
    if corrected_target_file:
        if not corrected_target_file.exists():
            _fail(f"corrected target file does not exist: {corrected_target_file}")
        raw_corrected_target = corrected_target_file.read_text(encoding="utf-8")

    workflow = RunTraceReviewWorkflow(review_writer=LocalTraceReviewWriter())
    try:
        result = _run(
            workflow.run(
                RunTraceReviewRequest(
                    trace_id=trace_id,
                    outcome=outcome,
                    quality=quality,
                    output_path=output,
                    failure_modes=tuple(failure_mode or ()),
                    reviewer_notes=reviewer_notes,
                    corrected_target_json=raw_corrected_target,
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    typer.echo(
        f"Recorded {result.quality.value}/{result.outcome.value} review "
        f"for trace {result.trace_id}"
    )
    typer.echo(f"Review: {result.review_id}")


def relabel_dataset(
    path: Path = typer.Option(..., help="Input JSONL dataset path to relabel."),
    output: Path = typer.Option(..., help="Output JSONL path for relabeled examples."),
    trace_id: str | None = typer.Option(
        None,
        "--trace-id",
        help="Only relabel the example with this metadata.trace_id.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Only relabel examples with this exact source.",
    ),
    input_outcome: OutcomeLabel | None = typer.Option(
        None,
        "--input-outcome",
        help="Only relabel examples currently carrying this outcome.",
    ),
    input_quality: QualityLabel | None = typer.Option(
        None,
        "--input-quality",
        help="Only relabel examples currently carrying this quality.",
    ),
    outcome: OutcomeLabel | None = typer.Option(
        None,
        "--outcome",
        help="New outcome label for matched examples.",
    ),
    quality: QualityLabel | None = typer.Option(
        None,
        "--quality",
        help="New quality label for matched examples.",
    ),
    failure_mode: list[FailureMode] | None = typer.Option(
        None,
        "--failure-mode",
        help="Replacement failure mode label. Can be passed more than once.",
    ),
    reviewer_notes: str | None = typer.Option(
        None,
        "--reviewer-notes",
        help="Replacement reviewer notes for matched examples.",
    ),
) -> None:
    """Relabel reviewed dataset examples and write a curated JSONL file."""

    workflow = RunDatasetRelabelWorkflow(
        example_reader=LocalDatasetExampleReader(),
        relabeler=LocalDatasetRelabeler(),
        example_writer=LocalDatasetExampleWriter(),
        validator=LocalDatasetValidator(),
    )
    try:
        result = _run(
            workflow.run(
                RunDatasetRelabelRequest(
                    path=path,
                    output_path=output,
                    trace_id=trace_id,
                    source=source,
                    input_outcome=input_outcome,
                    input_quality=input_quality,
                    outcome=outcome,
                    quality=quality,
                    failure_modes=tuple(failure_mode) if failure_mode is not None else None,
                    reviewer_notes=reviewer_notes,
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    validation = result.evaluation
    typer.echo(f"Relabeled {result.changed_count} of {result.original_count} examples to {output}")
    typer.echo(validation.summary)
    typer.echo(
        "Categories: " + _format_count_distribution(validation.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(validation.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(validation.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(validation.details.get("tool_profile")))
    for error in validation.details.get("errors", [])[:10]:
        typer.echo(f"- {error}")


def merge_dataset(
    input_path: list[Path] = typer.Option(
        ...,
        "--input",
        help="Input JSONL dataset path. Pass more than once.",
    ),
    output: Path = typer.Option(..., help="Output JSONL path for the merged dataset."),
    deduplicate_by: str = typer.Option(
        "source",
        "--deduplicate-by",
        help="Dedupe key: source or id.",
    ),
    validate_training_ready: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate the merged dataset before reporting success.",
    ),
) -> None:
    """Merge dataset JSONL files with simple deduplication."""

    workflow = RunDatasetMergeWorkflow(
        example_reader=LocalDatasetExampleReader(),
        merger=LocalDatasetMerger(),
        validator=LocalDatasetValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )
    try:
        result = _run(
            workflow.run(
                RunDatasetMergeRequest(
                    input_paths=tuple(input_path),
                    output_path=output,
                    deduplicate_by=deduplicate_by,
                    validate_training_ready=validate_training_ready,
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    validation = result.evaluation
    if not result.saved:
        typer.echo(validation.summary, err=True)
        for error in validation.details.get("errors", [])[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Merged {result.merged_count} examples to {output}")
    typer.echo(f"Skipped duplicates: {result.skipped_duplicates}")
    typer.echo(validation.summary)
    typer.echo(
        "Categories: " + _format_count_distribution(validation.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(validation.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(validation.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(validation.details.get("tool_profile")))
