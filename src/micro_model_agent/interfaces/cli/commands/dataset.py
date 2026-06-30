"""Dataset CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer

from micro_model_agent.domain.contracts import WorkflowStatus
from micro_model_agent.domain.datasets import (
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.infrastructure.dataset_curation import (
    DeduplicateBy,
    merge_datasets,
    relabel_examples,
)
from micro_model_agent.infrastructure.dataset_store import (
    JsonlDatasetExampleStore,
    load_dataset_examples,
)
from micro_model_agent.infrastructure.dataset_validation import (
    LocalDatasetValidator,
    export_sft_jsonl,
)
from micro_model_agent.infrastructure.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.trace_export import (
    TraceDatasetExporter,
    validate_trace_export_examples,
)
from micro_model_agent.infrastructure.trace_review import (
    JsonlTraceReviewStore,
    TraceReview,
)
from micro_model_agent.infrastructure.trace_store import JsonlTraceStore
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

    generator = SyntheticTemplateGenerator(template_dir)
    store = JsonlDatasetExampleStore(output)
    examples = _run(
        generator.generate(
            count,
            seed=seed,
            balance_categories=balance_categories,
            vary_scenarios=vary_scenarios,
            include_categories=tuple(include_category or ()),
            exclude_categories=tuple(exclude_category or ()),
        )
    )
    validation = _run(LocalDatasetValidator().validate(examples))
    if not validation.passed:
        typer.echo(validation.summary, err=True)
        for error in validation.details.get("errors", [])[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)
    _run(store.save_many(examples))
    typer.echo(f"Wrote {len(examples)} synthetic examples to {output}")
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

    examples = load_dataset_examples(path)
    result = _run(LocalDatasetValidator().validate(examples))
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

    if output_format != "sft-jsonl":
        _fail(f"Unsupported dataset export format: {output_format}")

    examples = load_dataset_examples(path)
    result = _run(LocalDatasetValidator().validate(examples))
    if not result.passed:
        typer.echo(result.summary, err=True)
        raise typer.Exit(1)

    export_sft_jsonl(output, examples)
    typer.echo(f"Exported {len(examples)} examples to {output}")


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

    store = JsonlTraceStore(trace_path)
    traces = _run(store.list())
    reviews_by_trace_id = _run(JsonlTraceReviewStore(review_path).latest_by_trace_id())
    exporter = TraceDatasetExporter()
    try:
        examples = exporter.export(
            traces,
            kind=kind,
            label_mode=label_mode,
            reviews_by_trace_id=reviews_by_trace_id,
            outcome=outcome,
            quality=quality,
            workflow_status=workflow_status,
            require_tool_call=require_tool_call,
            max_examples=max_examples,
        )
    except ValueError as exc:
        _fail(str(exc))

    errors = validate_trace_export_examples(examples)
    if errors:
        typer.echo(f"trace export found {len(errors)} error(s)", err=True)
        for error in errors[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)

    _run(JsonlDatasetExampleStore(output).save_many(examples))
    redacted_count = sum(1 for example in examples if example.metadata.get("redacted"))
    typer.echo(f"Exported {len(examples)} trace-derived examples to {output}")
    typer.echo(f"Traces read: {len(traces)}")
    typer.echo(f"Reviews read: {len(reviews_by_trace_id)}")
    typer.echo(f"Redacted examples: {redacted_count}")
    typer.echo(
        "Outcomes: "
        + _format_count_distribution(
            {
                outcome.value: sum(1 for example in examples if example.label.outcome is outcome)
                for outcome in OutcomeLabel
            }
        )
    )


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

    corrected_target: dict[str, Any] | None = None
    raw_corrected_target: str | None = corrected_target_json
    if corrected_target_file:
        if not corrected_target_file.exists():
            _fail(f"corrected target file does not exist: {corrected_target_file}")
        raw_corrected_target = corrected_target_file.read_text(encoding="utf-8")
    if raw_corrected_target:
        try:
            parsed = json.loads(raw_corrected_target)
        except json.JSONDecodeError as exc:
            _fail(f"corrected target must be valid JSON: {exc}")
        if not isinstance(parsed, dict):
            _fail("corrected target must be a JSON object")
        corrected_target = parsed

    review = TraceReview(
        trace_id=trace_id,
        label=DatasetLabel(
            outcome=outcome,
            quality=quality,
            failure_modes=tuple(failure_mode or ()),
            reviewer_notes=reviewer_notes,
        ),
        corrected_target=corrected_target,
    )
    _run(JsonlTraceReviewStore(output).save(review))
    typer.echo(f"Recorded {quality.value}/{outcome.value} review for trace {trace_id}")
    typer.echo(f"Review: {review.id}")


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

    if not any([outcome, quality, failure_mode, reviewer_notes is not None]):
        _fail("At least one label update is required")

    examples = load_dataset_examples(path)
    relabeled, changed = relabel_examples(
        examples,
        trace_id=trace_id,
        source=source,
        input_outcome=input_outcome,
        input_quality=input_quality,
        outcome=outcome,
        quality=quality,
        failure_modes=tuple(failure_mode) if failure_mode is not None else None,
        reviewer_notes=reviewer_notes,
    )
    _run(JsonlDatasetExampleStore(output).save_many(relabeled))
    validation = _run(LocalDatasetValidator().validate(relabeled))
    typer.echo(f"Relabeled {changed} of {len(examples)} examples to {output}")
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

    if deduplicate_by not in {"source", "id"}:
        _fail(f"Unsupported dedupe key: {deduplicate_by}")

    datasets = [load_dataset_examples(path) for path in input_path]
    merged, skipped = merge_datasets(
        datasets,
        deduplicate_by=cast(DeduplicateBy, deduplicate_by),
    )
    validation = _run(LocalDatasetValidator().validate(merged))
    if validate_training_ready and not validation.passed:
        typer.echo(validation.summary, err=True)
        for error in validation.details.get("errors", [])[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)

    _run(JsonlDatasetExampleStore(output).save_many(merged))
    typer.echo(f"Merged {len(merged)} examples to {output}")
    typer.echo(f"Skipped duplicates: {skipped}")
    typer.echo(validation.summary)
    typer.echo(
        "Categories: " + _format_count_distribution(validation.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(validation.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(validation.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(validation.details.get("tool_profile")))
