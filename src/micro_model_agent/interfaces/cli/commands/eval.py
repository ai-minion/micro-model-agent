"""Evaluation CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from micro_model_agent.evaluation.application.workflows import (
    RunEvaluationComparisonRequest,
    RunSyntheticEvaluationRequest,
    RunTraceEvaluationRequest,
    RunWorkspaceStagedEvaluationRequest,
    RunWorkspaceStagedReviewRequest,
    RunWorkspaceStagedReviewWriteRequest,
)
from micro_model_agent.infrastructure.composition import (
    build_evaluation_comparison_workflow,
    build_synthetic_evaluation_workflow,
    build_trace_evaluation_workflow,
    build_workspace_staged_evaluation_workflow,
    build_workspace_staged_review_workflow,
    default_evaluation_available_tools,
)
from micro_model_agent.interfaces.cli.common import (
    _fail,
    _format_count_distribution,
    _load_dotenv,
    _run,
    _select_evaluation_model,
)


def register_eval_commands(eval_app: typer.Typer) -> None:
    """Register evaluation command group handlers."""

    eval_app.command("synthetic")(eval_synthetic)
    eval_app.command("traces")(eval_traces)
    eval_app.command("workspace-staged")(eval_workspace_staged)
    eval_app.command("review-workspace-staged")(review_workspace_staged)
    eval_app.command("compare")(eval_compare)


def _format_behavioral_eval_failures(details: dict[str, Any]) -> list[str]:
    """Return short CLI lines for failing behavioral eval examples."""

    lines: list[str] = []
    examples = details.get("examples")
    if not isinstance(examples, list):
        return lines

    for raw_example in examples:
        if not isinstance(raw_example, dict):
            continue
        score = raw_example.get("score")
        if not isinstance(score, int | float) or score >= 1.0:
            continue

        example_id = raw_example.get("example_id", "unknown")
        category = raw_example.get("category") or "uncategorized"
        errors = raw_example.get("errors")
        error_text = ""
        if isinstance(errors, list) and errors:
            error_text = f": {errors[0]}"
        lines.append(f"- {category}/{example_id} scored {score:.2f}{error_text}")
    return lines


def eval_synthetic(
    run_id: str = typer.Option("latest", help="Training run id or alias to evaluate."),
    dataset: Path = typer.Option(
        Path("examples/synthetic-data/held-out.behavior.jsonl"),
        help="Held-out synthetic JSONL dataset to evaluate against.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Ollama model name to evaluate.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Transformers base model for direct PEFT adapter evaluation.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Local PEFT adapter path for direct Transformers evaluation.",
    ),
    ollama_base_url: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama host URL. Defaults to MICRO_MODEL_AGENT_OLLAMA_BASE_URL.",
    ),
    max_new_tokens: int = typer.Option(
        384,
        min=1,
        max=4096,
        help="Maximum generated tokens per evaluation example.",
    ),
    max_examples: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on evaluated examples.",
    ),
    pass_threshold: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum average behavioral score required to pass.",
    ),
    scripted_response: list[str] | None = typer.Option(
        None,
        "--scripted-response",
        help="Scripted JSON model response. Can be passed more than once.",
    ),
    scripted_response_file: Path | None = typer.Option(
        None,
        "--scripted-response-file",
        help="JSONL file containing scripted model responses for evaluation tests.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Evaluation report path. Defaults to <run>/evaluation.json.",
    ),
) -> None:
    """Evaluate a model or training run against synthetic behavior examples."""

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    _load_dotenv()
    model_selection = _select_evaluation_model(
        run_dir=run_dir,
        model=model,
        base_model=base_model,
        adapter_path=adapter_path,
        scripted_response=scripted_response,
        scripted_response_file=scripted_response_file,
        max_new_tokens=max_new_tokens,
        ollama_base_url=ollama_base_url,
    )

    workflow = build_synthetic_evaluation_workflow(pass_threshold=pass_threshold)
    try:
        workflow_result = _run(
            workflow.run(
                RunSyntheticEvaluationRequest(
                    run_id=run_id,
                    run_dir=run_dir,
                    dataset_path=dataset,
                    provider_kind=model_selection.provider_kind,
                    model_provider=model_selection.provider,
                    artifact=model_selection.artifact,
                    model=model_selection.model,
                    base_model=model_selection.base_model,
                    adapter_path=model_selection.adapter_path,
                    max_examples=max_examples,
                    output_path=output,
                    default_available_tools=default_evaluation_available_tools(),
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    result = workflow_result.evaluation
    typer.echo(f"{result.summary}; report written to {workflow_result.report_path}")
    if not result.passed:
        failure_lines = _format_behavioral_eval_failures(result.details)
        if failure_lines:
            typer.echo("Failures:", err=True)
            for line in failure_lines:
                typer.echo(line, err=True)
        raise typer.Exit(1)


def eval_traces(
    run_id: str = typer.Option("latest", help="Training run id or alias to evaluate."),
    dataset: Path = typer.Option(
        Path("examples/trace-data/held-out.trace.jsonl"),
        help="Held-out trace-derived JSONL dataset to evaluate against.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Ollama model name to evaluate.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Transformers base model for direct PEFT adapter evaluation.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Local PEFT adapter path for direct Transformers evaluation.",
    ),
    ollama_base_url: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama host URL. Defaults to MICRO_MODEL_AGENT_OLLAMA_BASE_URL.",
    ),
    max_new_tokens: int = typer.Option(
        512,
        min=1,
        max=4096,
        help="Maximum generated tokens per evaluation example.",
    ),
    max_examples: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on evaluated examples.",
    ),
    pass_threshold: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum average trace behavior score required to pass.",
    ),
    scripted_response: list[str] | None = typer.Option(
        None,
        "--scripted-response",
        help="Scripted JSON model response. Can be passed more than once.",
    ),
    scripted_response_file: Path | None = typer.Option(
        None,
        "--scripted-response-file",
        help="JSONL file containing scripted model responses for evaluation tests.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Evaluation report path. Defaults to <run>/evaluation.json.",
    ),
) -> None:
    """Evaluate a model or training run against held-out trace-derived examples."""

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    _load_dotenv()
    model_selection = _select_evaluation_model(
        run_dir=run_dir,
        model=model,
        base_model=base_model,
        adapter_path=adapter_path,
        scripted_response=scripted_response,
        scripted_response_file=scripted_response_file,
        max_new_tokens=max_new_tokens,
        ollama_base_url=ollama_base_url,
    )

    workflow = build_trace_evaluation_workflow(pass_threshold=pass_threshold)
    try:
        workflow_result = _run(
            workflow.run(
                RunTraceEvaluationRequest(
                    run_id=run_id,
                    run_dir=run_dir,
                    dataset_path=dataset,
                    provider_kind=model_selection.provider_kind,
                    model_provider=model_selection.provider,
                    model=model_selection.model,
                    base_model=model_selection.base_model,
                    adapter_path=model_selection.adapter_path,
                    max_examples=max_examples,
                    output_path=output,
                    default_available_tools=default_evaluation_available_tools(),
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    result = workflow_result.evaluation
    typer.echo(f"{result.summary}; report written to {workflow_result.report_path}")
    if not result.passed:
        raise typer.Exit(1)


def eval_workspace_staged(
    run_id: str = typer.Option("latest", help="Training run id or alias to evaluate."),
    dataset: Path = typer.Option(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl"),
        help="Held-out staged workspace JSONL dataset to evaluate against.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Ollama model name to evaluate.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Transformers base model for direct PEFT adapter evaluation.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Local PEFT adapter path for direct Transformers evaluation.",
    ),
    ollama_base_url: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama host URL. Defaults to MICRO_MODEL_AGENT_OLLAMA_BASE_URL.",
    ),
    max_new_tokens: int = typer.Option(
        1024,
        min=1,
        max=4096,
        help="Maximum generated tokens per evaluation example.",
    ),
    max_examples: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on evaluated examples.",
    ),
    pass_threshold: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum average staged workspace score required to pass.",
    ),
    rubric_version: str = typer.Option(
        "legacy",
        "--rubric-version",
        help="Staged rubric version: legacy, v2, or auto.",
    ),
    scripted_response: list[str] | None = typer.Option(
        None,
        "--scripted-response",
        help="Scripted JSON model response. Can be passed more than once.",
    ),
    scripted_response_file: Path | None = typer.Option(
        None,
        "--scripted-response-file",
        help="JSONL file containing scripted model responses for evaluation tests.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Evaluation report path. Defaults to <run>/evaluation.json.",
    ),
) -> None:
    """Evaluate staged workspace reasoning without applying patches."""

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    _load_dotenv()
    model_selection = _select_evaluation_model(
        run_dir=run_dir,
        model=model,
        base_model=base_model,
        adapter_path=adapter_path,
        scripted_response=scripted_response,
        scripted_response_file=scripted_response_file,
        max_new_tokens=max_new_tokens,
        ollama_base_url=ollama_base_url,
    )

    if rubric_version not in {"legacy", "v2", "auto"}:
        _fail(f"Unsupported workspace-staged rubric version: {rubric_version}")

    workflow = build_workspace_staged_evaluation_workflow(
        pass_threshold=pass_threshold,
        rubric_version=rubric_version,
    )
    try:
        workflow_result = _run(
            workflow.run(
                RunWorkspaceStagedEvaluationRequest(
                    run_id=run_id,
                    run_dir=run_dir,
                    dataset_path=dataset,
                    provider_kind=model_selection.provider_kind,
                    model_provider=model_selection.provider,
                    model=model_selection.model,
                    base_model=model_selection.base_model,
                    adapter_path=model_selection.adapter_path,
                    max_examples=max_examples,
                    output_path=output,
                    default_available_tools=default_evaluation_available_tools(),
                )
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    result = workflow_result.evaluation
    typer.echo(f"{result.summary}; report written to {workflow_result.report_path}")
    if not result.passed:
        raise typer.Exit(1)


def review_workspace_staged(
    dataset: Path = typer.Option(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl"),
        help="Staged workspace scenario JSONL dataset.",
    ),
    report: list[Path] | None = typer.Option(
        None,
        "--report",
        help="Staged workspace evaluation JSON report. Can be passed more than once.",
    ),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/workspace_staged_review_queue.jsonl"),
        "--output",
        help="Output JSONL review queue path.",
    ),
    simple_failure_threshold: float = typer.Option(
        0.4,
        min=0.0,
        max=1.0,
        help="Auto-reject examples whose best model score is at or below this value.",
    ),
    auto_accept_threshold: float = typer.Option(
        0.95,
        min=0.0,
        max=1.0,
        help="Mark examples as auto-accept candidates at or above this best score.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Prompt for a human decision and notes for each review record.",
    ),
) -> None:
    """Build or complete a staged workspace review queue from eval reports."""

    workflow = build_workspace_staged_review_workflow()
    try:
        build_result = workflow.build(
            RunWorkspaceStagedReviewRequest(
                dataset_path=dataset,
                report_paths=tuple(report or ()),
                output_path=output,
                simple_failure_threshold=simple_failure_threshold,
                auto_accept_threshold=auto_accept_threshold,
            )
        )
    except ValueError as exc:
        _fail(str(exc))

    records = build_result.records

    if interactive:
        for index, record in enumerate(records, start=1):
            typer.echo("")
            typer.echo(f"[{index}/{len(records)}] {record['category']}: {record['goal']}")
            typer.echo(f"Auto triage: {record['auto_triage']['decision']}")
            typer.echo(f"Reason: {record['auto_triage']['reason']}")
            for result_record in record["model_results"]:
                score = result_record["score"]
                score_text = "none" if score is None else f"{score:.2f}"
                typer.echo(
                    f"- {result_record['run_id'] or result_record['report_path']}: "
                    f"score={score_text} provider={result_record['provider']}"
                )
            decision = typer.prompt(
                "Decision [accepted/rejected/needs_review/skip]",
                default="needs_review",
            )
            if decision not in {"accepted", "rejected", "needs_review", "skip"}:
                _fail(f"unsupported review decision: {decision}")
            notes = typer.prompt("Notes", default="")
            record["review"] = {
                "decision": decision,
                "notes": notes or None,
            }

    write_result = workflow.write(
        RunWorkspaceStagedReviewWriteRequest(
            output_path=output,
            records=records,
        )
    )
    typer.echo(
        f"Wrote {write_result.record_count} staged workspace review record(s) "
        f"to {write_result.output_path}"
    )
    typer.echo("Auto triage: " + _format_count_distribution(write_result.auto_triage_counts))


def eval_compare(
    baseline_report: Path = typer.Option(
        ...,
        "--baseline-report",
        help="Persisted evaluation JSON report for the base model.",
    ),
    adapter_report: Path = typer.Option(
        ...,
        "--adapter-report",
        help="Persisted evaluation JSON report for the trained adapter.",
    ),
    minimum_score_delta: float = typer.Option(
        0.0,
        min=0.0,
        help="Minimum adapter score improvement over baseline.",
    ),
    minimum_metric_delta: list[str] | None = typer.Option(
        None,
        "--minimum-metric-delta",
        help="Required metric improvement as metric_name=delta. Can be passed more than once.",
    ),
    require_adapter_passed: bool = typer.Option(
        True,
        "--require-adapter-passed/--allow-failing-adapter",
        help="Require the adapter report itself to pass before comparing improvements.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Comparison report path. Defaults to <adapter-report>.comparison.json.",
    ),
) -> None:
    """Compare a baseline evaluation report with a trained adapter report."""

    thresholds = _parse_metric_delta_options(minimum_metric_delta or [])
    report_path = output or adapter_report.with_suffix(".comparison.json")
    workflow = build_evaluation_comparison_workflow()
    result = workflow.run(
        RunEvaluationComparisonRequest(
            baseline_report_path=baseline_report,
            adapter_report_path=adapter_report,
            output_path=report_path,
            minimum_score_delta=minimum_score_delta,
            minimum_metric_deltas=thresholds,
            require_adapter_passed=require_adapter_passed,
        )
    )
    comparison = result.comparison

    typer.echo(f"{comparison.summary}; report written to {result.output_path}")
    if not comparison.passed:
        for error in comparison.errors:
            typer.echo(error, err=True)
        raise typer.Exit(1)


def _parse_metric_delta_options(values: list[str]) -> dict[str, float]:
    """Parse metric threshold CLI values."""

    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            _fail(f"metric delta must use metric_name=delta: {value}")
        name, raw_delta = value.split("=", 1)
        metric_name = name.strip()
        if not metric_name:
            _fail(f"metric delta must include a metric name: {value}")
        try:
            delta = float(raw_delta)
        except ValueError:
            _fail(f"metric delta must include a numeric threshold: {value}")
        if delta < 0.0:
            _fail(f"metric delta must be non-negative: {value}")
        thresholds[metric_name] = delta
    return thresholds
