"""Evaluation CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import DatasetExample
from micro_model_agent.infrastructure.dataset_metadata import summarize_tool_profiles
from micro_model_agent.infrastructure.dataset_store import load_dataset_examples
from micro_model_agent.infrastructure.evaluation_comparison import compare_evaluation_results
from micro_model_agent.infrastructure.synthetic_evaluation import (
    SyntheticBehaviorEvaluationSuite,
    TraceBehaviorEvaluationSuite,
)
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS
from micro_model_agent.infrastructure.training_artifacts import (
    SyntheticEvaluationSuite,
    load_evaluation_result,
    write_evaluation_result,
)
from micro_model_agent.infrastructure.workspace_staged_evaluation import (
    WorkspaceStagedEvaluationSuite,
    build_workspace_staged_review_records,
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


def _with_evaluation_metadata(
    result: EvaluationResult,
    *,
    run_id: str,
    dataset: Path,
    examples: list[DatasetExample],
    provider_kind: str,
    model: str | None = None,
    base_model: str | None = None,
    adapter_path: Path | None = None,
) -> EvaluationResult:
    """Add report-level metadata without changing evaluator scoring."""

    return EvaluationResult(
        passed=result.passed,
        summary=result.summary,
        score=result.score,
        details={
            **result.details,
            "evaluation_metadata": {
                "run_id": run_id,
                "dataset_path": str(dataset),
                "provider": provider_kind,
                "model": model,
                "base_model": base_model,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "tool_profile": summarize_tool_profiles(
                    examples,
                    default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
                ),
            },
        },
    )


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

    if model_selection.provider is not None:
        examples = load_dataset_examples(dataset)
        if max_examples is not None:
            examples = examples[:max_examples]
        result = _run(
            SyntheticBehaviorEvaluationSuite(pass_threshold=pass_threshold).evaluate_model(
                model_selection.provider,
                examples,
            )
        )
        result = _with_evaluation_metadata(
            result,
            run_id=run_id,
            dataset=dataset,
            examples=examples,
            provider_kind=model_selection.provider_kind,
            model=model_selection.model,
            base_model=model_selection.base_model,
            adapter_path=model_selection.adapter_path,
        )
        report_path = write_evaluation_result(run_dir, result, output)
        typer.echo(f"{result.summary}; report written to {report_path}")
        if not result.passed:
            failure_lines = _format_behavioral_eval_failures(result.details)
            if failure_lines:
                typer.echo("Failures:", err=True)
                for line in failure_lines:
                    typer.echo(line, err=True)
            raise typer.Exit(1)
        return

    # Dry-run training artifacts do not contain runnable model weights. Keep the
    # old metadata smoke gate for those cases, but mark it clearly in the report.
    if model_selection.artifact is None:
        _fail("synthetic eval requires a training artifact, runnable model, or scripted response")
    result = _run(SyntheticEvaluationSuite().evaluate_artifact(model_selection.artifact))
    dataset_examples = load_dataset_examples(dataset)
    if max_examples is not None:
        dataset_examples = dataset_examples[:max_examples]
    result = _with_evaluation_metadata(
        result,
        run_id=run_id,
        dataset=dataset,
        examples=dataset_examples,
        provider_kind=model_selection.provider_kind,
        model=model_selection.model,
        base_model=model_selection.artifact.base_model,
        adapter_path=Path(model_selection.artifact.path),
    )
    report_path = write_evaluation_result(run_dir, result, output)
    typer.echo(f"{result.summary}; report written to {report_path}")
    if not result.passed:
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

    if model_selection.provider is None:
        _fail("trace eval requires a runnable model, adapter, or scripted response")

    examples = load_dataset_examples(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]
    result = _run(
        TraceBehaviorEvaluationSuite(pass_threshold=pass_threshold).evaluate_model(
            model_selection.provider,
            examples,
        )
    )
    result = _with_evaluation_metadata(
        result,
        run_id=run_id,
        dataset=dataset,
        examples=examples,
        provider_kind=model_selection.provider_kind,
        model=model_selection.model,
        base_model=model_selection.base_model,
        adapter_path=model_selection.adapter_path,
    )
    report_path = write_evaluation_result(run_dir, result, output)
    typer.echo(f"{result.summary}; report written to {report_path}")
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

    if model_selection.provider is None:
        _fail("workspace-staged eval requires a runnable model, adapter, or scripted response")
    if rubric_version not in {"legacy", "v2", "auto"}:
        _fail(f"Unsupported workspace-staged rubric version: {rubric_version}")

    examples = load_dataset_examples(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]
    result = _run(
        WorkspaceStagedEvaluationSuite(
            pass_threshold=pass_threshold,
            rubric_version=rubric_version,
        ).evaluate_model(model_selection.provider, examples)
    )
    result = _with_evaluation_metadata(
        result,
        run_id=run_id,
        dataset=dataset,
        examples=examples,
        provider_kind=model_selection.provider_kind,
        model=model_selection.model,
        base_model=model_selection.base_model,
        adapter_path=model_selection.adapter_path,
    )
    report_path = write_evaluation_result(run_dir, result, output)
    typer.echo(f"{result.summary}; report written to {report_path}")
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

    report_paths = list(report or [])
    if not report_paths:
        _fail("at least one --report is required")

    examples = load_dataset_examples(dataset)
    reports = [(path, load_evaluation_result(path)) for path in report_paths]
    records = build_workspace_staged_review_records(
        examples=examples,
        reports=reports,
        simple_failure_threshold=simple_failure_threshold,
        auto_accept_threshold=auto_accept_threshold,
    )

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

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")

    counts: dict[str, int] = {}
    for record in records:
        decision = str(record["auto_triage"]["decision"])
        counts[decision] = counts.get(decision, 0) + 1
    typer.echo(f"Wrote {len(records)} staged workspace review record(s) to {output}")
    typer.echo("Auto triage: " + _format_count_distribution(counts))


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
    baseline = load_evaluation_result(baseline_report)
    adapter = load_evaluation_result(adapter_report)
    comparison = compare_evaluation_results(
        baseline,
        adapter,
        minimum_score_delta=minimum_score_delta,
        minimum_metric_deltas=thresholds,
        require_adapter_passed=require_adapter_passed,
    )

    report_path = output or adapter_report.with_suffix(".comparison.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(comparison.as_record(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    typer.echo(f"{comparison.summary}; report written to {report_path}")
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
