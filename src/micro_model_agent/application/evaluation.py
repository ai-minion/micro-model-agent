"""Application workflows for evaluation operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from micro_model_agent.application.ports import (
    EvaluationComparisonReportWriter,
    EvaluationResultReader,
)
from micro_model_agent.domain.contracts import EvaluationResult


@dataclass(frozen=True, slots=True)
class EvaluationMetricDelta:
    """Delta for one numeric evaluation metric."""

    name: str
    baseline: float
    adapter: float
    delta: float
    minimum_delta: float | None = None
    passed: bool | None = None

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-ready metric comparison record."""

        return {
            "name": self.name,
            "baseline": self.baseline,
            "adapter": self.adapter,
            "delta": self.delta,
            "minimum_delta": self.minimum_delta,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EvaluationComparisonResult:
    """Pass/fail comparison between a base model and trained adapter report."""

    passed: bool
    summary: str
    baseline_passed: bool
    adapter_passed: bool
    baseline_score: float | None
    adapter_score: float | None
    score_delta: float | None
    minimum_score_delta: float
    metric_deltas: tuple[EvaluationMetricDelta, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-ready comparison report."""

        return {
            "passed": self.passed,
            "summary": self.summary,
            "baseline": {
                "passed": self.baseline_passed,
                "score": self.baseline_score,
            },
            "adapter": {
                "passed": self.adapter_passed,
                "score": self.adapter_score,
            },
            "score_delta": self.score_delta,
            "minimum_score_delta": self.minimum_score_delta,
            "metric_deltas": [delta.as_record() for delta in self.metric_deltas],
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class RunEvaluationComparisonRequest:
    """Request for comparing two persisted evaluation reports."""

    baseline_report_path: Path
    adapter_report_path: Path
    output_path: Path
    minimum_score_delta: float = 0.0
    minimum_metric_deltas: dict[str, float] = field(default_factory=dict)
    require_adapter_passed: bool = True


@dataclass(frozen=True, slots=True)
class RunEvaluationComparisonResult:
    """Result returned after comparing two evaluation reports."""

    output_path: Path
    comparison: EvaluationComparisonResult


class RunEvaluationComparisonWorkflow:
    """Load, compare, and persist evaluation comparison reports."""

    def __init__(
        self,
        *,
        evaluation_reader: EvaluationResultReader,
        comparison_writer: EvaluationComparisonReportWriter,
    ) -> None:
        self.evaluation_reader = evaluation_reader
        self.comparison_writer = comparison_writer

    def run(
        self,
        request: RunEvaluationComparisonRequest,
    ) -> RunEvaluationComparisonResult:
        """Run evaluation comparison."""

        baseline = self.evaluation_reader.load_evaluation_result(
            request.baseline_report_path
        )
        adapter = self.evaluation_reader.load_evaluation_result(request.adapter_report_path)
        comparison = compare_evaluation_results(
            baseline,
            adapter,
            minimum_score_delta=request.minimum_score_delta,
            minimum_metric_deltas=request.minimum_metric_deltas,
            require_adapter_passed=request.require_adapter_passed,
        )
        self.comparison_writer.write_evaluation_comparison_report(
            request.output_path,
            comparison.as_record(),
        )
        return RunEvaluationComparisonResult(
            output_path=request.output_path,
            comparison=comparison,
        )


def compare_evaluation_results(
    baseline: EvaluationResult,
    adapter: EvaluationResult,
    *,
    minimum_score_delta: float = 0.0,
    minimum_metric_deltas: dict[str, float] | None = None,
    require_adapter_passed: bool = True,
) -> EvaluationComparisonResult:
    """Compare two evaluation reports and apply score/metric improvement gates."""

    if minimum_score_delta < 0.0:
        raise ValueError("minimum_score_delta must be non-negative")

    metric_thresholds = dict(minimum_metric_deltas or {})
    errors: list[str] = []
    score_delta = _score_delta(baseline.score, adapter.score)

    if require_adapter_passed and not adapter.passed:
        errors.append("adapter report did not pass")
    if score_delta is None:
        errors.append("baseline and adapter reports must both include scores")
    elif score_delta < minimum_score_delta:
        errors.append(
            f"score delta {score_delta:.4f} is below minimum {minimum_score_delta:.4f}"
        )

    metric_deltas = _metric_deltas(
        _numeric_metrics(baseline),
        _numeric_metrics(adapter),
        metric_thresholds,
    )
    compared_metric_names = {delta.name for delta in metric_deltas}
    for metric_name in sorted(metric_thresholds):
        if metric_name not in compared_metric_names:
            errors.append(f"metric {metric_name!r} is missing from one or both reports")
    for delta in metric_deltas:
        if delta.passed is False and delta.minimum_delta is not None:
            errors.append(
                f"metric {delta.name!r} delta {delta.delta:.4f} is below "
                f"minimum {delta.minimum_delta:.4f}"
            )

    passed = not errors
    summary = _summary(passed, score_delta, minimum_score_delta, metric_deltas)
    return EvaluationComparisonResult(
        passed=passed,
        summary=summary,
        baseline_passed=baseline.passed,
        adapter_passed=adapter.passed,
        baseline_score=baseline.score,
        adapter_score=adapter.score,
        score_delta=score_delta,
        minimum_score_delta=minimum_score_delta,
        metric_deltas=tuple(metric_deltas),
        errors=tuple(errors),
    )


def _score_delta(baseline_score: float | None, adapter_score: float | None) -> float | None:
    if baseline_score is None or adapter_score is None:
        return None
    return adapter_score - baseline_score


def _numeric_metrics(result: EvaluationResult) -> dict[str, float]:
    metrics = result.details.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(name, str) and isinstance(value, int | float)
    }


def _metric_deltas(
    baseline_metrics: dict[str, float],
    adapter_metrics: dict[str, float],
    minimum_metric_deltas: dict[str, float],
) -> list[EvaluationMetricDelta]:
    metric_names = sorted(set(baseline_metrics) & set(adapter_metrics))
    deltas: list[EvaluationMetricDelta] = []
    for metric_name in metric_names:
        minimum_delta = minimum_metric_deltas.get(metric_name)
        delta = adapter_metrics[metric_name] - baseline_metrics[metric_name]
        deltas.append(
            EvaluationMetricDelta(
                name=metric_name,
                baseline=baseline_metrics[metric_name],
                adapter=adapter_metrics[metric_name],
                delta=delta,
                minimum_delta=minimum_delta,
                passed=None if minimum_delta is None else delta >= minimum_delta,
            )
        )
    return deltas


def _summary(
    passed: bool,
    score_delta: float | None,
    minimum_score_delta: float,
    metric_deltas: list[EvaluationMetricDelta],
) -> str:
    score_text = "unknown" if score_delta is None else f"{score_delta:+.2f}"
    checked_metrics = sum(1 for delta in metric_deltas if delta.minimum_delta is not None)
    status = "passed" if passed else "failed"
    return (
        f"evaluation comparison {status}: score delta {score_text} "
        f"(minimum +{minimum_score_delta:.2f}); checked {checked_metrics} metric threshold(s)"
    )
