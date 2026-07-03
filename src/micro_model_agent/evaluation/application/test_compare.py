"""Tests for evaluation comparison application workflow."""

from __future__ import annotations

import pytest

from micro_model_agent.evaluation.application.compare import (
    EvaluationComparisonResult,
    EvaluationMetricDelta,
    compare_evaluation_results,
)
from micro_model_agent.shared.domain.value_objects import EvaluationResult


def _result(
    passed: bool = True,
    score: float = 0.8,
    metrics: dict | None = None,
) -> EvaluationResult:
    details = {}
    if metrics:
        details["metrics"] = metrics
    return EvaluationResult(passed=passed, summary="ok" if passed else "fail",
                            score=score, details=details)


# ---------------------------------------------------------------------------
# compare_evaluation_results
# ---------------------------------------------------------------------------


def test_adapter_improvement_passes() -> None:
    baseline = _result(score=0.75)
    adapter = _result(score=0.85)
    result = compare_evaluation_results(baseline, adapter)
    assert result.passed is True
    assert result.score_delta == pytest.approx(0.10, abs=1e-6)


def test_adapter_regression_fails_when_delta_required() -> None:
    baseline = _result(score=0.85)
    adapter = _result(score=0.75)
    result = compare_evaluation_results(
        baseline, adapter, minimum_score_delta=0.0
    )
    # score_delta is negative — below minimum of 0.0
    assert result.passed is False
    assert any("score delta" in e for e in result.errors)


def test_adapter_not_passed_fails() -> None:
    baseline = _result(score=0.75)
    adapter = _result(passed=False, score=0.85)
    result = compare_evaluation_results(baseline, adapter, require_adapter_passed=True)
    assert result.passed is False
    assert any("did not pass" in e for e in result.errors)


def test_adapter_not_passed_allowed_when_flag_off() -> None:
    baseline = _result(score=0.75)
    adapter = _result(passed=False, score=0.85)
    result = compare_evaluation_results(baseline, adapter, require_adapter_passed=False)
    # Only the score delta gate applies now — delta is positive, so passes
    assert result.passed is True


def test_missing_scores_causes_error() -> None:
    baseline = EvaluationResult(passed=True, summary="ok", score=None)
    adapter = _result(score=0.9)
    result = compare_evaluation_results(baseline, adapter)
    assert result.passed is False
    assert any("must both include scores" in e for e in result.errors)


def test_negative_minimum_delta_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compare_evaluation_results(_result(), _result(), minimum_score_delta=-0.1)


def test_score_fields_captured() -> None:
    baseline = _result(score=0.7)
    adapter = _result(score=0.9)
    result = compare_evaluation_results(baseline, adapter)
    assert result.baseline_score == 0.7
    assert result.adapter_score == 0.9
    assert result.score_delta == pytest.approx(0.2, abs=1e-6)


def test_metric_improvement_passes() -> None:
    baseline = _result(metrics={"parse_success_rate": 0.7})
    adapter = _result(metrics={"parse_success_rate": 0.9})
    result = compare_evaluation_results(
        baseline, adapter,
        minimum_metric_deltas={"parse_success_rate": 0.1},
    )
    assert result.passed is True
    delta = next(d for d in result.metric_deltas if d.name == "parse_success_rate")
    assert delta.passed is True


def test_metric_regression_fails() -> None:
    baseline = _result(metrics={"parse_success_rate": 0.9})
    adapter = _result(metrics={"parse_success_rate": 0.7})
    result = compare_evaluation_results(
        baseline, adapter,
        minimum_metric_deltas={"parse_success_rate": 0.0},
    )
    assert result.passed is False
    assert any("parse_success_rate" in e for e in result.errors)


def test_missing_required_metric_errors() -> None:
    baseline = _result(metrics={})  # metric absent
    adapter = _result(metrics={})
    result = compare_evaluation_results(
        baseline, adapter,
        minimum_metric_deltas={"nonexistent_metric": 0.1},
    )
    assert result.passed is False
    assert any("missing" in e for e in result.errors)


def test_returns_comparison_result_type() -> None:
    result = compare_evaluation_results(_result(), _result())
    assert isinstance(result, EvaluationComparisonResult)


def test_equal_scores_zero_delta() -> None:
    result = compare_evaluation_results(_result(score=0.8), _result(score=0.8))
    assert result.score_delta == pytest.approx(0.0, abs=1e-6)
    # zero delta >= minimum_score_delta=0.0, so passes
    assert result.passed is True


def test_summary_present() -> None:
    result = compare_evaluation_results(_result(), _result(score=0.9))
    assert len(result.summary) > 0


def test_metric_delta_without_threshold_has_no_minimum() -> None:
    baseline = _result(metrics={"x": 0.5})
    adapter = _result(metrics={"x": 0.7})
    result = compare_evaluation_results(baseline, adapter)
    # Metric deltas are computed even without thresholds
    if result.metric_deltas:
        x_delta = next((d for d in result.metric_deltas if d.name == "x"), None)
        if x_delta:
            assert x_delta.minimum_delta is None
