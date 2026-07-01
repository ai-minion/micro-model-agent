"""Tests for baseline-vs-adapter evaluation comparisons."""

from __future__ import annotations

import pytest

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.infrastructure.evaluation.comparison import compare_evaluation_results


def test_compare_evaluation_results_passes_score_and_metric_thresholds() -> None:
    baseline = EvaluationResult(
        passed=False,
        summary="base",
        score=0.70,
        details={"metrics": {"correct_tool_rate": 0.60, "parse_success_rate": 0.80}},
    )
    adapter = EvaluationResult(
        passed=True,
        summary="adapter",
        score=0.84,
        details={"metrics": {"correct_tool_rate": 0.75, "parse_success_rate": 0.95}},
    )

    comparison = compare_evaluation_results(
        baseline,
        adapter,
        minimum_score_delta=0.10,
        minimum_metric_deltas={"correct_tool_rate": 0.10},
    )

    assert comparison.passed is True
    assert comparison.score_delta == pytest.approx(0.14)
    metric_deltas = {delta.name: delta for delta in comparison.metric_deltas}
    assert metric_deltas["correct_tool_rate"].delta == pytest.approx(0.15)
    assert metric_deltas["correct_tool_rate"].passed is True


def test_compare_evaluation_results_fails_missing_metric_threshold() -> None:
    baseline = EvaluationResult(
        passed=True,
        summary="base",
        score=0.80,
        details={"metrics": {"correct_tool_rate": 0.80}},
    )
    adapter = EvaluationResult(
        passed=True,
        summary="adapter",
        score=0.82,
        details={"metrics": {"correct_tool_rate": 0.82}},
    )

    comparison = compare_evaluation_results(
        baseline,
        adapter,
        minimum_metric_deltas={"valid_argument_rate": 0.05},
    )

    assert comparison.passed is False
    assert "metric 'valid_argument_rate' is missing from one or both reports" in comparison.errors
