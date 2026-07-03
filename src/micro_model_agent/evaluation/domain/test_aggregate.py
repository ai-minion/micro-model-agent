"""Domain invariant and event tests for the evaluation bounded context."""

from __future__ import annotations

import pytest

from micro_model_agent.evaluation.domain.aggregate import (
    EvaluationReport,
    EvaluationResult,
    EvaluationThreshold,
)
from micro_model_agent.evaluation.domain.events import (
    EvaluationCompleted,
    ThresholdBreached,
    ThresholdMet,
)
from micro_model_agent.evaluation.domain.exceptions import ScoreOutOfRangeError


def test_finalize_emits_completed_and_threshold_met() -> None:
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="run-1", threshold=threshold)
    report.finalize(0.9)

    events = report.pull_events()
    assert len(events) == 2
    kinds = {type(e) for e in events}
    assert EvaluationCompleted in kinds
    assert ThresholdMet in kinds


def test_finalize_emits_threshold_breached_when_below() -> None:
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="run-1", threshold=threshold)
    report.finalize(0.6)

    events = report.pull_events()
    kinds = {type(e) for e in events}
    assert ThresholdBreached in kinds
    assert ThresholdMet not in kinds

    breached = next(e for e in events if isinstance(e, ThresholdBreached))
    assert breached.threshold == 0.8
    assert breached.score == 0.6


def test_finalize_twice_raises() -> None:
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="run-1", threshold=threshold)
    report.finalize(0.9)
    with pytest.raises(RuntimeError):
        report.finalize(0.9)


def test_score_out_of_range_raises() -> None:
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="run-1", threshold=threshold)
    with pytest.raises(ScoreOutOfRangeError):
        report.finalize(1.5)


def test_threshold_invalid_range_raises() -> None:
    with pytest.raises(ValueError):
        EvaluationThreshold(minimum_score=1.1)


def test_passed_reflects_threshold() -> None:
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="run-1", threshold=threshold)
    assert report.passed is False  # not finalised yet
    report.finalize(0.85)
    assert report.passed is True


def test_add_result_accumulates() -> None:
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="run-1", threshold=threshold)
    report.add_result(EvaluationResult(category="tools", passed=True, score=0.9, summary="ok"))
    assert len(report.results) == 1


def test_add_result_after_finalize_raises() -> None:
    threshold = EvaluationThreshold(minimum_score=0.8)
    report = EvaluationReport(run_id="run-1", threshold=threshold)
    report.finalize(0.9)
    with pytest.raises(RuntimeError):
        report.add_result(EvaluationResult(category="x", passed=True, score=1.0, summary=""))
