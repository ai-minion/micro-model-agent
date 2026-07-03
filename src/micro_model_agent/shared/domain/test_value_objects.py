"""Tests for shared domain value objects."""

from __future__ import annotations

from micro_model_agent.shared.domain.value_objects import EvaluationResult


def test_evaluation_result_passed_true() -> None:
    result = EvaluationResult(passed=True, summary="ok", score=0.95)
    assert result.passed is True
    assert result.score == 0.95
    assert result.summary == "ok"


def test_evaluation_result_failed() -> None:
    result = EvaluationResult(passed=False, summary="failed", score=0.3)
    assert result.passed is False


def test_evaluation_result_score_defaults_to_none() -> None:
    result = EvaluationResult(passed=True, summary="no score")
    assert result.score is None


def test_evaluation_result_details_default_empty() -> None:
    result = EvaluationResult(passed=True, summary="ok")
    assert result.details == {}


def test_evaluation_result_with_details() -> None:
    result = EvaluationResult(
        passed=True,
        summary="ok",
        score=1.0,
        details={"trace_id": "abc", "patch_applied": True},
    )
    assert result.details["trace_id"] == "abc"
    assert result.details["patch_applied"] is True


def test_evaluation_result_is_frozen() -> None:
    """EvaluationResult must be immutable (frozen dataclass)."""
    result = EvaluationResult(passed=True, summary="ok")
    try:
        result.passed = False  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass  # expected: frozen dataclass raises on mutation
