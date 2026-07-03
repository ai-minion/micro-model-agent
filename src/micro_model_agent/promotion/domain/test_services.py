"""Domain invariant and event tests for the promotion bounded context."""

from __future__ import annotations

import pytest

from micro_model_agent.shared.domain.value_objects import EvaluationResult
from micro_model_agent.promotion.domain.services import PromotionGateService


def test_meets_threshold_passes_when_score_ok() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=True, summary="ok", score=0.95)
    assert svc.meets_threshold(result, minimum_score=0.9) is True


def test_meets_threshold_fails_when_score_low() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=True, summary="low", score=0.7)
    assert svc.meets_threshold(result, minimum_score=0.9) is False


def test_meets_threshold_fails_when_not_passed() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=False, summary="fail", score=0.95)
    assert svc.meets_threshold(result, minimum_score=0.9) is False


def test_meets_threshold_fails_when_score_none() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=True, summary="no score", score=None)
    assert svc.meets_threshold(result, minimum_score=0.0) is False


def test_meets_threshold_exact_boundary() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=True, summary="exact", score=0.8)
    assert svc.meets_threshold(result, minimum_score=0.8) is True
