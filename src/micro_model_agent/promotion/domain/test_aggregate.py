"""Domain invariant and event tests for the promotion bounded context."""

from __future__ import annotations

from datetime import datetime, UTC
from uuid import uuid4

import pytest

from micro_model_agent.promotion.domain.aggregate import ModelRegistry, PromotedModel
from micro_model_agent.promotion.domain.events import (
    ModelPackaged,
    ModelPromoted,
    PromotionGateFailed,
    PromotionGatePassed,
)
from micro_model_agent.promotion.domain.exceptions import GateThresholdNotMetError
from micro_model_agent.promotion.domain.services import PromotionGateService
from micro_model_agent.shared.domain.value_objects import EvaluationResult


def _model(artifact_id=None) -> PromotedModel:
    return PromotedModel(
        id=uuid4(),
        artifact_id=artifact_id or uuid4(),
        artifact_name="tinyllama-adapter",
        artifact_path="/tmp/adapter",
        base_model="tinyllama",
        promoted_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# ModelRegistry aggregate
# ---------------------------------------------------------------------------


def test_record_promotion_emits_model_promoted_event() -> None:
    registry = ModelRegistry()
    model = _model()
    registry.record_promotion(model)

    events = registry.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], ModelPromoted)
    assert events[0].model_id == model.id
    assert events[0].registry_id == registry.id


def test_list_models_returns_all_recorded() -> None:
    registry = ModelRegistry()
    m1 = _model()
    m2 = _model()
    registry.record_promotion(m1)
    registry.record_promotion(m2)

    models = registry.list_models()
    assert len(models) == 2
    assert {m.id for m in models} == {m1.id, m2.id}


def test_select_by_artifact_id() -> None:
    registry = ModelRegistry()
    artifact_id = uuid4()
    model = _model(artifact_id=artifact_id)
    registry.record_promotion(model)

    found = registry.select(artifact_id)
    assert found is not None
    assert found.artifact_id == artifact_id


def test_select_missing_returns_none() -> None:
    registry = ModelRegistry()
    assert registry.select(uuid4()) is None


def test_record_package_emits_model_packaged_event() -> None:
    registry = ModelRegistry()
    model = _model()
    registry.record_promotion(model)
    registry.pull_events()

    registry.record_package(model.id, "ollama")

    events = registry.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], ModelPackaged)
    assert events[0].model_id == model.id
    assert events[0].package_format == "ollama"


def test_evaluate_gate_pass_emits_passed_event() -> None:
    registry = ModelRegistry()
    artifact_id = uuid4()
    passed = registry.evaluate_gate(artifact_id, scores=[0.9, 0.85], minimum_score=0.8)

    assert passed is True
    events = registry.pull_events()
    assert any(isinstance(e, PromotionGatePassed) for e in events)
    assert not any(isinstance(e, PromotionGateFailed) for e in events)


def test_evaluate_gate_fail_emits_failed_event() -> None:
    registry = ModelRegistry()
    artifact_id = uuid4()
    passed = registry.evaluate_gate(artifact_id, scores=[0.9, 0.5], minimum_score=0.8)

    assert passed is False
    events = registry.pull_events()
    assert any(isinstance(e, PromotionGateFailed) for e in events)
    assert not any(isinstance(e, PromotionGatePassed) for e in events)


# ---------------------------------------------------------------------------
# PromotionGateService
# ---------------------------------------------------------------------------


def test_gate_service_passes_when_score_meets_threshold() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=True, summary="ok", score=0.9)
    assert svc.meets_threshold(result, 0.8) is True


def test_gate_service_fails_when_score_below_threshold() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=True, summary="low", score=0.7)
    assert svc.meets_threshold(result, 0.8) is False


def test_gate_service_fails_when_not_passed() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=False, summary="fail", score=0.95)
    assert svc.meets_threshold(result, 0.8) is False


def test_gate_service_fails_when_score_none() -> None:
    svc = PromotionGateService()
    result = EvaluationResult(passed=True, summary="no score", score=None)
    assert svc.meets_threshold(result, 0.0) is False


def test_gate_threshold_not_met_error() -> None:
    exc = GateThresholdNotMetError(score=0.6, minimum_score=0.8)
    assert "0.600" in str(exc)
    assert "0.800" in str(exc)
    assert exc.score == 0.6
    assert exc.minimum_score == 0.8
