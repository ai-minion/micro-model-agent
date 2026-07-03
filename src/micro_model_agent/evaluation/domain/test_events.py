"""Tests for evaluation and promotion domain events and exceptions."""

from __future__ import annotations

from uuid import uuid4

import pytest

from micro_model_agent.evaluation.domain.events import (
    EvaluationCompleted,
    ThresholdBreached,
    ThresholdMet,
)
from micro_model_agent.evaluation.domain.exceptions import ScoreOutOfRangeError
from micro_model_agent.promotion.domain.events import (
    ModelPackaged,
    ModelPromoted,
    PromotionGateFailed,
    PromotionGatePassed,
)
from micro_model_agent.promotion.domain.exceptions import GateThresholdNotMetError
from micro_model_agent.shared.domain.domain_event import DomainEvent

# ---------------------------------------------------------------------------
# Evaluation events
# ---------------------------------------------------------------------------


def test_evaluation_completed_fields() -> None:
    report_id = uuid4()
    event = EvaluationCompleted(report_id=report_id, score=0.87)
    assert event.report_id == report_id
    assert event.score == 0.87
    assert isinstance(event, DomainEvent)


def test_threshold_met_fields() -> None:
    event = ThresholdMet(report_id=uuid4(), score=0.9)
    assert event.score == 0.9


def test_threshold_breached_fields() -> None:
    event = ThresholdBreached(report_id=uuid4(), score=0.6, threshold=0.8)
    assert event.score == 0.6
    assert event.threshold == 0.8


def test_all_evaluation_events_are_domain_events() -> None:
    report_id = uuid4()
    for event in [
        EvaluationCompleted(report_id=report_id, score=0.9),
        ThresholdMet(report_id=report_id, score=0.9),
        ThresholdBreached(report_id=report_id, score=0.5, threshold=0.8),
    ]:
        assert isinstance(event, DomainEvent)


def test_score_out_of_range_error_includes_score() -> None:
    exc = ScoreOutOfRangeError(1.5)
    assert "1.5" in str(exc)
    assert exc.score == 1.5


def test_score_out_of_range_is_domain_exception() -> None:
    from micro_model_agent.shared.domain.exceptions import DomainException
    with pytest.raises(DomainException):
        raise ScoreOutOfRangeError(2.0)


# ---------------------------------------------------------------------------
# Promotion events
# ---------------------------------------------------------------------------


def test_promotion_gate_passed_fields() -> None:
    registry_id = uuid4()
    artifact_id = uuid4()
    event = PromotionGatePassed(
        registry_id=registry_id, artifact_id=artifact_id, minimum_score=0.8
    )
    assert event.artifact_id == artifact_id
    assert event.minimum_score == 0.8
    assert isinstance(event, DomainEvent)


def test_promotion_gate_failed_fields() -> None:
    event = PromotionGateFailed(
        registry_id=uuid4(), artifact_id=uuid4(), minimum_score=0.85
    )
    assert event.minimum_score == 0.85


def test_model_promoted_fields() -> None:
    model_id = uuid4()
    event = ModelPromoted(registry_id=uuid4(), model_id=model_id)
    assert event.model_id == model_id


def test_model_packaged_fields() -> None:
    event = ModelPackaged(
        registry_id=uuid4(), model_id=uuid4(), package_format="ollama"
    )
    assert event.package_format == "ollama"


def test_all_promotion_events_are_domain_events() -> None:
    reg_id = uuid4()
    art_id = uuid4()
    mod_id = uuid4()
    for event in [
        PromotionGatePassed(registry_id=reg_id, artifact_id=art_id, minimum_score=0.8),
        PromotionGateFailed(registry_id=reg_id, artifact_id=art_id, minimum_score=0.8),
        ModelPromoted(registry_id=reg_id, model_id=mod_id),
        ModelPackaged(registry_id=reg_id, model_id=mod_id, package_format="ollama"),
    ]:
        assert isinstance(event, DomainEvent)


def test_gate_threshold_not_met_error() -> None:
    from micro_model_agent.shared.domain.exceptions import DomainException
    exc = GateThresholdNotMetError(score=0.65, minimum_score=0.8)
    assert exc.score == 0.65
    assert exc.minimum_score == 0.8
    assert isinstance(exc, DomainException)
