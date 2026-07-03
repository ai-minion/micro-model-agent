"""Domain events for the evaluation bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from micro_model_agent.shared.domain.domain_event import DomainEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationCompleted(DomainEvent):
    """Raised when an EvaluationReport is finalised with a summary score."""

    report_id: UUID
    score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdMet(DomainEvent):
    """Raised (alongside EvaluationCompleted) when the score meets the threshold.

    This event is the Published Language token consumed by the promotion context
    to open the promotion gate.
    """

    report_id: UUID
    score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ThresholdBreached(DomainEvent):
    """Raised (alongside EvaluationCompleted) when the score misses the threshold.

    The promotion context subscribes to this event to close the gate.
    """

    report_id: UUID
    score: float
    threshold: float
