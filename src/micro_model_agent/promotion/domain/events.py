"""Domain events for the promotion bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from micro_model_agent.shared.domain.domain_event import DomainEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class PromotionGatePassed(DomainEvent):
    """Raised when all evaluation reports clear the minimum-score threshold."""

    registry_id: UUID
    artifact_id: UUID
    minimum_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PromotionGateFailed(DomainEvent):
    """Raised when at least one evaluation report misses the threshold."""

    registry_id: UUID
    artifact_id: UUID
    minimum_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelPromoted(DomainEvent):
    """Raised when a PromotedModel is recorded in the ModelRegistry."""

    registry_id: UUID
    model_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelPackaged(DomainEvent):
    """Raised when a promoted model is packaged for an inference runtime."""

    registry_id: UUID
    model_id: UUID
    package_format: str
