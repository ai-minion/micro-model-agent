"""ModelRegistry — promotion context aggregate root."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from micro_model_agent.promotion.domain.events import (
    ModelPackaged,
    ModelPromoted,
    PromotionGateFailed,
    PromotionGatePassed,
)
from micro_model_agent.promotion.domain.services import PromotionGateService
from micro_model_agent.shared.domain.entity import Entity


@dataclass(frozen=True, slots=True)
class PromotedModel:
    """A model artifact that has passed the promotion gate."""

    id: UUID
    artifact_id: UUID
    artifact_name: str
    artifact_path: str
    base_model: str
    promoted_at: datetime


class ModelRegistry(Entity):
    """Aggregate root that tracks all promoted model artifacts.

    Invariants enforced:
    - ``record_promotion`` deduplicates by ``artifact_id``.
    - Gate decisions use ``PromotionGateService`` from the domain layer.
    """

    def __init__(self, id: UUID | None = None) -> None:
        super().__init__(id)
        self._models: dict[UUID, PromotedModel] = {}
        self._gate_service = PromotionGateService()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def record_promotion(self, model: PromotedModel) -> None:
        """Record a promoted model in the registry."""

        self._models[model.id] = model
        self._events.append(
            ModelPromoted(registry_id=self.id, model_id=model.id)
        )

    def record_package(self, model_id: UUID, package_format: str) -> None:
        """Record that a model was packaged for an inference runtime."""

        self._events.append(
            ModelPackaged(
                registry_id=self.id,
                model_id=model_id,
                package_format=package_format,
            )
        )

    def evaluate_gate(
        self,
        artifact_id: UUID,
        scores: list[float],
        minimum_score: float,
    ) -> bool:
        """Run the promotion gate and emit the appropriate domain event.

        Returns True when all scores meet the minimum threshold.
        """

        from micro_model_agent.shared.domain.value_objects import EvaluationResult

        all_pass = all(
            self._gate_service.meets_threshold(
                EvaluationResult(
                    passed=s >= minimum_score,
                    summary="",
                    score=s,
                ),
                minimum_score,
            )
            for s in scores
        )
        if all_pass:
            self._events.append(
                PromotionGatePassed(
                    registry_id=self.id,
                    artifact_id=artifact_id,
                    minimum_score=minimum_score,
                )
            )
        else:
            self._events.append(
                PromotionGateFailed(
                    registry_id=self.id,
                    artifact_id=artifact_id,
                    minimum_score=minimum_score,
                )
            )
        return all_pass

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_models(self) -> tuple[PromotedModel, ...]:
        """Return all promoted models in insertion order."""

        return tuple(self._models.values())

    def select(self, artifact_id: UUID) -> PromotedModel | None:
        """Return the first model whose artifact_id matches, or None."""

        return next(
            (m for m in self._models.values() if m.artifact_id == artifact_id),
            None,
        )
