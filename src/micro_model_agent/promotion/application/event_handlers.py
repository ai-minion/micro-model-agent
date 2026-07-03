"""Promotion application event handlers.

These handlers subscribe to evaluation-context events via the shared EventBus
and open or close the promotion gate based on threshold events.
"""

from __future__ import annotations

import logging

from micro_model_agent.evaluation.domain.events import ThresholdBreached, ThresholdMet
from micro_model_agent.promotion.domain.repository import ModelRegistryRepository

_log = logging.getLogger(__name__)


class OnThresholdEvent:
    """Subscribe to ``ThresholdMet`` / ``ThresholdBreached`` and update the registry gate.

    When an evaluation report's score meets the threshold the promotion gate
    is opened (recorded in the ``ModelRegistry`` aggregate).  When it misses
    the gate is closed.  The aggregate itself emits ``PromotionGatePassed`` or
    ``PromotionGateFailed`` events that can be forwarded to further handlers.
    """

    def __init__(
        self,
        *,
        registry_repo: ModelRegistryRepository,
        minimum_score: float = 0.8,
    ) -> None:
        self.registry_repo = registry_repo
        self.minimum_score = minimum_score

    async def handle_met(self, event: ThresholdMet) -> None:
        """Open the promotion gate for a report whose score met the threshold."""

        registry = await self.registry_repo.get_or_create()

        registry.evaluate_gate(
            artifact_id=event.report_id,  # use report_id as proxy until artifact_id is tracked
            scores=[event.score],
            minimum_score=self.minimum_score,
        )
        await self.registry_repo.save(registry)
        _log.info(
            "Promotion gate OPENED for report %s (score %.3f ≥ %.3f)",
            event.report_id,
            event.score,
            self.minimum_score,
        )

    async def handle_breached(self, event: ThresholdBreached) -> None:
        """Record a gate failure for a report whose score missed the threshold."""

        registry = await self.registry_repo.get_or_create()
        registry.evaluate_gate(
            artifact_id=event.report_id,
            scores=[event.score],
            minimum_score=event.threshold,
        )
        await self.registry_repo.save(registry)
        _log.info(
            "Promotion gate CLOSED for report %s (score %.3f < %.3f)",
            event.report_id,
            event.score,
            event.threshold,
        )
