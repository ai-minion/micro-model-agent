"""Promotion context domain services.

The promotion gate service is stateless and operates entirely on domain value
objects.  It has no infrastructure dependency and can be unit-tested directly.
"""

from __future__ import annotations

from micro_model_agent.shared.domain.value_objects import EvaluationResult


class PromotionGateService:
    """Evaluate whether a model artifact meets the promotion threshold.

    The service owns the ``minimum_score`` gate logic so the application
    workflow is a pure orchestrator and ``MinimumScorePromotionPolicy`` in the
    infrastructure layer can be retired once all callers migrate here.
    """

    def meets_threshold(self, evaluation: EvaluationResult, minimum_score: float) -> bool:
        """Return True when the evaluation passes and its score meets the minimum.

        A ``None`` score is treated as failing — the evaluation must explicitly
        record a numeric value.
        """
        return (
            evaluation.passed
            and evaluation.score is not None
            and evaluation.score >= minimum_score
        )
