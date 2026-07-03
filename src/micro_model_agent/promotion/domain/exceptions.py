"""Domain exceptions for the promotion bounded context."""

from __future__ import annotations

from micro_model_agent.shared.domain.exceptions import DomainException


class GateThresholdNotMetError(DomainException):
    """Raised when a promotion gate decision rejects an artifact."""

    def __init__(self, score: float, minimum_score: float) -> None:
        super().__init__(
            f"Score {score:.3f} does not meet the minimum threshold {minimum_score:.3f}"
        )
        self.score = score
        self.minimum_score = minimum_score
