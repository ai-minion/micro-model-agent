"""Domain exceptions for the evaluation bounded context."""

from __future__ import annotations

from micro_model_agent.shared.domain.exceptions import DomainException


class ScoreOutOfRangeError(DomainException):
    """Raised when a summary score is not in the [0.0, 1.0] range."""

    def __init__(self, score: float) -> None:
        super().__init__(f"Evaluation score {score!r} must be in [0.0, 1.0]")
        self.score = score
