"""EvaluationReport — evaluation context aggregate root."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from micro_model_agent.evaluation.domain.events import (
    EvaluationCompleted,
    ThresholdBreached,
    ThresholdMet,
)
from micro_model_agent.evaluation.domain.exceptions import ScoreOutOfRangeError
from micro_model_agent.shared.domain.entity import Entity


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One scored result attached to an EvaluationReport."""

    category: str
    passed: bool
    score: float
    summary: str


@dataclass(frozen=True, slots=True)
class EvaluationThreshold:
    """Minimum-score contract for an evaluation run."""

    minimum_score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError(
                f"Threshold must be in [0, 1], got {self.minimum_score!r}"
            )


class EvaluationReport(Entity):
    """Aggregate root for a single evaluation run.

    Invariants enforced:
    - ``finalize()`` may only be called once.
    - Summary score must be in [0.0, 1.0].
    - Publishes ``ThresholdMet`` or ``ThresholdBreached`` based on the threshold.
    """

    def __init__(
        self,
        run_id: str,
        threshold: EvaluationThreshold,
        id: UUID | None = None,
    ) -> None:
        super().__init__(id)
        self.run_id = run_id
        self.threshold = threshold
        self._results: list[EvaluationResult] = []
        self.summary_score: float | None = None
        self._finalised = False

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def add_result(self, result: EvaluationResult) -> None:
        """Attach an individual scored result to the report."""

        if self._finalised:
            raise RuntimeError("Cannot add results to a finalised EvaluationReport")
        self._results.append(result)

    def finalize(self, summary_score: float) -> None:
        """Record the summary score and emit threshold events."""

        if self._finalised:
            raise RuntimeError("EvaluationReport is already finalised")
        if not 0.0 <= summary_score <= 1.0:
            raise ScoreOutOfRangeError(summary_score)

        self.summary_score = summary_score
        self._finalised = True
        self._events.append(
            EvaluationCompleted(report_id=self.id, score=summary_score)
        )
        if summary_score >= self.threshold.minimum_score:
            self._events.append(
                ThresholdMet(report_id=self.id, score=summary_score)
            )
        else:
            self._events.append(
                ThresholdBreached(
                    report_id=self.id,
                    score=summary_score,
                    threshold=self.threshold.minimum_score,
                )
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def results(self) -> tuple[EvaluationResult, ...]:
        """Immutable view of the individual scored results."""

        return tuple(self._results)

    @property
    def passed(self) -> bool:
        """True when the report is finalised and the score meets the threshold."""

        return (
            self._finalised
            and self.summary_score is not None
            and self.summary_score >= self.threshold.minimum_score
        )
