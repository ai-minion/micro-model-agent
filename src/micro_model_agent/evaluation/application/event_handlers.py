"""Evaluation application event handlers.

These handlers subscribe to training-context events via the shared EventBus
and trigger evaluation runs when a new model artifact is produced.
"""

from __future__ import annotations

import logging

from micro_model_agent.evaluation.domain.aggregate import EvaluationReport, EvaluationThreshold
from micro_model_agent.evaluation.domain.repository import EvaluationReportRepository
from micro_model_agent.training.domain.events import ArtifactProduced

_log = logging.getLogger(__name__)

DEFAULT_MINIMUM_SCORE = 0.8


class OnArtifactProduced:
    """Subscribe to ``ArtifactProduced`` and create an EvaluationReport placeholder.

    In the full pipeline this handler would also dispatch a background evaluation
    job.  For now it records an EvaluationReport aggregate that downstream code
    (e.g. a scheduled evaluator) can discover and fill.
    """

    def __init__(
        self,
        *,
        report_repo: EvaluationReportRepository,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
    ) -> None:
        self.report_repo = report_repo
        self.minimum_score = minimum_score

    async def handle(self, event: ArtifactProduced) -> None:
        """Create and persist an EvaluationReport for the produced artifact."""

        run_id = str(event.job_id)
        threshold = EvaluationThreshold(minimum_score=self.minimum_score)
        report = EvaluationReport(run_id=run_id, threshold=threshold)
        await self.report_repo.add(report)
        _log.info(
            "EvaluationReport %s created for artifact %s (job %s)",
            report.id,
            event.artifact_id,
            event.job_id,
        )
