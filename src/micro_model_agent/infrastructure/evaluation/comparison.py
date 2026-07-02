"""Filesystem adapters and compatibility imports for evaluation comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from micro_model_agent.application.evaluation_workflows import (
    EvaluationComparisonResult,
    EvaluationMetricDelta,
    compare_evaluation_results,
)

__all__ = [
    "EvaluationComparisonResult",
    "EvaluationMetricDelta",
    "LocalEvaluationComparisonReportWriter",
    "compare_evaluation_results",
]


class LocalEvaluationComparisonReportWriter:
    """Filesystem adapter for writing evaluation comparison reports."""

    def write_evaluation_comparison_report(
        self,
        path: Path,
        record: dict[str, Any],
    ) -> None:
        """Write one evaluation comparison report as JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
