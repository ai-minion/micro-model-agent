"""Evaluation context application ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from micro_model_agent.dataset.domain.value_objects import DatasetExample
from micro_model_agent.shared.domain.value_objects import EvaluationResult
from micro_model_agent.execution.application.ports import ModelProvider
from micro_model_agent.training.domain.value_objects import ModelArtifact


class EvaluationSuite(Protocol):
    """Runs a configured evaluation against a model artifact."""

    async def evaluate_artifact(self, artifact: ModelArtifact) -> EvaluationResult:
        """Evaluate a model artifact against configured tasks."""


class ModelBehaviorEvaluationSuite(Protocol):
    """Runs dataset examples against a model provider."""

    async def evaluate_model(
        self,
        model_provider: ModelProvider,
        examples: list[DatasetExample],
    ) -> EvaluationResult:
        """Evaluate model behavior on dataset examples."""


class EvaluationResultWriter(Protocol):
    """Persists evaluation reports."""

    def write_evaluation_result(
        self,
        run_dir: Path,
        result: EvaluationResult,
        output_path: Path | None = None,
    ) -> Path:
        """Persist one evaluation report and return its path."""


class EvaluationResultReader(Protocol):
    """Loads persisted evaluation reports."""

    def load_evaluation_result(self, path: Path) -> EvaluationResult:
        """Load one evaluation report."""


class EvaluationComparisonReportWriter(Protocol):
    """Persists evaluation comparison reports."""

    def write_evaluation_comparison_report(
        self,
        path: Path,
        record: dict[str, Any],
    ) -> None:
        """Write one JSON-ready evaluation comparison report."""


class WorkspaceStagedReviewBuilder(Protocol):
    """Builds staged workspace review records from examples and reports."""

    def build_workspace_staged_review_records(
        self,
        *,
        examples: list[DatasetExample],
        reports: list[tuple[Path, EvaluationResult]],
        simple_failure_threshold: float = 0.4,
        auto_accept_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Build JSON-ready staged workspace review records."""


class WorkspaceStagedReviewQueueWriter(Protocol):
    """Persists staged workspace review queue records."""

    def write_workspace_staged_review_records(
        self,
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        """Write staged workspace review records."""
