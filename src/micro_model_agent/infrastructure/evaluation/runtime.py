"""Runtime evaluation workflow composition helpers."""

from __future__ import annotations

from micro_model_agent.application.evaluation_workflows import (
    RunEvaluationComparisonWorkflow,
    RunSyntheticEvaluationWorkflow,
    RunTraceEvaluationWorkflow,
    RunWorkspaceStagedEvaluationWorkflow,
    RunWorkspaceStagedReviewWorkflow,
)
from micro_model_agent.infrastructure.datasets.metadata import (
    LocalDatasetToolProfileSummarizer,
)
from micro_model_agent.infrastructure.evaluation.artifact import SyntheticEvaluationSuite
from micro_model_agent.infrastructure.evaluation.comparison import (
    LocalEvaluationComparisonReportWriter,
)
from micro_model_agent.infrastructure.evaluation.reports import (
    LocalEvaluationResultReader,
    LocalEvaluationResultWriter,
)
from micro_model_agent.infrastructure.evaluation.synthetic_behavior import (
    SyntheticBehaviorEvaluationSuite,
)
from micro_model_agent.infrastructure.evaluation.trace_behavior import (
    TraceBehaviorEvaluationSuite,
)
from micro_model_agent.infrastructure.evaluation.workspace_staged import (
    WorkspaceStagedEvaluationSuite,
)
from micro_model_agent.infrastructure.evaluation.workspace_staged_review import (
    LocalWorkspaceStagedReviewBuilder,
    LocalWorkspaceStagedReviewQueueWriter,
)
from micro_model_agent.infrastructure.persistence.dataset_store import LocalDatasetExampleReader
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS

__all__ = [
    "build_evaluation_comparison_workflow",
    "build_synthetic_evaluation_workflow",
    "build_trace_evaluation_workflow",
    "build_workspace_staged_evaluation_workflow",
    "build_workspace_staged_review_workflow",
    "default_evaluation_available_tools",
]


def default_evaluation_available_tools() -> tuple[str, ...]:
    """Return the default tool names used in behavior evaluation prompts."""

    return tuple(TOOL_ARGUMENT_CONTRACTS)


def build_synthetic_evaluation_workflow(
    *,
    pass_threshold: float,
) -> RunSyntheticEvaluationWorkflow:
    """Build the standard synthetic evaluation workflow."""

    return RunSyntheticEvaluationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        behavior_suite=SyntheticBehaviorEvaluationSuite(pass_threshold=pass_threshold),
        artifact_suite=SyntheticEvaluationSuite(),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        evaluation_writer=LocalEvaluationResultWriter(),
    )


def build_trace_evaluation_workflow(
    *,
    pass_threshold: float,
) -> RunTraceEvaluationWorkflow:
    """Build the standard trace-derived evaluation workflow."""

    return RunTraceEvaluationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        behavior_suite=TraceBehaviorEvaluationSuite(pass_threshold=pass_threshold),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        evaluation_writer=LocalEvaluationResultWriter(),
    )


def build_workspace_staged_evaluation_workflow(
    *,
    pass_threshold: float,
    rubric_version: str,
) -> RunWorkspaceStagedEvaluationWorkflow:
    """Build the standard staged workspace evaluation workflow."""

    return RunWorkspaceStagedEvaluationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        behavior_suite=WorkspaceStagedEvaluationSuite(
            pass_threshold=pass_threshold,
            rubric_version=rubric_version,
        ),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        evaluation_writer=LocalEvaluationResultWriter(),
    )


def build_workspace_staged_review_workflow() -> RunWorkspaceStagedReviewWorkflow:
    """Build the standard staged workspace review queue workflow."""

    return RunWorkspaceStagedReviewWorkflow(
        example_reader=LocalDatasetExampleReader(),
        evaluation_reader=LocalEvaluationResultReader(),
        review_builder=LocalWorkspaceStagedReviewBuilder(),
        review_writer=LocalWorkspaceStagedReviewQueueWriter(),
    )


def build_evaluation_comparison_workflow() -> RunEvaluationComparisonWorkflow:
    """Build the standard evaluation comparison workflow."""

    return RunEvaluationComparisonWorkflow(
        evaluation_reader=LocalEvaluationResultReader(),
        comparison_writer=LocalEvaluationComparisonReportWriter(),
    )
