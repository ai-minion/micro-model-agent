"""Tests for evaluation runtime composition helpers."""

from __future__ import annotations

from micro_model_agent.application.evaluation_workflows import (
    RunEvaluationComparisonWorkflow,
    RunSyntheticEvaluationWorkflow,
    RunTraceEvaluationWorkflow,
    RunWorkspaceStagedEvaluationWorkflow,
    RunWorkspaceStagedReviewWorkflow,
)
from micro_model_agent.infrastructure.evaluation.runtime import (
    build_evaluation_comparison_workflow,
    build_synthetic_evaluation_workflow,
    build_trace_evaluation_workflow,
    build_workspace_staged_evaluation_workflow,
    build_workspace_staged_review_workflow,
    default_evaluation_available_tools,
)


def test_evaluation_runtime_builds_standard_workflows() -> None:
    assert default_evaluation_available_tools()
    assert isinstance(
        build_synthetic_evaluation_workflow(pass_threshold=0.8),
        RunSyntheticEvaluationWorkflow,
    )
    assert isinstance(
        build_trace_evaluation_workflow(pass_threshold=0.8),
        RunTraceEvaluationWorkflow,
    )
    assert isinstance(
        build_workspace_staged_evaluation_workflow(
            pass_threshold=0.8,
            rubric_version="legacy",
        ),
        RunWorkspaceStagedEvaluationWorkflow,
    )
    assert isinstance(build_workspace_staged_review_workflow(), RunWorkspaceStagedReviewWorkflow)
    assert isinstance(build_evaluation_comparison_workflow(), RunEvaluationComparisonWorkflow)
