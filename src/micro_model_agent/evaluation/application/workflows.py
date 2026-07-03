"""Compatibility shim for application evaluation workflows.

All symbols are now defined in dedicated sub-modules:
- ``micro_model_agent.evaluation.application.compare``
- ``micro_model_agent.evaluation.application.synthetic``
- ``micro_model_agent.evaluation.application.traces``
- ``micro_model_agent.evaluation.application.workspace_staged``
"""

from __future__ import annotations

from micro_model_agent.evaluation.application.compare import (
    EvaluationComparisonResult,
    EvaluationMetricDelta,
    RunEvaluationComparisonRequest,
    RunEvaluationComparisonResult,
    RunEvaluationComparisonWorkflow,
    compare_evaluation_results,
)
from micro_model_agent.evaluation.application.synthetic import (
    RunSyntheticEvaluationRequest,
    RunSyntheticEvaluationResult,
    RunSyntheticEvaluationWorkflow,
    SyntheticBehaviorEvaluationSuite,
    SyntheticBehaviorExampleScore,
    SyntheticExampleScorer,
)
from micro_model_agent.evaluation.application.traces import (
    RunTraceEvaluationRequest,
    RunTraceEvaluationResult,
    RunTraceEvaluationWorkflow,
    TraceBehaviorEvaluationSuite,
    TraceBehaviorExampleScore,
    TraceExampleScorer,
)
from micro_model_agent.evaluation.application.workspace_staged import (
    WORKSPACE_STAGED_STAGE_NAMES,
    WORKSPACE_STAGED_SYSTEM_PROMPT,
    RunWorkspaceStagedEvaluationRequest,
    RunWorkspaceStagedEvaluationResult,
    RunWorkspaceStagedEvaluationWorkflow,
    RunWorkspaceStagedReviewBuildResult,
    RunWorkspaceStagedReviewRequest,
    RunWorkspaceStagedReviewWorkflow,
    RunWorkspaceStagedReviewWriteRequest,
    RunWorkspaceStagedReviewWriteResult,
    WorkspaceStagedEvaluationSuite,
    WorkspaceStagedExampleScore,
    WorkspaceStagedExampleScorer,
    workspace_staged_prompt_payload,
)

__all__ = [
    "EvaluationComparisonResult",
    "EvaluationMetricDelta",
    "RunEvaluationComparisonRequest",
    "RunEvaluationComparisonResult",
    "RunEvaluationComparisonWorkflow",
    "RunSyntheticEvaluationRequest",
    "RunSyntheticEvaluationResult",
    "RunSyntheticEvaluationWorkflow",
    "RunTraceEvaluationRequest",
    "RunTraceEvaluationResult",
    "RunTraceEvaluationWorkflow",
    "RunWorkspaceStagedEvaluationRequest",
    "RunWorkspaceStagedEvaluationResult",
    "RunWorkspaceStagedEvaluationWorkflow",
    "RunWorkspaceStagedReviewBuildResult",
    "RunWorkspaceStagedReviewRequest",
    "RunWorkspaceStagedReviewWorkflow",
    "RunWorkspaceStagedReviewWriteRequest",
    "RunWorkspaceStagedReviewWriteResult",
    "SyntheticBehaviorEvaluationSuite",
    "SyntheticBehaviorExampleScore",
    "SyntheticExampleScorer",
    "TraceBehaviorEvaluationSuite",
    "TraceBehaviorExampleScore",
    "TraceExampleScorer",
    "WORKSPACE_STAGED_STAGE_NAMES",
    "WORKSPACE_STAGED_SYSTEM_PROMPT",
    "WorkspaceStagedEvaluationSuite",
    "WorkspaceStagedExampleScore",
    "WorkspaceStagedExampleScorer",
    "compare_evaluation_results",
    "workspace_staged_prompt_payload",
]
