"""Compatibility facade for static agent application workflows."""

from micro_model_agent.application.agent import (
    DefaultWorkflowEvaluator,
    RunAgentWorkflow,
    TraceDatasetBuilder,
    label_from_workflow_result,
)

__all__ = [
    "DefaultWorkflowEvaluator",
    "RunAgentWorkflow",
    "TraceDatasetBuilder",
    "label_from_workflow_result",
]
