"""Compatibility facade for static agent application workflows."""

from micro_model_agent.application.agent_workflows import (
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
