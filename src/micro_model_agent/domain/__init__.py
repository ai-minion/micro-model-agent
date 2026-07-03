"""Domain layer backward-compat re-exports.

All types now live in their bounded-context packages.  This package re-exports
everything for any remaining old-path callers.
"""
from __future__ import annotations

from micro_model_agent.shared.domain.value_objects import EvaluationResult  # noqa: F401
from micro_model_agent.execution.domain.value_objects import (  # noqa: F401
    AgentProfile,
    ModelProfile,
    ToolCall,
    ToolDefinition,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)
from micro_model_agent.repository_ops.domain.value_objects import (  # noqa: F401
    RepositoryProfile,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    SemanticSearchResult,
)
from micro_model_agent.dataset.domain.value_objects import (  # noqa: F401
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    DatasetSplit,
    DatasetSplitName,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.training.domain.value_objects import (  # noqa: F401
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
