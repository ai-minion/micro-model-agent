"""Backward-compatible re-export hub for application ports.

All port interfaces have canonical homes in their bounded-context packages:

  execution.application.ports  —  CodingAgentTask, ModelProvider, ToolExecutor, …
  dataset.application.ports    —  DatasetExampleStore, TraceReviewRecord, …
  training.application.ports   —  TrainingRunner, ArtifactStore
  evaluation.application.ports —  EvaluationSuite, EvaluationResultWriter, …
  promotion.application.ports  —  PromotedArtifactRecord, OllamaPackageRecord, …
  repository_ops.application.ports — SemanticRetriever

New code should import directly from the bounded-context packages.
"""

from __future__ import annotations

from micro_model_agent.dataset.application.ports import (  # noqa: F401
    DatasetBuilder,
    DatasetExampleReader,
    DatasetExampleStore,
    DatasetExampleWriter,
    DatasetExporter,
    DatasetFileHasher,
    DatasetMerger,
    DatasetRelabeler,
    DatasetToolProfileSummarizer,
    DatasetValidator,
    SyntheticDataGenerator,
    TraceDatasetExampleExporter,
    TraceDatasetExportValidator,
    TraceReviewReader,
    TraceReviewRecord,
    TraceReviewWriter,
)
from micro_model_agent.evaluation.application.ports import (  # noqa: F401
    EvaluationComparisonReportWriter,
    EvaluationResultReader,
    EvaluationResultWriter,
    EvaluationSuite,
    ModelBehaviorEvaluationSuite,
    WorkspaceStagedReviewBuilder,
    WorkspaceStagedReviewQueueWriter,
)
from micro_model_agent.execution.application.ports import (  # noqa: F401
    CodingAgentResult,
    CodingAgentTask,
    CodingWorkflowRunner,
    ModelProvider,
    ToolExecutor,
    TraceStore,
    WorkflowEvaluator,
    WorkflowTraceReader,
)
from micro_model_agent.promotion.application.ports import (  # noqa: F401
    ModelConfigurationUpdate,
    ModelPromotionPolicy,
    OllamaPackageRecord,
    PromotedAdapterPackager,
    PromotedArtifactRecord,
    PromotionGateResultWriter,
    PromotionRegistryReader,
    PromotionRegistryWriter,
    RepositoryModelConfigurationWriter,
    TrainingRunArtifactReader,
)
from micro_model_agent.repository_ops.application.ports import (  # noqa: F401
    SemanticRetriever,
)
from micro_model_agent.training.application.ports import (  # noqa: F401
    ArtifactStore,
    TrainingRunner,
)

__all__ = [
    "ArtifactStore",
    "CodingAgentResult",
    "CodingAgentTask",
    "CodingWorkflowRunner",
    "DatasetBuilder",
    "DatasetExampleReader",
    "DatasetExampleStore",
    "DatasetExampleWriter",
    "DatasetExporter",
    "DatasetFileHasher",
    "DatasetMerger",
    "DatasetRelabeler",
    "DatasetToolProfileSummarizer",
    "DatasetValidator",
    "EvaluationComparisonReportWriter",
    "EvaluationResultReader",
    "EvaluationResultWriter",
    "EvaluationSuite",
    "ModelBehaviorEvaluationSuite",
    "ModelConfigurationUpdate",
    "ModelPromotionPolicy",
    "ModelProvider",
    "OllamaPackageRecord",
    "PromotedAdapterPackager",
    "PromotedArtifactRecord",
    "PromotionGateResultWriter",
    "PromotionRegistryReader",
    "PromotionRegistryWriter",
    "RepositoryModelConfigurationWriter",
    "SemanticRetriever",
    "SyntheticDataGenerator",
    "ToolExecutor",
    "TraceDatasetExampleExporter",
    "TraceDatasetExportValidator",
    "TraceReviewReader",
    "TraceReviewRecord",
    "TraceReviewWriter",
    "TraceStore",
    "TrainingRunArtifactReader",
    "TrainingRunner",
    "WorkflowEvaluator",
    "WorkflowTraceReader",
    "WorkspaceStagedReviewBuilder",
    "WorkspaceStagedReviewQueueWriter",
]
