"""Ports used by application services.

A port is an interface the application layer depends on. In Python, a
``Protocol`` says "any object with these methods is acceptable", which keeps the
workflow code independent from the concrete storage, model, and tool classes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from micro_model_agent.domain.contracts import (
    EvaluationResult,
    RetrievalQuery,
    SemanticSearchResult,
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowTrace,
)
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    DatasetSplit,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.domain.training import ModelArtifact, TrainingConfig, TrainingRun


@dataclass(frozen=True, slots=True)
class CodingAgentTask:
    """Input for a coding workflow runner."""

    goal: str
    # dry_run means "validate the patch but do not write it to disk".
    dry_run: bool = True
    require_approval: bool = True
    expected_changed_files: list[str] = field(default_factory=list)
    verification_command_name: str | None = None
    semantic_intent: str = "code"
    semantic_limit: int = 5


@dataclass(frozen=True, slots=True)
class CodingAgentResult:
    """Structured result returned by a coding workflow runner."""

    trace_id: UUID
    ok: bool
    summary: str
    patch_applied: bool
    verification_passed: bool | None
    changed_files: list[str]
    trace: WorkflowTrace


@dataclass(frozen=True, slots=True)
class PromotedArtifactRecord:
    """Application-owned view of one promoted model registry record."""

    artifact_id: UUID
    artifact_name: str
    artifact_path: str
    artifact_kind: str
    base_model: str
    run_dir: str
    promotion_report_path: str
    evaluation_report_paths: tuple[str, ...]
    minimum_score: float
    reviewer_notes: str | None = None
    approved_by: str | None = None
    id: UUID | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ModelConfigurationUpdate:
    """Application-owned result for repository model configuration updates."""

    ok: bool
    config_path: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class OllamaPackageRecord:
    """Application-owned result for packaging a promoted adapter for Ollama."""

    model_name: str
    base_model: str
    adapter_path: str
    modelfile_path: str
    manifest_path: str
    command: tuple[str, ...]
    created: bool
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceReviewRecord:
    """Application-owned record for one human trace review."""

    trace_id: str
    label: DatasetLabel
    corrected_target: dict[str, Any] | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ModelProvider(Protocol):
    """Anything that can turn a prompt into model text."""

    async def complete(self, prompt: str) -> str:
        """Generate a model completion for a workflow prompt."""


class ToolExecutor(Protocol):
    """Anything that can run a typed tool call and return a typed result."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a typed tool call."""


class SemanticRetriever(Protocol):
    """Search service for model-facing repository or documentation context."""

    async def search(self, query: RetrievalQuery) -> SemanticSearchResult:
        """Run structured semantic retrieval."""


class WorkflowEvaluator(Protocol):
    """Evaluator for judging whether a finished workflow succeeded."""

    async def evaluate(self, trace: WorkflowTrace) -> EvaluationResult:
        """Evaluate a completed workflow."""


class TraceStore(Protocol):
    """Persistence boundary for workflow traces."""

    async def save(self, trace: WorkflowTrace) -> None:
        """Persist a workflow trace."""

    async def get(self, trace_id: str) -> WorkflowTrace | None:
        """Load a workflow trace by id."""


class WorkflowTraceReader(Protocol):
    """Loads stored workflow traces for application workflows."""

    async def list_workflow_traces(self) -> list[WorkflowTrace]:
        """List stored workflow traces."""


class TraceReviewReader(Protocol):
    """Loads stored human trace reviews."""

    async def latest_trace_reviews_by_trace_id(self) -> Mapping[str, object]:
        """Return the newest review record for each trace id."""


class TraceReviewWriter(Protocol):
    """Persists human trace review records."""

    async def save_trace_review(self, path: Path, review: TraceReviewRecord) -> None:
        """Persist one trace review record."""


class TraceDatasetExampleExporter(Protocol):
    """Converts workflow traces into dataset examples."""

    def export_trace_dataset_examples(
        self,
        traces: list[WorkflowTrace],
        *,
        kind: DatasetExampleKind,
        label_mode: str,
        reviews_by_trace_id: Mapping[str, object],
        outcome: OutcomeLabel | None = None,
        quality: QualityLabel | None = None,
        workflow_status: WorkflowStatus | None = None,
        require_tool_call: bool = False,
        max_examples: int | None = None,
    ) -> list[DatasetExample]:
        """Export trace-derived dataset examples."""


class TraceDatasetExportValidator(Protocol):
    """Validates trace-derived dataset examples."""

    def validate_trace_dataset_examples(self, examples: list[DatasetExample]) -> list[str]:
        """Return trace export validation errors."""


class CodingWorkflowRunner(Protocol):
    """Application port for an implementation that can run a coding workflow."""

    trace_store: TraceStore

    async def run(self, task: CodingAgentTask) -> CodingAgentResult:
        """Run a coding workflow task."""


class DatasetExampleStore(Protocol):
    """Persistence boundary for examples used in training/evaluation datasets."""

    async def save(self, example: DatasetExample) -> None:
        """Persist a training or evaluation dataset example."""

    async def list(self, kind: str | None = None) -> list[DatasetExample]:
        """List persisted dataset examples, optionally filtered by kind."""


class DatasetExampleReader(Protocol):
    """Loads dataset examples from a concrete source."""

    def load_dataset_examples(self, path: Path) -> list[DatasetExample]:
        """Load dataset examples from a path."""


class DatasetExampleWriter(Protocol):
    """Persists dataset examples to a concrete destination."""

    async def save_dataset_examples(self, path: Path, examples: list[DatasetExample]) -> None:
        """Persist dataset examples to a path."""


class DatasetMerger(Protocol):
    """Combines dataset examples with a deduplication policy."""

    def merge_datasets(
        self,
        datasets: list[list[DatasetExample]],
        *,
        deduplicate_by: str,
    ) -> tuple[list[DatasetExample], int]:
        """Merge datasets and return merged examples plus skipped duplicate count."""


class DatasetRelabeler(Protocol):
    """Applies label updates to matching dataset examples."""

    def relabel_examples(
        self,
        examples: list[DatasetExample],
        *,
        trace_id: str | None = None,
        source: str | None = None,
        input_outcome: OutcomeLabel | None = None,
        input_quality: QualityLabel | None = None,
        outcome: OutcomeLabel | None = None,
        quality: QualityLabel | None = None,
        failure_modes: tuple[FailureMode, ...] | None = None,
        reviewer_notes: str | None = None,
    ) -> tuple[list[DatasetExample], int]:
        """Return relabeled examples and the number changed."""


class DatasetExporter(Protocol):
    """Exports dataset examples to a concrete format."""

    def export_dataset_examples(self, path: Path, examples: list[DatasetExample]) -> None:
        """Export dataset examples to a path."""


class SyntheticDataGenerator(Protocol):
    """Source of generated dataset examples."""

    async def generate(
        self,
        count: int,
        *,
        seed: int | None = None,
        balance_categories: bool = True,
        vary_scenarios: bool = True,
        include_categories: tuple[str, ...] = (),
        exclude_categories: tuple[str, ...] = (),
    ) -> list[DatasetExample]:
        """Generate synthetic examples from local templates and schemas."""


class DatasetValidator(Protocol):
    """Checks examples before they are exported or used for training."""

    async def validate(self, examples: list[DatasetExample]) -> EvaluationResult:
        """Validate dataset examples before export or training."""


class DatasetFileHasher(Protocol):
    """Hashes concrete dataset files for training metadata."""

    def dataset_file_sha256(self, path: Path) -> str:
        """Return a dataset file SHA-256 digest."""


class DatasetToolProfileSummarizer(Protocol):
    """Summarizes tool-profile metadata across dataset examples."""

    def summarize_dataset_tool_profiles(
        self,
        examples: list[DatasetExample],
    ) -> dict[str, Any]:
        """Return JSON-ready tool-profile summary metadata."""


class DatasetBuilder(Protocol):
    """Creates named splits such as train, validation, and test."""

    async def build_split(self, examples: list[DatasetExample], name: str) -> DatasetSplit:
        """Build a named training, validation, test, or evaluation split."""


class TrainingRunner(Protocol):
    """Starts a local training run or a dry run."""

    async def run(self, config: TrainingConfig) -> TrainingRun:
        """Run or dry-run a local training job."""


class ArtifactStore(Protocol):
    """Persistence boundary for produced model artifacts."""

    async def save(self, artifact: ModelArtifact) -> None:
        """Persist metadata for a training or packaging artifact."""

    async def get(self, artifact_id: str) -> ModelArtifact | None:
        """Load artifact metadata by id."""


class EvaluationSuite(Protocol):
    """Runs a configured evaluation against a model artifact."""

    async def evaluate_artifact(self, artifact: ModelArtifact) -> EvaluationResult:
        """Evaluate a model artifact against configured tasks."""


class ModelPromotionPolicy(Protocol):
    """Decides whether an evaluated artifact can be promoted."""

    async def can_promote(self, artifact: ModelArtifact, evaluation: EvaluationResult) -> bool:
        """Decide whether a model artifact is eligible for promotion."""


class TrainingRunArtifactReader(Protocol):
    """Loads model artifacts produced by training runs."""

    def load_artifact_from_training_run(self, run_dir: Path) -> ModelArtifact:
        """Load artifact metadata for one training run."""


class EvaluationResultReader(Protocol):
    """Loads persisted evaluation reports."""

    def load_evaluation_result(self, path: Path) -> EvaluationResult:
        """Load one evaluation report."""


class PromotionGateResultWriter(Protocol):
    """Persists model promotion gate decisions."""

    def write_promotion_gate_result(
        self,
        run_dir: Path,
        *,
        promoted: bool,
        minimum_score: float,
        evaluation_reports: list[tuple[Path, EvaluationResult, bool]],
    ) -> Path:
        """Write one promotion gate decision and return its path."""


class PromotionRegistryWriter(Protocol):
    """Persists records for approved model artifacts."""

    def record_promoted_artifact(
        self,
        registry_path: Path,
        *,
        artifact: ModelArtifact,
        run_dir: Path,
        promotion_report_path: Path,
        reviewer_notes: str | None = None,
        approved_by: str | None = None,
    ) -> PromotedArtifactRecord:
        """Record a promoted artifact and return the stored record."""


class PromotionRegistryReader(Protocol):
    """Loads records for approved model artifacts."""

    def load_promotion_registry(self, registry_path: Path) -> list[PromotedArtifactRecord]:
        """Load promoted artifact records."""


class RepositoryModelConfigurationWriter(Protocol):
    """Persists repository-local model defaults."""

    def update_model_configuration(
        self,
        repository_root: Path,
        *,
        base_model: str,
        adapter_path: str,
        selected_promotion: dict[str, object],
    ) -> ModelConfigurationUpdate:
        """Update repository-local model defaults."""


class PromotedAdapterPackager(Protocol):
    """Packages promoted adapters for a local model runtime."""

    def package_promoted_adapter_for_ollama(
        self,
        *,
        record: PromotedArtifactRecord,
        model_name: str,
        output_dir: Path,
        ollama_base_model: str | None = None,
        create: bool = False,
    ) -> OllamaPackageRecord:
        """Package one promoted adapter for Ollama."""
