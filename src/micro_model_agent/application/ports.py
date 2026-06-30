"""Ports used by application services.

A port is an interface the application layer depends on. In Python, a
``Protocol`` says "any object with these methods is acceptable", which keeps the
workflow code independent from the concrete storage, model, and tool classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from micro_model_agent.domain.contracts import (
    EvaluationResult,
    RetrievalQuery,
    SemanticSearchResult,
    ToolCall,
    ToolResult,
    WorkflowTrace,
)
from micro_model_agent.domain.datasets import DatasetExample, DatasetSplit
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


class SyntheticDataGenerator(Protocol):
    """Source of generated dataset examples."""

    async def generate(self, count: int) -> list[DatasetExample]:
        """Generate synthetic examples from local templates and schemas."""


class DatasetValidator(Protocol):
    """Checks examples before they are exported or used for training."""

    async def validate(self, examples: list[DatasetExample]) -> EvaluationResult:
        """Validate dataset examples before export or training."""


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
