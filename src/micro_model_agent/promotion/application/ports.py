"""Promotion context application ports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from micro_model_agent.shared.domain.value_objects import EvaluationResult
from micro_model_agent.training.domain.value_objects import ModelArtifact

# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class ModelPromotionPolicy(Protocol):
    """Decides whether an evaluated artifact can be promoted.

    Deprecated: prefer ``PromotionGateService`` from promotion domain.
    """

    async def can_promote(self, artifact: ModelArtifact, evaluation: EvaluationResult) -> bool:
        """Decide whether a model artifact is eligible for promotion."""


class TrainingRunArtifactReader(Protocol):
    """Loads model artifacts produced by training runs."""

    def load_artifact_from_training_run(self, run_dir: Path) -> ModelArtifact:
        """Load artifact metadata for one training run."""


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
