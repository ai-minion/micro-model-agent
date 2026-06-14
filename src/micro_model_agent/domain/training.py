"""Domain contracts for local training runs and artifacts.

These objects describe what was requested, what was produced, and whether a
local training job succeeded. They are deliberately framework-neutral so a fake
runner and a real Hugging Face runner can both report the same shape of data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class TrainingRunStatus(StrEnum):
    """Lifecycle state for a training run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingRunKind(StrEnum):
    """Training run type."""

    SYNTHETIC = "synthetic"
    TRACE_DERIVED = "trace_derived"
    MIXED = "mixed"


class ModelArtifactKind(StrEnum):
    """Kinds of model artifacts produced by local training."""

    ADAPTER = "adapter"
    MERGED_MODEL = "merged_model"
    OLLAMA_MODEL = "ollama_model"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for a local fine-tuning run."""

    base_model: str
    output_dir: str
    # Optional training knobs can be omitted so callers can rely on backend
    # defaults for simple dry runs.
    max_steps: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    seed: int = 42
    # dry_run lets the pipeline validate inputs and write metadata without
    # starting an expensive GPU training job.
    dry_run: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Metadata for a model artifact produced by training or packaging."""

    # This is metadata about the artifact, not the model bytes themselves.
    name: str
    kind: ModelArtifactKind
    path: str
    base_model: str
    id: UUID = field(default_factory=uuid4)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """A recorded local training run."""

    kind: TrainingRunKind
    config: TrainingConfig
    # The default status represents a run record before any work has started.
    status: TrainingRunStatus = TrainingRunStatus.PENDING
    id: UUID = field(default_factory=uuid4)
    dataset_version: str | None = None
    artifacts: tuple[ModelArtifact, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
