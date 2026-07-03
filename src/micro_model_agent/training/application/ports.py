"""Training context application ports."""

from __future__ import annotations

from typing import Protocol

from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    TrainingConfig,
    TrainingRun,
)


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
