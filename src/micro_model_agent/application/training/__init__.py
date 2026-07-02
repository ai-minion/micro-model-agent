"""Compatibility facade for training application workflows."""

from micro_model_agent.application.training.workflows import (
    RunSyntheticTrainingRequest,
    RunSyntheticTrainingResult,
    RunSyntheticTrainingWorkflow,
)

__all__ = [
    "RunSyntheticTrainingRequest",
    "RunSyntheticTrainingResult",
    "RunSyntheticTrainingWorkflow",
]
