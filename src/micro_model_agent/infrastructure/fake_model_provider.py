"""Compatibility imports for model-provider adapters."""

from micro_model_agent.infrastructure.models.fake import (
    ScriptedModelProvider,
    StaticModelProvider,
)

__all__ = ["ScriptedModelProvider", "StaticModelProvider"]
