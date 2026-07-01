"""Concrete model-provider adapters."""

from micro_model_agent.infrastructure.models.fake import (
    ScriptedModelProvider,
    StaticModelProvider,
)
from micro_model_agent.infrastructure.models.ollama import OllamaModelProvider
from micro_model_agent.infrastructure.models.transformers import (
    TransformersPeftModelProvider,
)

__all__ = [
    "OllamaModelProvider",
    "ScriptedModelProvider",
    "StaticModelProvider",
    "TransformersPeftModelProvider",
]
