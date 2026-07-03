"""Concrete model-provider adapters — backward-compat re-exports.

Canonical location: execution.infrastructure.models.*
"""

from micro_model_agent.execution.infrastructure.models.fake import (
    ScriptedModelProvider,
    StaticModelProvider,
)
from micro_model_agent.execution.infrastructure.models.ollama import OllamaModelProvider
from micro_model_agent.execution.infrastructure.models.transformers import (
    TransformersPeftModelProvider,
)

__all__ = [
    "OllamaModelProvider",
    "ScriptedModelProvider",
    "StaticModelProvider",
    "TransformersPeftModelProvider",
]
