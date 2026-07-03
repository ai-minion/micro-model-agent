"""Concrete model-provider adapters."""

from micro_model_agent.execution.infrastructure.models.fake import (
    ScriptedModelProvider,
    StaticModelProvider,
)

try:
    from micro_model_agent.execution.infrastructure.models.ollama import OllamaModelProvider
except ImportError:
    OllamaModelProvider = None  # type: ignore[assignment,misc]

try:
    from micro_model_agent.execution.infrastructure.models.transformers import (
        TransformersPeftModelProvider,
    )
except ImportError:
    TransformersPeftModelProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "OllamaModelProvider",
    "ScriptedModelProvider",
    "StaticModelProvider",
    "TransformersPeftModelProvider",
]
