"""Ollama-backed model provider.

Ollama runs the model server locally. This class adapts its Python client to the
small ModelProvider protocol used by the rest of the application.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ollama import AsyncClient


class OllamaModelProvider:
    """Model provider that sends prompts to an Ollama model."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        json_mode: bool = True,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.json_mode = json_mode
        self.options = dict(options or {})
        self.client = AsyncClient(host=base_url)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        # Use the chat endpoint so Ollama applies the model's own chat template.
        result = await self.client.chat(
            model=self.model_name,
            messages=messages,
            format="json" if self.json_mode else None,
            options=self.options or None,
        )
        response = self._response_text(cast(object, result))
        if response is None:
            raise RuntimeError("Ollama response did not include completion text")
        return response

    def _response_text(self, result: object) -> str | None:
        """Read completion text from either dict-like or object-like chat responses."""

        if isinstance(result, Mapping):
            message = result.get("message") or {}
            response = message.get("content") if isinstance(message, Mapping) else None
        else:
            message = getattr(result, "message", None)
            response = getattr(message, "content", None) if message is not None else None
        return response if isinstance(response, str) else None
