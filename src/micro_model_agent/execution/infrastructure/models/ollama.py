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

    async def complete(self, prompt: str) -> str:
        # stream=False asks Ollama for one complete response instead of chunks.
        result = await self.client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=False,
            format="json" if self.json_mode else None,
            options=self.options or None,
        )
        response = self._response_text(cast(object, result))
        if response is None:
            raise RuntimeError("Ollama response did not include completion text")
        return response

    def _response_text(self, result: object) -> str | None:
        """Read completion text from either dict-like or object-like responses."""

        if isinstance(result, Mapping):
            response = result.get("response")
        else:
            response = getattr(result, "response", None)
        return response if isinstance(response, str) else None
