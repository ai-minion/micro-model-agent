"""Ollama-backed model provider.

Ollama runs the model server locally. This class adapts its Python client to the
small ModelProvider protocol used by the rest of the application.
"""

from __future__ import annotations

import json
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

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        # Use the chat endpoint so Ollama applies the model's own chat template.
        result = await self.client.chat(
            model=self.model_name,
            messages=messages,
            tools=tools,
            format="json" if self.json_mode and not tools else None,
            options=self.options or None,
        )
        response = self._response_text(cast(object, result))
        if response is None:
            raise RuntimeError("Ollama response did not include completion text")
        return response

    def _response_text(self, result: object) -> str | None:
        """Read completion text from either dict-like or object-like chat responses.

        When the model returns a native tool call (``tool_calls`` populated,
        ``content`` empty) the first tool call is serialised to the
        ``{"name": ..., "arguments": ...}`` format that the decision parser
        already handles, so the rest of the pipeline needs no changes.
        """

        if isinstance(result, Mapping):
            message = result.get("message") or {}
            content = message.get("content") if isinstance(message, Mapping) else None
            if content:
                return str(content)
            tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        else:
            message = getattr(result, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content:
                return str(content)
            tool_calls = getattr(message, "tool_calls", None) if message is not None else None

        if tool_calls:
            first = tool_calls[0] if isinstance(tool_calls, (list, tuple)) else tool_calls
            if isinstance(first, Mapping):
                func = first.get("function") or {}
                name = func.get("name") if isinstance(func, Mapping) else None
                arguments = func.get("arguments") if isinstance(func, Mapping) else None
            else:
                func = getattr(first, "function", None)
                name = getattr(func, "name", None) if func is not None else None
                arguments = getattr(func, "arguments", None) if func is not None else None
            if isinstance(name, str):
                args = arguments if isinstance(arguments, dict) else {}
                return json.dumps({"name": name, "arguments": args})
        return None
