"""Fake model providers for tests and local workflow dry runs."""

from __future__ import annotations

from collections.abc import Sequence


class StaticModelProvider:
    """Model provider that always returns the same completion."""

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        # Store prompts so tests can assert what the workflow asked the model.
        self.prompts.append(prompt)
        return self.completion


class ScriptedModelProvider:
    """Model provider that returns completions in a fixed order."""

    def __init__(self, completions: Sequence[str]) -> None:
        self.completions = tuple(completions)
        self.prompts: list[str] = []
        self._next_index = 0

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._next_index >= len(self.completions):
            raise RuntimeError("scripted model completions exhausted")
        # Return the next scripted response, then advance the cursor.
        completion = self.completions[self._next_index]
        self._next_index += 1
        return completion
