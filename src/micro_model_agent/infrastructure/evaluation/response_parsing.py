"""Shared helpers for parsing model evaluation responses."""

from __future__ import annotations

import json
from typing import Any, cast


def json_object_from_response(raw_response: str) -> dict[str, Any]:
    """Extract the first JSON object from plain text or fenced markdown."""

    payload = strip_markdown_fence(raw_response.strip())
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(payload)
    except json.JSONDecodeError:
        first_brace = payload.find("{")
        if first_brace < 0:
            raise ValueError("model response must be a JSON object") from None
        try:
            parsed, _ = decoder.raw_decode(payload[first_brace:])
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from None

    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return cast(dict[str, Any], parsed)


def strip_markdown_fence(response: str) -> str:
    """Remove markdown code fences around JSON when present."""

    if not response.startswith("```"):
        return response

    lines = response.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
