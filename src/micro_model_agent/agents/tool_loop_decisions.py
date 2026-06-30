"""Decision parsing for model-authored tool-loop turns."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

# Python 3.12 type aliases keep the model response categories readable below.
type DecisionKind = Literal["tool_call", "final_response", "parse_error"]


@dataclass(frozen=True, slots=True)
class ModelDecision:
    """Internal parsed version of the model's JSON response."""

    kind: DecisionKind
    raw_response: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    response: str = ""
    ok: bool = False
    error: str | None = None
    reason: str | None = None


def parse_model_response(raw_response: str) -> ModelDecision:
    """Parse the model's JSON into one internal decision object."""

    try:
        payload = json_object_from_response(raw_response)
    except ValueError as exc:
        return ModelDecision(
            kind="parse_error",
            raw_response=raw_response,
            error=str(exc),
        )

    final_response = payload.get("final_response", payload.get("response"))
    if final_response is not None:
        if not isinstance(final_response, str):
            return ModelDecision(
                kind="parse_error",
                raw_response=raw_response,
                error="final_response must be a string",
            )
        ok_value = payload.get("ok", True)
        return ModelDecision(
            kind="final_response",
            raw_response=raw_response,
            response=final_response,
            ok=bool(ok_value) if isinstance(ok_value, bool) else True,
        )

    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str):
        return ModelDecision(
            kind="parse_error",
            raw_response=raw_response,
            error="tool_name must be a string",
        )
    if not isinstance(arguments, dict):
        return ModelDecision(
            kind="parse_error",
            raw_response=raw_response,
            error="arguments must be an object",
        )

    reason = payload.get("reason")
    return ModelDecision(
        kind="tool_call",
        raw_response=raw_response,
        tool_name=tool_name,
        arguments=cast(dict[str, Any], arguments),
        ok=True,
        reason=reason if isinstance(reason, str) else None,
    )


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
    """Remove ``` fences because some models wrap JSON in markdown."""

    if not response.startswith("```"):
        return response

    lines = response.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
