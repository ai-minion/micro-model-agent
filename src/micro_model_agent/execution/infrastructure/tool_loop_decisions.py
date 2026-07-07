"""Decision parsing for model-authored tool-loop turns."""

from __future__ import annotations

import json
import re
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
    """Parse native model output into one internal decision object."""

    try:
        native_tool_call = native_tool_call_from_response(raw_response)
    except ValueError as exc:
        return ModelDecision(
            kind="parse_error",
            raw_response=raw_response,
            error=str(exc),
        )
    if native_tool_call is not None:
        tool_name = native_tool_call.get("name")
        arguments = native_tool_call.get("arguments")
        if not isinstance(tool_name, str):
            return ModelDecision(
                kind="parse_error",
                raw_response=raw_response,
                error="tool call name must be a string",
            )
        if not isinstance(arguments, dict):
            return ModelDecision(
                kind="parse_error",
                raw_response=raw_response,
                error="tool call arguments must be an object",
            )
        # Normalize: model may emit {"name":"final_response","arguments":{...}}
        # in native format the same way it does in legacy JSON format.
        if tool_name == "final_response":
            response_text = (
                arguments.get("body") or arguments.get("response") or
                arguments.get("text") or arguments.get("message") or ""
            )
            if not isinstance(response_text, str):
                response_text = str(response_text)
            ok_value = arguments.get("ok", True)
            return ModelDecision(
                kind="final_response",
                raw_response=raw_response,
                response=response_text,
                ok=bool(ok_value) if isinstance(ok_value, bool) else True,
            )

        return ModelDecision(
            kind="tool_call",
            raw_response=raw_response,
            tool_name=tool_name,
            arguments=cast(dict[str, Any], arguments),
            ok=True,
            reason=_content_before_tool_call(raw_response),
        )

    try:
        payload = json_object_from_response(raw_response)
    except ValueError as exc:
        content = strip_native_response_markup(raw_response).strip()
        if content:
            return ModelDecision(
                kind="final_response",
                raw_response=raw_response,
                response=content,
                ok=True,
            )
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

    tool_name = payload.get("tool_name", payload.get("name"))
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

    # Normalize: model sometimes emits {"tool_name":"final_response","arguments":{"body":"..."}}
    # instead of the canonical {"final_response":"...","ok":true}.
    if tool_name == "final_response":
        response_text = (
            arguments.get("body") or arguments.get("response") or
            arguments.get("text") or arguments.get("message") or ""
        )
        if not isinstance(response_text, str):
            response_text = str(response_text)
        ok_value = payload.get("ok", arguments.get("ok", True))
        return ModelDecision(
            kind="final_response",
            raw_response=raw_response,
            response=response_text,
            ok=bool(ok_value) if isinstance(ok_value, bool) else True,
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


def native_tool_call_from_response(raw_response: str) -> dict[str, Any] | None:
    """Extract a Qwen-style native tool call from assistant text."""

    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw_response, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from None
        if not isinstance(payload, dict):
            raise ValueError("tool call payload must be a JSON object")
        return cast(dict[str, Any], payload)
    try:
        payload = json_object_from_response(raw_response)
    except ValueError:
        return None
    if "name" in payload and "arguments" in payload:
        return payload
    return None


def strip_native_response_markup(raw_response: str) -> str:
    """Remove native tool/reasoning tags when treating assistant text as content."""

    without_tool_call = re.sub(
        r"<tool_call>.*?</tool_call>",
        "",
        raw_response,
        flags=re.DOTALL,
    )
    # Some models wrap schema or call output in <tools>...</tools>.
    without_tools = re.sub(
        r"<tools>.*?</tools>",
        "",
        without_tool_call,
        flags=re.DOTALL,
    )
    without_think = re.sub(
        r"<think>.*?</think>",
        "",
        without_tools,
        flags=re.DOTALL,
    )
    # Strip bare end-of-turn stop markers, e.g. "Done.<stop>" or "<stop/>".
    without_stop = re.sub(r"<stop\s*/?>", "", without_think)
    return without_stop


def _content_before_tool_call(raw_response: str) -> str | None:
    content = raw_response.split("<tool_call>", 1)[0]
    content = strip_native_response_markup(content).strip()
    return content or None


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
