"""Pure scoring rubrics for trace-derived evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, cast

from micro_model_agent.dataset.domain.value_objects import DatasetExample

DEFAULT_TOOL_PROFILE_NAME = "coding-agent-v1"


__all__ = [
    "TraceExampleScore",
    "TraceRubric",
    "expected_trace_final_response",
    "expected_trace_patch",
    "expected_trace_tool_names",
    "json_object_from_response",
    "normalize_trace_text",
    "score_trace_example",
    "strip_markdown_fence",
    "trace_category",
    "trace_final_response_match",
    "trace_id",
    "trace_patch_match",
    "trace_similarity",
    "trace_tool_history_match",
    "trace_tool_names_from_response",
]


@dataclass(frozen=True, slots=True)
class TraceExampleScore:
    """Scorecard for one held-out trace-derived example."""

    example_id: str
    trace_id: str | None
    category: str | None
    raw_response: str
    parsed_response: dict[str, Any] | None
    tool_profile: dict[str, Any]
    score: float
    parse_success: bool
    final_response_match: bool
    patch_match: bool
    tool_history_match: bool
    errors: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-ready detail record for reports."""

        return {
            "example_id": self.example_id,
            "trace_id": self.trace_id,
            "category": self.category,
            "score": self.score,
            "parse_success": self.parse_success,
            "final_response_match": self.final_response_match,
            "patch_match": self.patch_match,
            "tool_history_match": self.tool_history_match,
            "errors": list(self.errors),
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "tool_profile": self.tool_profile,
        }


class TraceRubric:
    """Score trace-derived responses without owning concrete tool defaults."""

    def __init__(self, *, default_available_tools: Sequence[str] = ()) -> None:
        self.default_available_tools = tuple(default_available_tools)

    def score_example(
        self,
        example: DatasetExample,
        raw_response: str,
    ) -> TraceExampleScore:
        """Score one model response against a trace-derived dataset example."""

        return score_trace_example(
            example,
            raw_response,
            default_available_tools=self.default_available_tools,
        )


def score_trace_example(
    example: DatasetExample,
    raw_response: str,
    *,
    default_available_tools: Sequence[str] = (),
) -> TraceExampleScore:
    """Score one model response against a trace-derived dataset example."""

    errors: list[str] = []
    tool_profile = _tool_profile_for_example(
        example,
        default_available_tools=default_available_tools,
    )
    try:
        response = json_object_from_response(raw_response)
    except ValueError as exc:
        return TraceExampleScore(
            example_id=str(example.id),
            trace_id=trace_id(example),
            category=trace_category(example),
            raw_response=raw_response,
            parsed_response=None,
            tool_profile=tool_profile,
            score=0.0,
            parse_success=False,
            final_response_match=False,
            patch_match=False,
            tool_history_match=False,
            errors=(str(exc),),
        )

    checks: list[bool] = []
    final_response_match = trace_final_response_match(example, response)
    if expected_trace_final_response(example) is not None:
        checks.append(final_response_match)
        if not final_response_match:
            errors.append("final_response did not match expected trace response")

    patch_match = trace_patch_match(example, response)
    if expected_trace_patch(example) is not None:
        checks.append(patch_match)
        if not patch_match:
            errors.append("patch did not match expected trace patch")

    tool_history_match = trace_tool_history_match(example, response)
    if expected_trace_tool_names(example):
        checks.append(tool_history_match)
        if not tool_history_match:
            errors.append("tool_history did not include expected tool calls")

    if not checks:
        errors.append("trace example has no scorable target fields")

    score = sum(1.0 for check in checks if check) / len(checks) if checks else 0.0
    return TraceExampleScore(
        example_id=str(example.id),
        trace_id=trace_id(example),
        category=trace_category(example),
        raw_response=raw_response,
        parsed_response=response,
        tool_profile=tool_profile,
        score=score,
        parse_success=True,
        final_response_match=final_response_match,
        patch_match=patch_match,
        tool_history_match=tool_history_match,
        errors=tuple(errors),
    )


def expected_trace_final_response(example: DatasetExample) -> str | None:
    value = example.target.get("final_response") or example.target.get("summary")
    return value if isinstance(value, str) and value.strip() else None


def expected_trace_patch(example: DatasetExample) -> str | None:
    value = example.target.get("patch")
    return value if isinstance(value, str) and value.strip() else None


def trace_final_response_match(
    example: DatasetExample,
    response: dict[str, Any],
) -> bool:
    expected = expected_trace_final_response(example)
    if expected is None:
        return True
    actual = response.get("final_response", response.get("response"))
    if not isinstance(actual, str):
        return False
    return trace_similarity(expected, actual) >= 0.8


def trace_patch_match(example: DatasetExample, response: dict[str, Any]) -> bool:
    expected = expected_trace_patch(example)
    if expected is None:
        return True
    actual = response.get("patch")
    if not isinstance(actual, str):
        return False
    return normalize_trace_text(expected) == normalize_trace_text(actual)


def trace_tool_history_match(
    example: DatasetExample,
    response: dict[str, Any],
) -> bool:
    expected_tools = expected_trace_tool_names(example)
    if not expected_tools:
        return True
    actual_tools = trace_tool_names_from_response(response)
    return actual_tools[: len(expected_tools)] == expected_tools


def expected_trace_tool_names(example: DatasetExample) -> list[str]:
    names: list[str] = []
    history = example.input.get("tool_history")
    if not isinstance(history, list):
        return names
    for item in history:
        if not isinstance(item, dict):
            continue
        tool_call = item.get("tool_call")
        if isinstance(tool_call, dict) and isinstance(tool_call.get("tool_name"), str):
            names.append(tool_call["tool_name"])
    return names


def trace_tool_names_from_response(response: dict[str, Any]) -> list[str]:
    history = response.get("tool_history")
    if isinstance(history, list):
        names: list[str] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name")
            tool_call = item.get("tool_call")
            if isinstance(tool_name, str):
                names.append(tool_name)
            elif isinstance(tool_call, dict) and isinstance(tool_call.get("tool_name"), str):
                names.append(tool_call["tool_name"])
        return names

    tool_name = response.get("tool_name")
    return [tool_name] if isinstance(tool_name, str) else []


def trace_similarity(expected: str, actual: str) -> float:
    return SequenceMatcher(
        None,
        normalize_trace_text(expected).casefold(),
        normalize_trace_text(actual).casefold(),
    ).ratio()


def normalize_trace_text(value: str) -> str:
    return " ".join(value.split())


def trace_id(example: DatasetExample) -> str | None:
    value = example.metadata.get("trace_id")
    return value if isinstance(value, str) else None


def trace_category(example: DatasetExample) -> str | None:
    category = example.metadata.get("category")
    return category if isinstance(category, str) else None


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


def _tool_profile_for_example(
    example: DatasetExample,
    *,
    default_available_tools: Sequence[str] = (),
) -> dict[str, Any]:
    available_tools = _available_tools(example, default_available_tools)
    profile_name = _metadata_profile_name(example)
    if profile_name is None:
        profile_name = DEFAULT_TOOL_PROFILE_NAME if default_available_tools else "custom"

    return {
        "name": profile_name,
        "tool_schema_version": example.tool_schema_version,
        "available_tools": list(available_tools),
        "tools_used": list(_tools_used(example)),
    }


def _available_tools(
    example: DatasetExample,
    default_available_tools: Sequence[str],
) -> tuple[str, ...]:
    input_tools = example.input.get("available_tools")
    if isinstance(input_tools, list) and all(isinstance(tool, str) for tool in input_tools):
        return _dedupe(input_tools)

    metadata_profile = example.metadata.get("tool_profile")
    if isinstance(metadata_profile, dict):
        metadata_tools = metadata_profile.get("available_tools")
        if isinstance(metadata_tools, list) and all(
            isinstance(tool, str) for tool in metadata_tools
        ):
            return _dedupe(metadata_tools)

    return _dedupe(default_available_tools)


def _tools_used(example: DatasetExample) -> tuple[str, ...]:
    tools: list[str] = []
    target_tool = example.target.get("tool_name")
    if isinstance(target_tool, str):
        tools.append(target_tool)

    history = example.input.get("tool_history")
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            tool_call = item.get("tool_call")
            if isinstance(tool_call, dict) and isinstance(tool_call.get("tool_name"), str):
                tools.append(tool_call["tool_name"])

    return _dedupe(tools)


def _metadata_profile_name(example: DatasetExample) -> str | None:
    metadata_profile = example.metadata.get("tool_profile")
    if not isinstance(metadata_profile, dict):
        return None
    name = metadata_profile.get("name")
    return name if isinstance(name, str) and name else None


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
