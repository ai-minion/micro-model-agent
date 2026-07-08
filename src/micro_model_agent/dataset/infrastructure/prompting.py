"""Prompt payload helpers for synthetic dataset training and evaluation."""

from __future__ import annotations

from typing import Any

from micro_model_agent.dataset.domain.value_objects import DatasetExample, OutcomeLabel
from micro_model_agent.repository_ops.infrastructure.catalog import TOOL_ARGUMENT_CONTRACTS


def synthetic_prompt_payload(
    example: DatasetExample,
    *,
    include_tool_schemas: bool = False,
) -> dict[str, object]:
    """Build the prompt payload for non-trace synthetic examples.

    Repair examples often include a previous bad model response. The model should
    learn from that response, not copy assistant-shaped keys such as ``refusal``
    or obsolete argument names into its next answer, so the bad response is
    summarized into prompt facts.
    """

    available_tools = _available_tools(example)
    contract = _response_contract(example)
    payload: dict[str, object] = {
        "goal": example.input.get("goal", ""),
        "available_tools": available_tools,
        "context": example.input.get("context", ""),
        "input": _sanitized_input(example),
        "response_contract": contract,
    }
    if include_tool_schemas:
        from micro_model_agent.repository_ops.infrastructure.catalog import (
            builtin_native_tool_schemas,
        )

        payload["tool_schemas"] = builtin_native_tool_schemas(available_tools)
    return payload


def _available_tools(example: DatasetExample) -> list[str]:
    available_tools = example.input.get("available_tools")
    if isinstance(available_tools, list) and all(
        isinstance(tool_name, str) for tool_name in available_tools
    ):
        return available_tools
    return list(TOOL_ARGUMENT_CONTRACTS)


def _response_contract(example: DatasetExample) -> dict[str, object]:
    if example.label.outcome is OutcomeLabel.REJECTED or "refusal" in example.target:
        return {
            "type": "refusal",
            "required_keys": ["refusal"],
            "forbidden_keys": ["tool_name", "arguments", "final_response"],
        }
    if "final_response" in example.target:
        return {
            "type": "final_response",
            "required_keys": ["final_response", "ok"],
            "forbidden_keys": ["tool_name", "arguments", "refusal"],
        }
    return {
        "type": "tool_call",
        "required_keys": ["tool_name", "arguments"],
        "forbidden_keys": ["refusal", "final_response", "ok"],
        "allowed_top_level_keys": ["tool_name", "arguments", "reason"],
        "forbidden_top_level_keys": [
            "argument_keys",
            "argument_values",
            "argument_changes",
            "argument_reconciliation",
            "selected_tool",
            "changed_fields",
        ],
    }


def _sanitized_input(example: DatasetExample) -> dict[str, object]:
    payload = dict(example.input)
    # Scenario variation is useful for generation bookkeeping, but it is not a
    # runtime user field and can be copied into tool arguments if exposed.
    payload.pop("variant_focus", None)
    bad_output = payload.pop("bad_output", None)
    if isinstance(bad_output, dict):
        payload["previous_invalid_response"] = _summarize_bad_output(bad_output)
    return payload


def _summarize_bad_output(bad_output: dict[str, Any]) -> dict[str, object]:
    summary: dict[str, object] = {}

    tool_name = bad_output.get("tool_name")
    if isinstance(tool_name, str):
        summary["invalid_selected_tool"] = tool_name

    arguments = bad_output.get("arguments")
    if isinstance(arguments, dict):
        summary["invalid_argument_field_names"] = sorted(str(key) for key in arguments)
        summary["invalid_argument_value_notes"] = _sanitize_argument_values(arguments)

    if "refusal" in bad_output:
        summary["invalid_response_kind"] = "refusal_text"
    if "final_response" in bad_output:
        summary["invalid_response_kind"] = "final_response_text"

    reason = bad_output.get("reason")
    if isinstance(reason, str) and reason.strip():
        summary["previous_reason"] = reason

    return summary


def _sanitize_argument_values(arguments: dict[str, Any]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in arguments.items():
        if key == "command":
            sanitized["invalid_shell_command"] = value
        elif key == "command_key":
            sanitized["obsolete_test_command_field"] = value
        elif key == "pattern":
            sanitized["invalid_search_pattern_field"] = value
        else:
            sanitized[str(key)] = value
    return sanitized
