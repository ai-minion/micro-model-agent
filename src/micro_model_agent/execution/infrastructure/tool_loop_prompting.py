"""Prompt rendering helpers for ToolLoopAgent."""

from __future__ import annotations

import json
from typing import Any, cast

from micro_model_agent.execution.application.tool_loop import PromptContext, ToolLoopAgentTask
from micro_model_agent.execution.domain.value_objects import WorkflowStep
from micro_model_agent.execution.infrastructure.tool_loop_history import tool_history


def build_prompt(
    task: ToolLoopAgentTask,
    transcript: list[dict[str, Any]],
    *,
    tool_calls_made: int,
    turn_number: int,
    final_response_only: bool,
    missing_required_tools: tuple[str, ...],
    orchestration_hints: list[str],
    steps: list[WorkflowStep],
) -> list[dict[str, str]]:
    """Build the chat messages for one model turn."""

    payload: dict[str, Any] = {
        "goal": task.goal,
        "available_tools": [] if final_response_only else list(task.available_tools),
        "context": prompt_context(task.context),
        "loop_budget": loop_budget_payload(
            task,
            tool_calls_made=tool_calls_made,
            turn_number=turn_number,
            final_response_only=final_response_only,
        ),
    }
    if task.required_tools:
        payload["required_tools"] = list(task.required_tools)
        payload["missing_required_tools"] = list(missing_required_tools)
    if orchestration_hints:
        payload["orchestration_hints"] = orchestration_hints
    tool_history_payload = tool_history(
        steps,
        max_prompt_chars=task.max_tool_result_prompt_chars,
    )
    if tool_history_payload:
        payload["tool_history"] = tool_history_payload
    policy_results = [entry for entry in transcript if entry.get("role") == "tool"]
    if policy_results:
        payload["tool_results"] = policy_results
    if final_response_only:
        system_prompt = (
            "You are MicroModelAgent's workflow executor. "
            "No more tool calls are allowed. "
            "Use the provided tool_results to answer the user. "
            "Respond with a concise final answer. "
            "Final responses must be concise and must not repeat full tool output."
        )
    else:
        system_prompt = (
            "You are MicroModelAgent's workflow executor. "
            "Choose safe typed tool calls and follow retrieved context. "
            "Use the provided tool-call format when a tool is needed. "
            "Call at least one available tool before answering. "
            "This loop has a small fixed turn budget; each turn must either make "
            "new progress or finish. "
            "Track loop_budget carefully; if this is the last turn, prefer "
            "answering unless a required tool still has to be called. "
            "Do not reread the same file or repeat the same write unless the previous "
            "result was missing or failed. "
            "After a useful write succeeds, either run one distinct verification tool "
            "or answer the user. "
            "Final responses must be concise and must not repeat full tool output. "
            "For greenfield creation or scaffolding tasks, prefer repo.write_files over "
            "repo.write_patch. If repo.search or repo.semantic_search repeatedly returns "
            "no matches for a creation task, stop searching and create the requested files. "
            "When the task is complete, respond with a concise final answer."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def prompt_tools(task: ToolLoopAgentTask, *, final_response_only: bool) -> list[dict[str, Any]]:
    """Return native chat-template tool definitions for the available tools."""

    if final_response_only:
        return []
    return [
        _native_tool_schema(tool_name, schema)
        for tool_name, schema in available_tool_schemas(task).items()
    ]


def available_tool_schemas(task: ToolLoopAgentTask) -> dict[str, Any]:
    """Return schemas only for tools that are both available and documented."""

    return {
        tool_name: task.tool_schemas[tool_name]
        for tool_name in task.available_tools
        if tool_name in task.tool_schemas
    }


def _native_tool_schema(tool_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    description = str(schema.get("description", ""))
    parameters = dict(schema.get("arguments_schema") or {"type": "object"})
    allowed_command_names = schema.get("allowed_command_names")
    if isinstance(allowed_command_names, list) and allowed_command_names:
        parameters = _schema_with_command_enum(parameters, allowed_command_names)
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": parameters,
        },
    }


def _schema_with_command_enum(schema: dict[str, Any], allowed: list[Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(schema))
    properties = updated.setdefault("properties", {})
    command_name = properties.setdefault("command_name", {})
    command_name["enum"] = [str(name) for name in allowed]
    return cast(dict[str, Any], updated)


def loop_budget_payload(
    task: ToolLoopAgentTask,
    *,
    tool_calls_made: int,
    turn_number: int,
    final_response_only: bool,
) -> dict[str, Any]:
    """Expose loop limits so the model can plan instead of guessing."""

    turns_remaining = max(task.max_turns - turn_number + 1, 0)
    tool_calls_remaining: int | None = None
    if task.max_tool_calls is not None:
        tool_calls_remaining = max(task.max_tool_calls - tool_calls_made, 0)
    return {
        "turn_number": turn_number,
        "max_turns": task.max_turns,
        "turns_remaining_including_current": turns_remaining,
        "tool_calls_made": tool_calls_made,
        "max_tool_calls": task.max_tool_calls,
        "tool_calls_remaining": tool_calls_remaining,
        "final_response_only": final_response_only,
    }


def prompt_context(context: PromptContext) -> PromptContext:
    """Normalize an empty context to a string so JSON output stays simple."""

    if context:
        return context
    return ""
