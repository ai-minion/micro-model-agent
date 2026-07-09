"""Prompt rendering helpers for ToolLoopAgent."""

from __future__ import annotations

import json
from typing import Any, cast

from micro_model_agent.execution.application.tool_loop import PromptContext, ToolLoopAgentTask
from micro_model_agent.execution.domain.value_objects import WorkflowStep
from micro_model_agent.execution.infrastructure.tool_loop_history import (
    concise_tool_error,
    summarize_tool_arguments,
    summarize_tool_output,
    truncate_tool_output,
)


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

    orchestration_state: dict[str, Any] = {
        "loop_budget": loop_budget_payload(
            task,
            tool_calls_made=tool_calls_made,
            turn_number=turn_number,
            final_response_only=final_response_only,
        ),
    }
    context = prompt_context(task.context)
    if missing_required_tools:
        orchestration_state["required_tools_pending"] = list(missing_required_tools)
    if orchestration_hints:
        orchestration_state["orchestration_hints"] = orchestration_hints
    if final_response_only:
        system_prompt = (
            "You are MicroModelAgent's workflow executor. "
            "No more tool calls are allowed. "
            "Use the timeline tool messages to answer the user. "
            "Respond with a concise final answer. "
            "Final responses must be concise and must not repeat full tool output. "
        )
    else:
        system_prompt = (
            "You are MicroModelAgent's workflow executor. "
            "Choose safe typed tool calls and follow retrieved context. "
            "Call at least one available tool before answering. "
            "If required_tools_pending is present, call one of those tools before "
            "answering. "
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
            "When the task is complete, respond with a concise final answer. "
        )
    system_prompt += "Orchestration state: " + json.dumps(
        orchestration_state,
        sort_keys=True,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message(task.goal, context)},
        *timeline_messages(steps, max_prompt_chars=task.max_tool_result_prompt_chars),
    ]


def user_message(goal: str, context: PromptContext) -> str:
    """Return the plain user-facing task message."""

    if not context:
        return goal
    if isinstance(context, str):
        return f"{goal}\n\nContext:\n{context}"
    return f"{goal}\n\nContext:\n{json.dumps(context, sort_keys=True)}"


def timeline_messages(
    steps: list[WorkflowStep],
    *,
    max_prompt_chars: int,
) -> list[dict[str, str]]:
    """Return prior assistant/tool turns as chronological chat messages."""

    messages: list[dict[str, str]] = []
    for step in steps:
        if step.tool_call is not None:
            messages.append(
                {"role": "assistant", "content": canonical_tool_call_content(step)}
            )
        else:
            raw_response = step.output.get("raw_response")
            if isinstance(raw_response, str) and raw_response:
                messages.append({"role": "assistant", "content": raw_response})

        if step.tool_call is not None and step.tool_result is not None:
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        tool_result_payload(step, max_prompt_chars=max_prompt_chars),
                        sort_keys=True,
                    ),
                }
            )
            continue

        policy_error = step.output.get("error")
        if isinstance(policy_error, str) and policy_error:
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "tool_name": "orchestration_policy",
                            "ok": False,
                            "error": policy_error,
                        },
                        sort_keys=True,
                    ),
                }
            )

    return messages


def canonical_tool_call_content(step: WorkflowStep) -> str:
    """Render executed tool calls in the native format expected by chat templates."""

    assert step.tool_call is not None
    payload = {
        "name": step.tool_call.tool_name,
        "arguments": step.tool_call.arguments,
    }
    return (
        "<tool_call>\n"
        + json.dumps(payload, sort_keys=True)
        + "\n</tool_call>"
    )


def tool_result_payload(
    step: WorkflowStep,
    *,
    max_prompt_chars: int,
) -> dict[str, Any]:
    """Return one compact tool result for a timeline tool message."""

    assert step.tool_call is not None
    assert step.tool_result is not None
    payload: dict[str, Any] = {
        "tool_name": step.tool_call.tool_name,
        "arguments": summarize_tool_arguments(
            step.tool_call.tool_name,
            step.tool_call.arguments,
        ),
        "ok": step.tool_result.ok,
    }
    if step.tool_result.ok:
        output = step.tool_result.output
        if len(json.dumps(output, sort_keys=True)) > max_prompt_chars:
            output = summarize_tool_output(output)
        payload["output"] = truncate_tool_output(output, max_prompt_chars)
    else:
        payload["error"] = concise_tool_error(step.tool_result)
    return payload


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
        "turn": turn_number,
        "turns_remaining": turns_remaining,
        "tool_calls_remaining": tool_calls_remaining,
        "final_response_only": final_response_only,
    }


def prompt_context(context: PromptContext) -> PromptContext:
    """Normalize an empty context to a string so JSON output stays simple."""

    if context:
        return context
    return ""
