"""Prompt rendering helpers for ToolLoopAgent."""

from __future__ import annotations

import json
from typing import Any

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
    """Build the prompt that asks the model for one JSON decision."""

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
    tool_schemas = available_tool_schemas(task)
    if tool_schemas and not final_response_only:
        payload["tool_schemas"] = tool_schemas

    if final_response_only:
        # When no more tools are allowed, the system prompt removes tool
        # choices and asks for a final_response JSON object.
        system_prompt = (
            "You are MicroModelAgent's workflow executor. "
            "No more tool calls are allowed. "
            "Use the provided tool_results to answer the user. "
            "Respond with exactly one JSON object and no markdown: "
            '{"final_response":"Concise answer to the user.","ok":true}. '
            "Final responses must be concise and must not repeat full tool output."
        )
    else:
        # In normal turns, the model sees available tools and the exact JSON
        # shapes it can return.
        system_prompt = (
            "You are MicroModelAgent's workflow executor. "
            "Choose safe typed tool calls and follow retrieved context. "
            "Respond with exactly one JSON object and no markdown. "
            "Call at least one available tool before final_response. "
            "This loop has a small fixed turn budget; each turn must either make "
            "new progress or finish. "
            "Track loop_budget carefully; if this is the last turn, prefer "
            "final_response unless a required tool still has to be called. "
            "Do not reread the same file or repeat the same write unless the previous "
            "result was missing or failed. "
            "After a useful write succeeds, either run one distinct verification tool "
            "or return final_response. "
            "Final responses must be concise and must not repeat full tool output. "
            "For greenfield creation or scaffolding tasks, prefer repo.write_files over "
            "repo.write_patch. If repo.search or repo.semantic_search repeatedly returns "
            "no matches for a creation task, stop searching and create the requested files. "
            "For a tool call, return "
            '{"tool_name":"repo.read","arguments":{"files":[{"path":"README.md"}]},'
            '"reason":"..."}. '
            "When the task is complete, return "
            '{"final_response":"Concise answer to the user.","ok":true}.'
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def available_tool_schemas(task: ToolLoopAgentTask) -> dict[str, Any]:
    """Return schemas only for tools that are both available and documented."""

    return {
        tool_name: task.tool_schemas[tool_name]
        for tool_name in task.available_tools
        if tool_name in task.tool_schemas
    }


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
