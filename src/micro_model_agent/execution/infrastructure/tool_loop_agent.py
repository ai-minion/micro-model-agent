"""Model-driven agent loop for typed tool calls.

Unlike ``CodingAgent``, this agent lets the model choose which tool to call on
each turn. The Python code still enforces budgets, allowed tools, required
tools, JSON parsing, and trace capture.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from micro_model_agent.execution.application.ports import ModelProvider, ToolExecutor, TraceStore
from micro_model_agent.execution.application.tool_loop import (
    DEFAULT_TOOL_NAMES,
    ToolLoopAgentResult,
    ToolLoopAgentTask,
)
from micro_model_agent.execution.domain.value_objects import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)
from micro_model_agent.execution.infrastructure.tool_loop_decisions import parse_model_response
from micro_model_agent.execution.infrastructure.tool_loop_policy import (
    dedup_block_count_for_call,
    discovery_sufficient_for_final_response,
    has_unresolved_failed_tool_step,
    is_duplicate_read_or_search,
    is_duplicate_successful_write,
    is_repeated_unavailable_tool,
    missing_required_tools,
    missing_verification_after_write,
    orchestration_hints,
    should_allow_extra_finalization_turn,
    tool_budget_exhausted,
)
from micro_model_agent.execution.infrastructure.tool_loop_prompting import (
    build_prompt,
    prompt_tools,
)

__all__ = [
    "DEFAULT_TOOL_NAMES",
    "ToolLoopAgent",
    "ToolLoopAgentResult",
    "ToolLoopAgentTask",
]


class ToolLoopAgent:
    """Run a request through model-authored tool calls until a final response."""

    def __init__(
        self,
        model_provider: ModelProvider,
        tool_executor: ToolExecutor,
        trace_store: TraceStore,
    ) -> None:
        self.model_provider = model_provider
        self.tool_executor = tool_executor
        self.trace_store = trace_store

    async def run(self, task: ToolLoopAgentTask) -> ToolLoopAgentResult:
        if task.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        trace = WorkflowTrace(
            goal=task.goal,
            status=WorkflowStatus.RUNNING,
            run_metadata=task.run_metadata,
        )
        await self.trace_store.save(trace)
        # transcript is the compact prompt history fed back to the model; steps
        # is the richer audit trail saved for users.
        steps: list[WorkflowStep] = []
        transcript: list[dict[str, Any]] = [{"role": "user", "content": task.goal}]
        if task.context:
            transcript.append({"role": "context", "content": task.context})

        tool_calls_made = 0
        turn_number = 1
        used_extra_finalization_turn = False
        try:
            while turn_number <= task.max_turns or (
                not used_extra_finalization_turn
                and should_allow_extra_finalization_turn(task, tool_calls_made, steps)
            ):
                force_final_response = turn_number > task.max_turns
                if force_final_response:
                    used_extra_finalization_turn = True
                budget_exhausted = tool_budget_exhausted(
                    task,
                    tool_calls_made,
                    steps,
                )
                discovery_sufficient = discovery_sufficient_for_final_response(
                    task,
                    steps,
                )
                final_response_only = (
                    force_final_response or budget_exhausted or discovery_sufficient
                )
                if task.on_turn:
                    await task.on_turn(
                        turn_number,
                        task.max_turns,
                        _thinking_message(tool_calls_made, final_response_only),
                    )
                messages = build_prompt(
                    task,
                    transcript,
                    tool_calls_made=tool_calls_made,
                    turn_number=turn_number,
                    final_response_only=final_response_only,
                    missing_required_tools=missing_required_tools(task, steps),
                    orchestration_hints=orchestration_hints(task, steps),
                    steps=steps,
                )
                tools = prompt_tools(task, final_response_only=final_response_only)
                try:
                    raw_response = await self._complete_model(task, messages, tools=tools)
                except TimeoutError:
                    steps.append(
                        WorkflowStep(
                            name=f"model_turn_{turn_number}",
                            status=WorkflowStatus.FAILED,
                            output=self._step_output(
                                {
                                    "error": "model_completion_timeout",
                                    "timeout_seconds": task.model_timeout_seconds,
                                },
                                messages=messages,
                                tools=tools,
                            ),
                        )
                    )
                    await self._save_progress(trace, steps)
                    return await self._finish(
                        trace=trace,
                        steps=steps,
                        run_metadata=task.run_metadata,
                        ok=False,
                        response="model completion timed out",
                        tool_calls_made=tool_calls_made,
                        error="model_completion_timeout",
                    )
                decision = parse_model_response(raw_response)

                if decision.kind == "final_response":
                    if task.require_tool_call and tool_calls_made == 0:
                        output: dict[str, Any] = self._step_output(
                            {
                                "raw_response": decision.raw_response,
                                "error": "final_response_before_tool_call",
                            },
                            messages=messages,
                            tools=tools,
                        )
                        steps.append(
                            WorkflowStep(
                                name=f"model_turn_{turn_number}",
                                status=WorkflowStatus.FAILED,
                                output=output,
                            )
                        )
                        await self._save_progress(trace, steps)
                        transcript.append({"role": "assistant", "content": decision.raw_response})
                        transcript.append(
                            {
                                "role": "tool",
                                "tool_name": "orchestration_policy",
                                "ok": False,
                                "error": (
                                    "At least one tool call is required before answering. "
                                    "Choose an available tool and provide valid arguments."
                                ),
                            }
                        )
                        turn_number += 1
                        continue
                    missing_required = missing_required_tools(task, steps)
                    if missing_required:
                        output = self._step_output(
                            {
                                "raw_response": decision.raw_response,
                                "error": "final_response_before_required_tools",
                                "missing_required_tools": list(missing_required),
                            },
                            messages=messages,
                            tools=tools,
                        )
                        steps.append(
                            WorkflowStep(
                                name=f"model_turn_{turn_number}",
                                status=WorkflowStatus.FAILED,
                                output=output,
                            )
                        )
                        await self._save_progress(trace, steps)
                        transcript.append({"role": "assistant", "content": decision.raw_response})
                        transcript.append(
                            {
                                "role": "tool",
                                "tool_name": "orchestration_policy",
                                "ok": False,
                                "error": (
                                    "Before answering, call these required tools: "
                                    + ", ".join(missing_required)
                                ),
                            }
                        )
                        turn_number += 1
                        continue
                    verification_needed = missing_verification_after_write(task, steps)
                    if verification_needed:
                        output = self._step_output(
                            {
                                "raw_response": decision.raw_response,
                                "error": "final_response_before_verification",
                            },
                            messages=messages,
                            tools=tools,
                        )
                        steps.append(
                            WorkflowStep(
                                name=f"model_turn_{turn_number}",
                                status=WorkflowStatus.FAILED,
                                output=output,
                            )
                        )
                        await self._save_progress(trace, steps)
                        transcript.append({"role": "assistant", "content": decision.raw_response})
                        transcript.append(
                            {
                                "role": "tool",
                                "tool_name": "orchestration_policy",
                                "ok": False,
                                "error": (
                                    "A write succeeded during this repair task, but verification "
                                    "has not passed after the latest write. Run test.run before "
                                    "answering."
                                ),
                            }
                        )
                        turn_number += 1
                        continue
                    final_ok = decision.ok and not has_unresolved_failed_tool_step(steps)
                    return await self._finish(
                        trace=trace,
                        steps=[
                            *steps,
                            WorkflowStep(
                                name="final_response",
                                status=WorkflowStatus.SUCCEEDED
                                if final_ok
                                else WorkflowStatus.FAILED,
                                output=self._step_output(
                                    {
                                        "raw_response": decision.raw_response,
                                        "response": decision.response,
                                        "ok": final_ok,
                                    },
                                    messages=messages,
                                    tools=tools,
                                ),
                            ),
                        ],
                        run_metadata=task.run_metadata,
                        ok=final_ok,
                        response=decision.response,
                        tool_calls_made=tool_calls_made,
                    )

                if decision.kind == "parse_error":
                    output = self._step_output(
                        {
                            "raw_response": decision.raw_response,
                            "error": decision.error,
                        },
                        messages=messages,
                        tools=tools,
                    )
                    steps.append(
                        WorkflowStep(
                            name=f"model_turn_{turn_number}",
                            status=WorkflowStatus.FAILED,
                            output=output,
                        )
                    )
                    await self._save_progress(trace, steps)
                    transcript.append({"role": "assistant", "content": decision.raw_response})
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_name": "model_response_parser",
                            "ok": False,
                            "error": decision.error,
                        }
                    )
                    turn_number += 1
                    continue

                if final_response_only:
                    output = self._step_output(
                        {
                            "raw_response": decision.raw_response,
                            "error": (
                                "tool_call_after_discovery_sufficient"
                                if discovery_sufficient
                                else "tool_call_after_budget_exhausted"
                            ),
                        },
                        messages=messages,
                        tools=tools,
                    )
                    steps.append(
                        WorkflowStep(
                            name=f"model_turn_{turn_number}",
                            status=WorkflowStatus.FAILED,
                            output=output,
                        )
                    )
                    await self._save_progress(trace, steps)
                    transcript.append({"role": "assistant", "content": decision.raw_response})
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_name": "orchestration_policy",
                            "ok": False,
                            "error": (
                                "No more tool calls are allowed. Use the existing tool_results "
                                "and answer the user."
                            ),
                        }
                    )
                    turn_number += 1
                    continue

                tool_call = ToolCall(
                    tool_name=decision.tool_name or "",
                    arguments=decision.arguments,
                )
                if tool_call.tool_name not in task.available_tools:
                    if is_repeated_unavailable_tool(tool_call, steps):
                        # Already told model this tool is unavailable; block and
                        # ask for a final answer to stop the loop.
                        unavail_result = ToolResult(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            ok=False,
                            error="repeated_unavailable_tool",
                            output={
                                "duplicate": True,
                                "message": (
                                    f"'{tool_call.tool_name}' is not available. "
                                    "Do not call this tool again. "
                                    "Answer the user."
                                ),
                            },
                        )
                        output = self._step_output(
                            {
                                "raw_response": decision.raw_response,
                                "error": "repeated_unavailable_tool",
                            },
                            messages=messages,
                            tools=tools,
                        )
                        steps.append(
                            WorkflowStep(
                                name=f"model_turn_{turn_number}",
                                status=WorkflowStatus.FAILED,
                                tool_call=tool_call,
                                tool_result=unavail_result,
                                output=output,
                            )
                        )
                        await self._save_progress(trace, steps)
                        transcript.append({"role": "assistant", "content": decision.raw_response})
                        turn_number += 1
                        continue
                    tool_result = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        ok=False,
                        error=f"tool is not available: {tool_call.tool_name}",
                    )
                elif is_duplicate_successful_write(tool_call, steps):
                    dedup_result = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        ok=True,
                        output={
                            "applied": False,
                            "duplicate": True,
                            "message": (
                                "File already patched in a previous step. "
                                "No further writes are needed. "
                                "Answer the user."
                            ),
                        },
                    )
                    output = self._step_output(
                        {
                            "raw_response": decision.raw_response,
                            "error": "duplicate_successful_write",
                        },
                        messages=messages,
                        tools=tools,
                    )
                    steps.append(
                        WorkflowStep(
                            name=f"model_turn_{turn_number}",
                            status=WorkflowStatus.FAILED,
                            tool_call=tool_call,
                            tool_result=dedup_result,
                            output=output,
                        )
                    )
                    await self._save_progress(trace, steps)
                    transcript.append({"role": "assistant", "content": decision.raw_response})
                    turn_number += 1
                    continue
                elif is_duplicate_read_or_search(tool_call, steps):
                    # This is a policy rejection, not a real tool result.
                    # Add it to the transcript as role="tool" (like other policy
                    # rejections) so the model sees it as immediate active
                    # feedback in tool_results, not buried in tool_history.
                    output = self._step_output(
                        {
                            "raw_response": decision.raw_response,
                            "error": "duplicate_read_or_search",
                        },
                        messages=messages,
                        tools=tools,
                    )
                    steps.append(
                        WorkflowStep(
                            name=f"model_turn_{turn_number}",
                            status=WorkflowStatus.FAILED,
                            tool_call=tool_call,
                            tool_result=ToolResult(
                                tool_call_id=tool_call.id,
                                tool_name=tool_call.tool_name,
                                ok=False,
                                error="duplicate_read_or_search",
                            ),
                            output=output,
                        )
                    )
                    await self._save_progress(trace, steps)
                    transcript.append({"role": "assistant", "content": decision.raw_response})
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_name": "orchestration_policy",
                            "ok": False,
                            "error": (
                                f"You already retrieved this from {tool_call.tool_name} "
                                "in a previous step "
                                "and the content is visible in your tool_history. "
                                "Do not repeat this call. "
                                "Use the existing results to answer the user."
                            ),
                        }
                    )
                    turn_number += 1
                    # Hard-stop: if the model has now been blocked on this
                    # exact call 2+ times, it is stuck in a dedup loop.
                    # Advance past max_turns so the next iteration runs with
                    # force_final_response=True (final-answer-only prompt).
                    if dedup_block_count_for_call(tool_call, steps) >= 2:
                        turn_number = task.max_turns + 1
                    continue
                else:
                    if task.on_turn:
                        await task.on_turn(
                            turn_number,
                            task.max_turns,
                            _tool_call_label(tool_call.tool_name, tool_call.arguments),
                        )
                    tool_result = await self.tool_executor.execute(tool_call)

                tool_calls_made += 1
                steps.append(
                    WorkflowStep(
                        name=f"tool_call_{tool_calls_made}",
                        status=WorkflowStatus.SUCCEEDED
                        if tool_result.ok
                        else WorkflowStatus.FAILED,
                        tool_call=tool_call,
                        tool_result=tool_result,
                        output=self._step_output(
                            {
                                "raw_response": decision.raw_response,
                                "reason": decision.reason,
                            },
                            messages=messages,
                            tools=tools,
                        ),
                    )
                )
                await self._save_progress(trace, steps)
                transcript.append({"role": "assistant", "content": decision.raw_response})
                turn_number += 1

            return await self._finish(
                trace=trace,
                steps=steps,
                run_metadata=task.run_metadata,
                ok=False,
                response="model did not produce a final response before max_turns",
                tool_calls_made=tool_calls_made,
                error="max_turns_exceeded",
            )
        except asyncio.CancelledError:
            return await self._finish(
                trace=trace,
                steps=steps,
                run_metadata=task.run_metadata,
                ok=False,
                response="model run was cancelled before completion",
                tool_calls_made=tool_calls_made,
                error="cancelled",
                status=WorkflowStatus.CANCELLED,
            )

    async def _finish(
        self,
        *,
        trace: WorkflowTrace,
        steps: list[WorkflowStep],
        run_metadata: dict[str, Any],
        ok: bool,
        response: str,
        tool_calls_made: int,
        error: str | None = None,
        status: WorkflowStatus | None = None,
    ) -> ToolLoopAgentResult:
        """Save the final trace and return the caller-facing result."""

        final_output = {
            "ok": ok,
            "response": response,
            "tool_calls_made": tool_calls_made,
        }
        if error:
            final_output["error"] = error
        if run_metadata:
            final_output["run_metadata"] = run_metadata

        trace = replace(
            trace,
            status=status or (WorkflowStatus.SUCCEEDED if ok else WorkflowStatus.FAILED),
            steps=steps,
            final_output=final_output,
            updated_at=datetime.now(UTC),
        )
        await self.trace_store.save(trace)
        return ToolLoopAgentResult(
            trace_id=trace.id,
            ok=ok,
            response=response,
            turns_used=len(steps),
            tool_calls_made=tool_calls_made,
            trace=trace,
        )

    async def _complete_model(
        self,
        task: ToolLoopAgentTask,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> str:
        """Complete one model turn, optionally bounded by a per-turn timeout."""

        completion = self.model_provider.complete(messages, tools=tools or None)
        if task.model_timeout_seconds is None:
            return await completion
        return await asyncio.wait_for(completion, timeout=task.model_timeout_seconds)

    async def _save_progress(
        self,
        trace: WorkflowTrace,
        steps: list[WorkflowStep],
    ) -> None:
        """Persist the latest running trace state after each observable step."""

        await self.trace_store.save(
            replace(
                trace,
                status=WorkflowStatus.RUNNING,
                steps=list(steps),
                updated_at=datetime.now(UTC),
            )
        )

    def _step_output(
        self,
        output: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Attach the turn messages and (if present) raw_response to a trace step."""

        prompt: list[dict[str, Any]] | dict[str, Any] = (
            {"messages": messages, "tools": tools} if tools else messages
        )
        return {**output, "prompt": prompt}


def _thinking_message(tool_calls_made: int, final_response_only: bool) -> str:
    """Return a turn-start progress label for MCP consumers."""
    if final_response_only:
        return "finalizing"
    if tool_calls_made == 1:
        return "thinking (1 tool call made)"
    if tool_calls_made > 1:
        return f"thinking ({tool_calls_made} tool calls made)"
    return "thinking"


def _tool_call_label(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a concise label for a tool call to surface to MCP consumers."""
    detail = _tool_call_detail(tool_name, arguments)
    return f"{tool_name}: {detail}" if detail else tool_name


def _tool_call_detail(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "repo.read":
        files = arguments.get("files", [])
        paths = [f["path"] if isinstance(f, dict) else str(f) for f in files]
        return _join_with_limit([p for p in paths if p])
    if tool_name == "repo.search":
        return str(arguments.get("query") or arguments.get("glob") or "")
    if tool_name == "repo.write_patch":
        changed = arguments.get("expected_changed_files", [])
        return _join_with_limit([str(f) for f in changed])
    if tool_name == "repo.write_files":
        files = arguments.get("files", [])
        paths = [f["path"] if isinstance(f, dict) else str(f) for f in files]
        return _join_with_limit([p for p in paths if p])
    if tool_name == "test.run":
        return str(arguments.get("command_name") or "")
    if tool_name == "git.diff":
        paths = arguments.get("paths", [])
        return _join_with_limit([str(p) for p in paths])
    return ""


def _join_with_limit(items: list[str], limit: int = 3) -> str:
    if not items:
        return ""
    shown = items[:limit]
    extra = len(items) - limit
    result = ", ".join(shown)
    return f"{result} +{extra} more" if extra > 0 else result
