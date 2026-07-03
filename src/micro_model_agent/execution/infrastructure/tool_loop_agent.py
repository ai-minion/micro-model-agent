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

from micro_model_agent.execution.infrastructure.tool_loop_decisions import parse_model_response
from micro_model_agent.execution.infrastructure.tool_loop_policy import (
    has_unresolved_failed_tool_step,
    is_duplicate_successful_write,
    missing_required_tools,
    missing_verification_after_write,
    orchestration_hints,
    should_allow_extra_finalization_turn,
    tool_budget_exhausted,
)
from micro_model_agent.execution.infrastructure.tool_loop_prompting import build_prompt
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

        trace = WorkflowTrace(goal=task.goal, status=WorkflowStatus.RUNNING)
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
                final_response_only = force_final_response or budget_exhausted
                prompt = build_prompt(
                    task,
                    transcript,
                    tool_calls_made=tool_calls_made,
                    turn_number=turn_number,
                    final_response_only=final_response_only,
                    missing_required_tools=missing_required_tools(task, steps),
                    orchestration_hints=orchestration_hints(task, steps),
                    steps=steps,
                )
                try:
                    raw_response = await self._complete_model(task, prompt)
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
                                prompt=prompt,
                                capture_prompt=task.capture_prompts,
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
                        output: dict[str, Any] = {
                            "raw_response": decision.raw_response,
                            "error": "final_response_before_tool_call",
                        }
                        if task.capture_prompts:
                            output["prompt"] = prompt
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
                                    "At least one tool call is required before final_response. "
                                    "Choose an available tool and provide valid arguments."
                                ),
                            }
                        )
                        turn_number += 1
                        continue
                    missing_required = missing_required_tools(task, steps)
                    if missing_required:
                        output = {
                            "raw_response": decision.raw_response,
                            "error": "final_response_before_required_tools",
                            "missing_required_tools": list(missing_required),
                        }
                        if task.capture_prompts:
                            output["prompt"] = prompt
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
                                    "Before final_response, call these required tools: "
                                    + ", ".join(missing_required)
                                ),
                            }
                        )
                        turn_number += 1
                        continue
                    verification_needed = missing_verification_after_write(task, steps)
                    if verification_needed:
                        output = {
                            "raw_response": decision.raw_response,
                            "error": "final_response_before_verification",
                        }
                        if task.capture_prompts:
                            output["prompt"] = prompt
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
                                    "final_response."
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
                                    prompt=prompt,
                                    capture_prompt=task.capture_prompts,
                                ),
                            ),
                        ],
                        run_metadata=task.run_metadata,
                        ok=final_ok,
                        response=decision.response,
                        tool_calls_made=tool_calls_made,
                    )

                if decision.kind == "parse_error":
                    output = {
                        "raw_response": decision.raw_response,
                        "error": decision.error,
                    }
                    if task.capture_prompts:
                        output["prompt"] = prompt
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

                if force_final_response or tool_budget_exhausted(
                    task,
                    tool_calls_made,
                    steps,
                ):
                    output = {
                        "raw_response": decision.raw_response,
                        "error": "tool_call_after_budget_exhausted",
                    }
                    if task.capture_prompts:
                        output["prompt"] = prompt
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
                                "and return final_response."
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
                    tool_result = ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        ok=False,
                        error=f"tool is not available: {tool_call.tool_name}",
                    )
                elif is_duplicate_successful_write(tool_call, steps):
                    output = {
                        "raw_response": decision.raw_response,
                        "error": "duplicate_successful_write",
                    }
                    if task.capture_prompts:
                        output["prompt"] = prompt
                    steps.append(
                        WorkflowStep(
                            name=f"model_turn_{turn_number}",
                            status=WorkflowStatus.FAILED,
                            tool_call=tool_call,
                            output=output,
                        )
                    )
                    await self._save_progress(trace, steps)
                    transcript.append({"role": "assistant", "content": decision.raw_response})
                    turn_number += 1
                    continue
                else:
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
                            prompt=prompt,
                            capture_prompt=task.capture_prompts,
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

    async def _complete_model(self, task: ToolLoopAgentTask, prompt: str) -> str:
        """Complete one model turn, optionally bounded by a per-turn timeout."""

        completion = self.model_provider.complete(prompt)
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
        prompt: str,
        capture_prompt: bool,
    ) -> dict[str, Any]:
        """Attach the exact prompt to a trace step when collection asks for it."""

        if not capture_prompt:
            return output
        return {**output, "prompt": prompt}
