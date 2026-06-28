"""Model-driven agent loop for typed tool calls.

Unlike ``CodingAgent``, this agent lets the model choose which tool to call on
each turn. The Python code still enforces budgets, allowed tools, required
tools, JSON parsing, and trace capture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID

from micro_model_agent.application.ports import ModelProvider, ToolExecutor, TraceStore
from micro_model_agent.domain.contracts import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)

DEFAULT_TOOL_NAMES: tuple[str, ...] = (
    "repo.search",
    "repo.read",
    "repo.semantic_search",
    "repo.write_patch",
    "test.run",
    "git.diff",
)

# Python 3.12 type aliases keep the model response categories readable below.
type DecisionKind = Literal["tool_call", "final_response", "parse_error"]
type PromptContext = dict[str, Any] | str


@dataclass(frozen=True, slots=True)
class ToolLoopAgentTask:
    """Input for a model-driven tool-call loop."""

    goal: str
    available_tools: tuple[str, ...] = DEFAULT_TOOL_NAMES
    # max_turns prevents an unproductive model/tool conversation from running forever.
    max_turns: int = 8
    context: PromptContext = field(default_factory=dict)
    tool_schemas: dict[str, Any] = field(default_factory=dict)
    require_tool_call: bool = True
    max_tool_result_prompt_chars: int = 12_000
    # max_tool_calls can force the model to stop gathering data and answer from
    # the tool results it already has.
    max_tool_calls: int | None = None
    required_tools: tuple[str, ...] = ()
    capture_prompts: bool = False
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolLoopAgentResult:
    """Structured result returned by the tool-loop agent."""

    trace_id: UUID
    ok: bool
    response: str
    tool_calls_made: int
    trace: WorkflowTrace


@dataclass(frozen=True, slots=True)
class _ModelDecision:
    """Internal parsed version of the model's JSON response."""

    kind: DecisionKind
    raw_response: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    response: str = ""
    ok: bool = False
    error: str | None = None
    reason: str | None = None


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
        # transcript is the compact prompt history fed back to the model; steps
        # is the richer audit trail saved for users.
        steps: list[WorkflowStep] = []
        transcript: list[dict[str, Any]] = [{"role": "user", "content": task.goal}]
        if task.context:
            transcript.append({"role": "context", "content": task.context})

        tool_calls_made = 0
        for turn_number in range(1, task.max_turns + 1):
            # Each turn asks the model for exactly one JSON object: either a
            # tool call or a final response.
            prompt = self._build_prompt(
                task,
                transcript,
                tool_calls_made=tool_calls_made,
                steps=steps,
            )
            raw_response = await self.model_provider.complete(prompt)
            decision = self._parse_model_response(raw_response)

            if decision.kind == "final_response":
                # Some tasks require evidence from tools before an answer is
                # allowed, so early final responses are fed back as policy errors.
                if task.require_tool_call and tool_calls_made == 0:
                    output = {
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
                    continue
                missing_required_tools = self._missing_required_tools(task, steps)
                if missing_required_tools:
                    output = {
                        "raw_response": decision.raw_response,
                        "error": "final_response_before_required_tools",
                        "missing_required_tools": list(missing_required_tools),
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
                    transcript.append({"role": "assistant", "content": decision.raw_response})
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_name": "orchestration_policy",
                            "ok": False,
                            "error": (
                                "Before final_response, call these required tools: "
                                + ", ".join(missing_required_tools)
                            ),
                        }
                    )
                    continue
                # The final answer is only successful if the model said it was
                # OK and no previous tool call failed.
                final_ok = decision.ok and not self._has_failed_tool_step(steps)
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
                # Bad JSON does not immediately end the workflow. The parser
                # error is shown to the model so it has a chance to recover.
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
                transcript.append({"role": "assistant", "content": decision.raw_response})
                transcript.append(
                    {
                        "role": "tool",
                        "tool_name": "model_response_parser",
                        "ok": False,
                        "error": decision.error,
                    }
                )
                continue

            if self._tool_budget_exhausted(task, tool_calls_made, steps):
                # Once the budget is exhausted, the agent tells the model to
                # answer using existing tool results instead of calling more tools.
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
                continue

            tool_call = ToolCall(
                tool_name=decision.tool_name or "",
                arguments=decision.arguments,
            )
            # The model can ask for any string, but the executor only runs tools
            # that were explicitly made available for this task.
            if tool_call.tool_name not in task.available_tools:
                tool_result = ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    ok=False,
                    error=f"tool is not available: {tool_call.tool_name}",
                )
            else:
                tool_result = await self.tool_executor.execute(tool_call)

            tool_calls_made += 1
            # Store both the tool result and the raw model response that led to it.
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
            transcript.append({"role": "assistant", "content": decision.raw_response})

        return await self._finish(
            trace=trace,
            steps=steps,
            run_metadata=task.run_metadata,
            ok=False,
            response="model did not produce a final response before max_turns",
            tool_calls_made=tool_calls_made,
            error="max_turns_exceeded",
        )

    def _build_prompt(
        self,
        task: ToolLoopAgentTask,
        transcript: list[dict[str, Any]],
        *,
        tool_calls_made: int,
        steps: list[WorkflowStep],
    ) -> str:
        """Build the prompt that asks the model for one JSON decision."""

        budget_exhausted = self._tool_budget_exhausted(task, tool_calls_made, steps)
        missing_required_tools = self._missing_required_tools(task, steps)
        payload: dict[str, Any] = {
            "goal": task.goal,
            "available_tools": [] if budget_exhausted else list(task.available_tools),
            "context": self._prompt_context(task.context),
        }
        if task.required_tools:
            payload["required_tools"] = list(task.required_tools)
            payload["missing_required_tools"] = list(missing_required_tools)
        tool_history = self._tool_history(
            steps,
            max_prompt_chars=task.max_tool_result_prompt_chars,
        )
        if tool_history:
            payload["tool_history"] = tool_history
        policy_results = [entry for entry in transcript if entry.get("role") == "tool"]
        if policy_results:
            payload["tool_results"] = policy_results
        tool_schemas = self._available_tool_schemas(task)
        if tool_schemas and not budget_exhausted:
            payload["tool_schemas"] = tool_schemas

        if budget_exhausted:
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
                "Final responses must be concise and must not repeat full tool output. "
                "For a tool call, return "
                '{"tool_name":"repo.read","arguments":{"files":[{"path":"README.md"}]},'
                '"reason":"..."}. '
                "When the task is complete, return "
                '{"final_response":"Concise answer to the user.","ok":true}.'
            )
        return (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{json.dumps(payload, sort_keys=True)}\n"
            "<|assistant|>\n"
        )

    def _available_tool_schemas(self, task: ToolLoopAgentTask) -> dict[str, Any]:
        """Return schemas only for tools that are both available and documented."""

        return {
            tool_name: task.tool_schemas[tool_name]
            for tool_name in task.available_tools
            if tool_name in task.tool_schemas
        }

    def _prompt_context(self, context: PromptContext) -> PromptContext:
        """Normalize an empty context to a string so JSON output stays simple."""

        if context:
            return context
        return ""

    def _tool_budget_exhausted(
        self,
        task: ToolLoopAgentTask,
        tool_calls_made: int,
        steps: list[WorkflowStep],
    ) -> bool:
        """Return true when the task has used all allowed tool calls."""

        return (
            task.max_tool_calls is not None
            and tool_calls_made >= task.max_tool_calls
            and not self._missing_required_tools(task, steps)
        )

    def _has_failed_tool_step(self, steps: list[WorkflowStep]) -> bool:
        """Check whether any executed tool reported failure."""

        return any(step.tool_result is not None and not step.tool_result.ok for step in steps)

    def _missing_required_tools(
        self,
        task: ToolLoopAgentTask,
        steps: list[WorkflowStep],
    ) -> tuple[str, ...]:
        """List required tools that have not successfully appeared in the trace."""

        used_tools = {
            step.tool_call.tool_name
            for step in steps
            if step.tool_call is not None and step.tool_result is not None
        }
        return tuple(tool_name for tool_name in task.required_tools if tool_name not in used_tools)

    def _parse_model_response(self, raw_response: str) -> _ModelDecision:
        """Parse the model's JSON into one internal decision object."""

        try:
            payload = self._json_object_from_response(raw_response)
        except ValueError as exc:
            return _ModelDecision(
                kind="parse_error",
                raw_response=raw_response,
                error=str(exc),
            )

        final_response = payload.get("final_response", payload.get("response"))
        if final_response is not None:
            if not isinstance(final_response, str):
                return _ModelDecision(
                    kind="parse_error",
                    raw_response=raw_response,
                    error="final_response must be a string",
                )
            ok_value = payload.get("ok", True)
            return _ModelDecision(
                kind="final_response",
                raw_response=raw_response,
                response=final_response,
                ok=bool(ok_value) if isinstance(ok_value, bool) else True,
            )

        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str):
            return _ModelDecision(
                kind="parse_error",
                raw_response=raw_response,
                error="tool_name must be a string",
            )
        if not isinstance(arguments, dict):
            return _ModelDecision(
                kind="parse_error",
                raw_response=raw_response,
                error="arguments must be an object",
            )

        reason = payload.get("reason")
        return _ModelDecision(
            kind="tool_call",
            raw_response=raw_response,
            tool_name=tool_name,
            arguments=cast(dict[str, Any], arguments),
            ok=True,
            reason=reason if isinstance(reason, str) else None,
        )

    def _json_object_from_response(self, raw_response: str) -> dict[str, Any]:
        """Extract the first JSON object from plain text or fenced markdown."""

        payload = self._strip_markdown_fence(raw_response.strip())
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

    def _strip_markdown_fence(self, response: str) -> str:
        """Remove ``` fences because some models wrap JSON in markdown."""

        if not response.startswith("```"):
            return response

        lines = response.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _tool_result_message(
        self,
        tool_result: ToolResult,
        *,
        max_prompt_chars: int,
    ) -> dict[str, Any]:
        """Convert a ToolResult into the compact transcript message format."""

        return {
            "role": "tool",
            "tool_call_id": str(tool_result.tool_call_id),
            "tool_name": tool_result.tool_name,
            "ok": tool_result.ok,
            "output": self._truncate_tool_output(tool_result.output, max_prompt_chars),
            "error": tool_result.error,
        }

    def _tool_history(
        self,
        steps: list[WorkflowStep],
        *,
        max_prompt_chars: int,
    ) -> list[dict[str, Any]]:
        """Return prior tool calls/results, compacting outputs near the prompt budget."""

        history: list[dict[str, Any]] = []
        for step in steps:
            if step.tool_call is None or step.tool_result is None:
                continue
            history.append(
                {
                    "step": step.name,
                    "tool_call_id": str(step.tool_call.id),
                    "tool_name": step.tool_call.tool_name,
                    "arguments": step.tool_call.arguments,
                    "ok": step.tool_result.ok,
                    "output": step.tool_result.output,
                    "error": step.tool_result.error,
                }
            )

        if not history:
            return []

        history_json = json.dumps(history, sort_keys=True)
        if max_prompt_chars < 1 or len(history_json) <= max_prompt_chars:
            return history

        compacted: list[dict[str, Any]] = []
        for entry in history:
            compacted.append(
                {
                    **entry,
                    "output": self._summarize_tool_output(entry["output"]),
                }
            )

        compacted_json = json.dumps(compacted, sort_keys=True)
        if len(compacted_json) <= max_prompt_chars:
            return compacted

        # Keep the most recent calls with full call metadata. Older calls are
        # summarized into a count so repetition is still visible under pressure.
        kept: list[dict[str, Any]] = []
        omitted = 0
        for entry in reversed(compacted):
            candidate = [*reversed(kept), entry]
            candidate_json = json.dumps(candidate, sort_keys=True)
            if len(candidate_json) <= max_prompt_chars or not kept:
                kept.append(entry)
            else:
                omitted += 1
        result = list(reversed(kept))
        if omitted:
            result.insert(
                0,
                {
                    "compacted_history": True,
                    "omitted_older_tool_calls": omitted,
                },
            )
        return result

    def _summarize_tool_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Summarize bulky tool output while preserving decision-relevant facts."""

        summary: dict[str, Any] = {}
        if "matches" in output and isinstance(output["matches"], list):
            summary["match_count"] = len(output["matches"])
            summary["truncated"] = output.get("truncated", False)
        if "results" in output and isinstance(output["results"], list):
            summary["result_count"] = len(output["results"])
        if "files" in output and isinstance(output["files"], list):
            summary["file_count"] = len(output["files"])
            summary["paths"] = [
                item.get("path")
                for item in output["files"]
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            ][:20]
        if "changed_files" in output:
            summary["changed_files"] = output["changed_files"]
        if "errors" in output and output["errors"]:
            summary["errors"] = output["errors"]
        if "ok" in output:
            summary["ok"] = output["ok"]
        return summary or self._truncate_tool_output(output, 500)

    def _truncate_tool_output(
        self,
        output: dict[str, Any],
        max_prompt_chars: int,
    ) -> dict[str, Any]:
        """Shorten large tool outputs before they are placed back in the prompt."""

        output_json = json.dumps(output, sort_keys=True)
        if max_prompt_chars < 1 or len(output_json) <= max_prompt_chars:
            return output
        return {
            "truncated_for_prompt": True,
            "preview": output_json[:max_prompt_chars],
            "original_character_count": len(output_json),
        }

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
            status=WorkflowStatus.SUCCEEDED if ok else WorkflowStatus.FAILED,
            steps=steps,
            final_output=final_output,
            updated_at=datetime.now(UTC),
        )
        await self.trace_store.save(trace)
        return ToolLoopAgentResult(
            trace_id=trace.id,
            ok=ok,
            response=response,
            tool_calls_made=tool_calls_made,
            trace=trace,
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
