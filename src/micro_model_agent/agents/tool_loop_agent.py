"""Model-driven agent loop for typed tool calls.

Unlike ``CodingAgent``, this agent lets the model choose which tool to call on
each turn. The Python code still enforces budgets, allowed tools, required
tools, JSON parsing, and trace capture.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from micro_model_agent.agents.tool_loop_decisions import parse_model_response
from micro_model_agent.application.ports import ModelProvider, ToolExecutor, TraceStore
from micro_model_agent.application.tool_loop import (
    DEFAULT_TOOL_NAMES,
    PromptContext,
    ToolLoopAgentResult,
    ToolLoopAgentTask,
)
from micro_model_agent.domain.contracts import (
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
                and self._should_allow_extra_finalization_turn(task, tool_calls_made, steps)
            ):
                force_final_response = turn_number > task.max_turns
                if force_final_response:
                    used_extra_finalization_turn = True
                prompt = self._build_prompt(
                    task,
                    transcript,
                    tool_calls_made=tool_calls_made,
                    turn_number=turn_number,
                    force_final_response=force_final_response,
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
                        await self._save_progress(trace, steps)
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
                        turn_number += 1
                        continue
                    verification_needed = self._missing_verification_after_write(task, steps)
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
                    final_ok = decision.ok and not self._has_unresolved_failed_tool_step(steps)
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

                if force_final_response or self._tool_budget_exhausted(
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
                elif self._is_duplicate_successful_write(tool_call, steps):
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
    def _build_prompt(
        self,
        task: ToolLoopAgentTask,
        transcript: list[dict[str, Any]],
        *,
        tool_calls_made: int,
        turn_number: int,
        force_final_response: bool = False,
        steps: list[WorkflowStep],
    ) -> str:
        """Build the prompt that asks the model for one JSON decision."""

        budget_exhausted = self._tool_budget_exhausted(task, tool_calls_made, steps)
        final_response_only = force_final_response or budget_exhausted
        missing_required_tools = self._missing_required_tools(task, steps)
        payload: dict[str, Any] = {
            "goal": task.goal,
            "available_tools": [] if final_response_only else list(task.available_tools),
            "context": self._prompt_context(task.context),
            "loop_budget": self._loop_budget_payload(
                task,
                tool_calls_made=tool_calls_made,
                turn_number=turn_number,
                final_response_only=final_response_only,
            ),
        }
        if task.required_tools:
            payload["required_tools"] = list(task.required_tools)
            payload["missing_required_tools"] = list(missing_required_tools)
        orchestration_hints = self._orchestration_hints(task, steps)
        if orchestration_hints:
            payload["orchestration_hints"] = orchestration_hints
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

    def _loop_budget_payload(
        self,
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

    def _should_allow_extra_finalization_turn(
        self,
        task: ToolLoopAgentTask,
        tool_calls_made: int,
        steps: list[WorkflowStep],
    ) -> bool:
        """Allow exactly one final-answer turn after a useful last action."""

        return (
            self._tool_budget_exhausted(task, tool_calls_made, steps)
            or (bool(steps) and steps[-1].tool_result is not None)
            or self._is_duplicate_write_block_step(steps[-1] if steps else None)
        )

    def _is_duplicate_write_block_step(self, step: WorkflowStep | None) -> bool:
        """Return whether a step blocked a repeated write and should now finalize."""

        return (
            step is not None
            and step.tool_call is not None
            and step.tool_call.tool_name == "repo.write_files"
            and step.output.get("error") == "duplicate_successful_write"
        )

    def _has_unresolved_failed_tool_step(self, steps: list[WorkflowStep]) -> bool:
        """Check whether a tool failure still represents the final task state."""

        last_success_by_tool = {
            step.tool_call.tool_name: index
            for index, step in enumerate(steps)
            if step.tool_call is not None
            and step.tool_result is not None
            and step.tool_result.ok
        }
        latest_successful_write = max(
            (
                index
                for index, step in enumerate(steps)
                if step.tool_call is not None
                and step.tool_call.tool_name in {"repo.write_patch", "repo.write_files"}
                and step.tool_result is not None
                and step.tool_result.ok
            ),
            default=None,
        )
        for index, step in enumerate(steps):
            if (
                step.tool_call is None
                or step.tool_result is None
                or step.tool_result.ok
                or self._is_repaired_tool_failure(
                    step,
                    step_index=index,
                    last_success_by_tool=last_success_by_tool,
                    latest_successful_write=latest_successful_write,
                )
            ):
                continue
            return True
        return False

    def _is_repaired_tool_failure(
        self,
        step: WorkflowStep,
        *,
        step_index: int,
        last_success_by_tool: dict[str, int],
        latest_successful_write: int | None,
    ) -> bool:
        """Return true when a later action supersedes a failed tool attempt."""

        if step.tool_call is None:
            return False
        tool_name = step.tool_call.tool_name
        if last_success_by_tool.get(tool_name, -1) > step_index:
            return True
        return tool_name == "test.run" and (
            latest_successful_write is not None and latest_successful_write > step_index
        )

    def _is_duplicate_successful_write(
        self,
        tool_call: ToolCall,
        steps: list[WorkflowStep],
    ) -> bool:
        """Return true when repo.write_files repeats a prior successful file set."""

        if tool_call.tool_name != "repo.write_files":
            return False
        requested_paths = self._write_file_paths(tool_call.arguments)
        if not requested_paths:
            return False
        for step in steps:
            if (
                step.tool_call is None
                or step.tool_result is None
                or not step.tool_result.ok
                or step.tool_call.tool_name != "repo.write_files"
            ):
                continue
            if self._write_file_paths(step.tool_call.arguments) == requested_paths:
                return True
        return False

    def _write_file_paths(self, arguments: dict[str, Any]) -> tuple[str, ...]:
        """Extract a stable file-path tuple from repo.write_files arguments."""

        files = arguments.get("files")
        if not isinstance(files, list):
            return ()
        paths = []
        for file in files:
            if not isinstance(file, dict) or not isinstance(file.get("path"), str):
                return ()
            paths.append(file["path"])
        return tuple(sorted(paths))

    def _orchestration_hints(
        self,
        task: ToolLoopAgentTask,
        steps: list[WorkflowStep],
    ) -> list[str]:
        """Return compact process hints derived from repeated tool outcomes."""

        hints: list[str] = []
        if self._looks_like_creation_task(task):
            empty_searches = sum(1 for step in steps if self._is_empty_search_step(step))
            if empty_searches >= 2 and "repo.write_files" in task.available_tools:
                hints.append(
                    "This appears to be a creation/scaffolding task in an empty workspace. "
                    "Repeated search calls returned no matches; stop searching and use "
                    "repo.write_files to create the requested files."
                )

        failed_patch_writes = [
            step
            for step in steps
            if step.tool_call is not None
            and step.tool_call.tool_name == "repo.write_patch"
            and step.tool_result is not None
            and not step.tool_result.ok
            and step.tool_result.error == "tool argument validation failed"
        ]
        if failed_patch_writes and "repo.write_files" in task.available_tools:
            hints.append(
                "A repo.write_patch call failed validation. For new files and scaffolds, "
                "use repo.write_files with explicit path/content entries instead of "
                "hand-authoring unified diffs."
            )
        failed_write_files = [
            step
            for step in steps
            if step.tool_call is not None
            and step.tool_call.tool_name == "repo.write_files"
            and step.tool_result is not None
            and not step.tool_result.ok
            and step.tool_result.error == "tool argument validation failed"
        ]
        if failed_write_files and "repo.write_files" in task.available_tools:
            hints.append(
                'A repo.write_files call failed validation. Use arguments shaped exactly like '
                '{"files":[{"path":"README.md","content":"# Title\\n"}],'
                '"dry_run":false,"require_approval":false}; do not use a paths map, '
                "filename keys, or top-level content."
            )
        if self._missing_verification_after_write(task, steps):
            hints.append(
                "This task is about a failing test/import/verification issue. A file write "
                "has succeeded, but no test.run has passed after the latest write. Run "
                "test.run before final_response."
            )
        successful_writes = [
            step
            for step in steps
            if step.tool_call is not None
            and step.tool_call.tool_name == "repo.write_files"
            and step.tool_result is not None
            and step.tool_result.ok
        ]
        if successful_writes:
            hints.append(
                "A repo.write_files call already succeeded. Do not repeat the same write. "
                "Return final_response, or run a distinct verification tool if one is available."
            )
        return hints

    def _looks_like_creation_task(self, task: ToolLoopAgentTask) -> bool:
        """Heuristic for greenfield creation/scaffolding requests."""

        text = f"{task.goal} {task.context}".lower()
        return any(
            term in text
            for term in (
                "create",
                "scaffold",
                "skeleton",
                "greenfield",
                "empty repository",
                "empty workspace",
                "new file",
                "new project",
            )
        )

    def _is_empty_search_step(self, step: WorkflowStep) -> bool:
        """Return true for successful search-style calls with zero results."""

        if step.tool_call is None or step.tool_result is None or not step.tool_result.ok:
            return False
        if step.tool_call.tool_name == "repo.search":
            matches = step.tool_result.output.get("matches")
            return isinstance(matches, list) and len(matches) == 0
        if step.tool_call.tool_name == "repo.semantic_search":
            results = step.tool_result.output.get("results")
            return isinstance(results, list) and len(results) == 0
        return False

    def _missing_required_tools(
        self,
        task: ToolLoopAgentTask,
        steps: list[WorkflowStep],
    ) -> tuple[str, ...]:
        """List required tools that have not successfully appeared in the trace."""

        used_tools = {
            step.tool_call.tool_name
            for step in steps
            if step.tool_call is not None
            and step.tool_result is not None
            and step.tool_result.ok
        }
        return tuple(
            tool_name
            for tool_name in task.required_tools
            if not self._required_tool_satisfied(tool_name, used_tools)
        )

    def _required_tool_satisfied(self, required_tool: str, used_tools: set[str]) -> bool:
        """Treat successful repository writes as equivalent required write evidence."""

        if required_tool in {"repo.write_patch", "repo.write_files"}:
            return bool({"repo.write_patch", "repo.write_files"} & used_tools)
        return required_tool in used_tools

    def _missing_verification_after_write(
        self,
        task: ToolLoopAgentTask,
        steps: list[WorkflowStep],
    ) -> bool:
        """Require test.run after writes for explicit test/import repair tasks."""

        if "test.run" not in task.available_tools or not self._looks_like_verification_task(task):
            return False
        latest_write_index: int | None = None
        latest_passing_test_index: int | None = None
        for index, step in enumerate(steps):
            if step.tool_call is None or step.tool_result is None or not step.tool_result.ok:
                continue
            if step.tool_call.tool_name in {"repo.write_patch", "repo.write_files"}:
                latest_write_index = index
            if step.tool_call.tool_name == "test.run":
                output_ok = step.tool_result.output.get("ok")
                exit_code = step.tool_result.output.get("exit_code")
                if output_ok is True or exit_code == 0:
                    latest_passing_test_index = index
        return latest_write_index is not None and (
            latest_passing_test_index is None or latest_passing_test_index < latest_write_index
        )

    def _looks_like_verification_task(self, task: ToolLoopAgentTask) -> bool:
        """Heuristic for tasks whose success depends on running verification."""

        text = f"{task.goal} {task.context}".lower()
        return any(
            term in text
            for term in (
                "pytest",
                "test failure",
                "tests fail",
                "test fails",
                "failing test",
                "failing import",
                "import error",
                "modulenotfounderror",
                "runs successfully",
                "verification",
            )
        )

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
                    "arguments": self._summarize_tool_arguments(
                        step.tool_call.tool_name,
                        step.tool_call.arguments,
                    ),
                    "ok": step.tool_result.ok,
                    "output": step.tool_result.output,
                    "error": step.tool_result.error,
                }
            )

        if not history:
            return []

        history = self._compact_path_tool_history(history)
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

    def _compact_path_tool_history(
        self,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep only the latest read/write context for each repository path."""

        compacted_reversed: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for entry in reversed(history):
            paths = self._history_entry_paths(entry)
            if not paths:
                compacted_reversed.append(entry)
                continue

            unseen_paths = [path for path in paths if path not in seen_paths]
            if not unseen_paths:
                continue

            compacted_reversed.append(self._history_entry_with_paths(entry, unseen_paths))
            seen_paths.update(unseen_paths)
        return list(reversed(compacted_reversed))

    def _history_entry_paths(self, entry: dict[str, Any]) -> list[str]:
        """Return repository paths represented by one prompt history entry."""

        tool_name = entry.get("tool_name")
        arguments = entry.get("arguments")
        if not isinstance(arguments, dict):
            return []
        if tool_name == "repo.write_files":
            paths = arguments.get("paths")
            if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
                return list(paths)
        if tool_name == "repo.read":
            files = arguments.get("files")
            if isinstance(files, list):
                return [
                    file["path"]
                    for file in files
                    if isinstance(file, dict) and isinstance(file.get("path"), str)
                ]
        return []

    def _history_entry_with_paths(
        self,
        entry: dict[str, Any],
        paths: list[str],
    ) -> dict[str, Any]:
        """Return a history entry narrowed to the requested path subset."""

        tool_name = entry.get("tool_name")
        allowed_paths = set(paths)
        narrowed = dict(entry)
        arguments = entry.get("arguments")
        if isinstance(arguments, dict):
            narrowed["arguments"] = self._history_arguments_with_paths(
                tool_name,
                arguments,
                allowed_paths,
            )
        output = entry.get("output")
        if isinstance(output, dict):
            narrowed["output"] = self._history_output_with_paths(output, allowed_paths)
        return narrowed

    def _history_arguments_with_paths(
        self,
        tool_name: object,
        arguments: dict[str, Any],
        allowed_paths: set[str],
    ) -> dict[str, Any]:
        narrowed = dict(arguments)
        if tool_name == "repo.write_files":
            narrowed["paths"] = [
                path
                for path in arguments.get("paths", [])
                if isinstance(path, str) and path in allowed_paths
            ]
            narrowed["file_count"] = len(narrowed["paths"])
        elif tool_name == "repo.read":
            files = arguments.get("files")
            if isinstance(files, list):
                narrowed["files"] = [
                    file
                    for file in files
                    if isinstance(file, dict) and file.get("path") in allowed_paths
                ]
        return narrowed

    def _history_output_with_paths(
        self,
        output: dict[str, Any],
        allowed_paths: set[str],
    ) -> dict[str, Any]:
        narrowed = dict(output)
        files = output.get("files")
        if isinstance(files, list):
            narrowed["files"] = [
                file
                for file in files
                if isinstance(file, dict) and file.get("path") in allowed_paths
            ]
            if "file_count" in narrowed:
                narrowed["file_count"] = len(narrowed["files"])
        paths = output.get("paths")
        if isinstance(paths, list):
            narrowed["paths"] = [
                path for path in paths if isinstance(path, str) and path in allowed_paths
            ]
        changed_files = output.get("changed_files")
        if isinstance(changed_files, list):
            narrowed["changed_files"] = [
                path
                for path in changed_files
                if isinstance(path, str) and path in allowed_paths
            ]
        return narrowed

    def _summarize_tool_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep prior tool arguments useful without inviting content copying."""

        if tool_name != "repo.write_files":
            return arguments
        paths = self._write_file_paths(arguments)
        return {
            "dry_run": arguments.get("dry_run"),
            "file_count": len(paths),
            "paths": list(paths),
        }

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
