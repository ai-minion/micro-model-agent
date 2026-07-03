"""Reference coding agent workflow.

This agent follows a fixed sequence: retrieve context, ask the model for a
patch, preview/apply that patch, optionally run tests, and save a trace.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from micro_model_agent.execution.application.ports import (
    CodingAgentResult,
    CodingAgentTask,
    ModelProvider,
    ToolExecutor,
    TraceStore,
)
from micro_model_agent.execution.domain.value_objects import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)


class CodingAgent:
    """Small reference workflow for coding tasks."""

    def __init__(
        self,
        model_provider: ModelProvider,
        tool_executor: ToolExecutor,
        trace_store: TraceStore,
    ) -> None:
        self.model_provider = model_provider
        self.tool_executor = tool_executor
        self.trace_store = trace_store

    async def run(self, task: CodingAgentTask) -> CodingAgentResult:
        steps: list[WorkflowStep] = []
        trace = WorkflowTrace(goal=task.goal, status=WorkflowStatus.RUNNING)

        # Step 1: collect relevant repository context before asking the model to
        # write code. The tool result is recorded as a workflow step.
        retrieval = await self._execute_tool(
            name="retrieve_context",
            tool_name="repo.semantic_search",
            arguments={
                "query": task.goal,
                "intent": task.semantic_intent,
                "limit": task.semantic_limit,
            },
        )
        steps.append(retrieval)

        # Step 2: build the prompt, call the model, and normalize the patch so
        # command-line tools that expect a trailing newline can read it cleanly.
        prompt = self._build_prompt(task, retrieval.tool_result)
        patch = await self.model_provider.complete(prompt)
        if patch and not patch.endswith("\n"):
            patch = f"{patch}\n"
        steps.append(
            WorkflowStep(
                name="generate_patch",
                status=WorkflowStatus.SUCCEEDED if patch else WorkflowStatus.FAILED,
                output={"patch": patch},
            )
        )

        # Step 3: delegate patch validation/application to the repository tool
        # so path checks and approval rules live in one place.
        write_patch = await self._execute_tool(
            name="preview_or_apply_patch",
            tool_name="repo.write_patch",
            arguments={
                "patch": patch,
                "dry_run": task.dry_run,
                "require_approval": task.require_approval,
                "expected_changed_files": task.expected_changed_files,
            },
        )
        steps.append(write_patch)

        verification_passed: bool | None = None
        # Step 4: tests only run after a successful non-dry-run patch. In a dry
        # run, the trace records that verification was intentionally skipped.
        if self._step_ok(write_patch) and task.verification_command_name and not task.dry_run:
            verification = await self._execute_tool(
                name="verify_changes",
                tool_name="test.run",
                arguments={"command_name": task.verification_command_name},
            )
            verification_passed = self._step_ok(verification)
            steps.append(verification)
        elif task.verification_command_name and task.dry_run:
            steps.append(
                WorkflowStep(
                    name="verify_changes",
                    status=WorkflowStatus.SUCCEEDED,
                    output={"skipped": True, "reason": "dry_run"},
                )
            )
        elif task.verification_command_name:
            steps.append(
                WorkflowStep(
                    name="verify_changes",
                    status=WorkflowStatus.FAILED,
                    output={"skipped": True, "reason": "patch_failed"},
                )
            )

        # Step 5: once real files changed, capture the diff as extra trace data.
        if self._step_ok(write_patch) and not task.dry_run:
            steps.append(
                await self._execute_tool(
                    name="summarize_diff",
                    tool_name="git.diff",
                    arguments={"paths": self._changed_files(write_patch)},
                )
            )

        # Finish by turning the collected step data into a compact result for
        # callers and a full WorkflowTrace for audit/debugging.
        patch_applied = bool(
            write_patch.tool_result and write_patch.tool_result.output.get("applied")
        )
        changed_files = self._changed_files(write_patch)
        ok = self._step_ok(write_patch) and verification_passed is not False
        summary = self._summary(ok=ok, task=task, patch_applied=patch_applied)
        final_output: dict[str, Any] = {
            "ok": ok,
            "summary": summary,
            "patch_applied": patch_applied,
            "verification_passed": verification_passed,
            "changed_files": changed_files,
        }
        trace = replace(
            trace,
            status=WorkflowStatus.SUCCEEDED if ok else WorkflowStatus.FAILED,
            steps=steps,
            final_output=final_output,
            updated_at=datetime.now(UTC),
        )
        await self.trace_store.save(trace)

        return CodingAgentResult(
            trace_id=trace.id,
            ok=ok,
            summary=summary,
            patch_applied=patch_applied,
            verification_passed=verification_passed,
            changed_files=changed_files,
            trace=trace,
        )

    async def _execute_tool(
        self,
        name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> WorkflowStep:
        """Execute one tool and wrap its call/result in a workflow step."""

        tool_call = ToolCall(tool_name=tool_name, arguments=arguments)
        tool_result = await self.tool_executor.execute(tool_call)
        return WorkflowStep(
            name=name,
            status=WorkflowStatus.SUCCEEDED if tool_result.ok else WorkflowStatus.FAILED,
            tool_call=tool_call,
            tool_result=tool_result,
        )

    def _build_prompt(self, task: CodingAgentTask, retrieval_result: ToolResult | None) -> str:
        """Create the text prompt sent to the model provider."""

        context = retrieval_result.output if retrieval_result else {}
        return (
            "Generate a unified diff for the requested coding task.\n"
            f"Goal: {task.goal}\n"
            f"Retrieved context: {context}\n"
            "Return only the unified diff."
        )

    def _step_ok(self, step: WorkflowStep) -> bool:
        """Return true when a step has a successful tool result."""

        return bool(step.tool_result and step.tool_result.ok)

    def _changed_files(self, step: WorkflowStep) -> list[str]:
        """Extract changed file paths from a patch-writing tool result."""

        if step.tool_result is None:
            return []
        changed_files = step.tool_result.output.get("changed_files", [])
        return [str(path) for path in changed_files] if isinstance(changed_files, list) else []

    def _summary(self, ok: bool, task: CodingAgentTask, patch_applied: bool) -> str:
        if not ok:
            return "coding task failed"
        if task.dry_run:
            return "coding task dry-run patch validated"
        if patch_applied:
            return "coding task patch applied and verified"
        return "coding task completed"
