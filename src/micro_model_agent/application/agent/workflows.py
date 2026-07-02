"""Application services for running and labeling agent workflows."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from micro_model_agent.application.ports.contracts import (
    CodingAgentResult,
    CodingAgentTask,
    CodingWorkflowRunner,
    DatasetExampleStore,
    WorkflowEvaluator,
)
from micro_model_agent.domain.contracts import EvaluationResult, WorkflowTrace
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)


class DefaultWorkflowEvaluator:
    """Evaluate a completed coding workflow from its captured trace."""

    async def evaluate(self, trace: WorkflowTrace) -> EvaluationResult:
        # The evaluator reads the normalized final_output dictionary instead of
        # knowing every implementation detail of the agent that produced it.
        final = trace.final_output
        ok = bool(final.get("ok"))
        verification_passed = final.get("verification_passed")
        patch_applied = bool(final.get("patch_applied"))
        score = 1.0 if ok else 0.0
        if ok and verification_passed is None:
            score = 0.8
        if ok and not patch_applied:
            score = min(score, 0.7)
        return EvaluationResult(
            passed=ok,
            summary="workflow passed" if ok else "workflow failed",
            score=score,
            details={
                "trace_id": str(trace.id),
                "status": trace.status.value,
                "patch_applied": patch_applied,
                "verification_passed": verification_passed,
                "changed_files": list(final.get("changed_files", [])),
            },
        )


class TraceDatasetBuilder:
    """Build fine-tuning-ready dataset examples from workflow traces."""

    def build_example(
        self,
        trace: WorkflowTrace,
        label: DatasetLabel,
        *,
        kind: DatasetExampleKind = DatasetExampleKind.REPAIR,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetExample:
        # Training data separates what the model saw (input_payload) from the
        # answer we want it to learn to produce (target_payload).
        patch = self._generated_patch(trace)
        input_payload = {
            "goal": trace.goal,
            "retrieved_context": self._retrieval_output(trace),
            "steps": [step.name for step in trace.steps],
            "tool_history": self._tool_history(trace),
        }
        target_payload = {
            "patch": patch,
            "summary": trace.final_output.get("summary") or trace.final_output.get("response"),
            "final_response": trace.final_output.get("response"),
            "changed_files": list(trace.final_output.get("changed_files", [])),
        }
        return DatasetExample(
            kind=kind,
            input=input_payload,
            target=target_payload,
            label=label,
            source=f"trace:{trace.id}",
            metadata={
                "trace_id": str(trace.id),
                "workflow_status": trace.status.value,
                **(metadata or {}),
            },
        )

    def _generated_patch(self, trace: WorkflowTrace) -> str | None:
        """Return the patch produced by the generate_patch step, if present."""

        for step in trace.steps:
            if step.name == "generate_patch":
                patch = step.output.get("patch")
                return str(patch) if patch is not None else None
        return None

    def _retrieval_output(self, trace: WorkflowTrace) -> dict[str, Any]:
        """Return the repository context gathered before patch generation."""

        for step in trace.steps:
            if step.name == "retrieve_context" and step.tool_result:
                return dict(step.tool_result.output)
        return {}

    def _tool_history(self, trace: WorkflowTrace) -> list[dict[str, Any]]:
        """Return tool calls and results captured by model-driven loop traces."""

        history: list[dict[str, Any]] = []
        for step in trace.steps:
            if not step.tool_call and not step.tool_result:
                continue
            item: dict[str, Any] = {"step": step.name, "status": step.status.value}
            if step.tool_call:
                item["tool_call"] = {
                    "tool_name": step.tool_call.tool_name,
                    "arguments": step.tool_call.arguments,
                }
            if step.tool_result:
                item["tool_result"] = {
                    "tool_name": step.tool_result.tool_name,
                    "ok": step.tool_result.ok,
                    "output": step.tool_result.output,
                    "error": step.tool_result.error,
                }
            history.append(item)
        return history


class RunAgentWorkflow:
    """Run the coding agent, evaluate the trace, and optionally store a labeled example."""

    def __init__(
        self,
        agent: CodingWorkflowRunner,
        evaluator: WorkflowEvaluator | None = None,
        dataset_store: DatasetExampleStore | None = None,
        dataset_builder: TraceDatasetBuilder | None = None,
    ) -> None:
        self.agent = agent
        self.evaluator = evaluator or DefaultWorkflowEvaluator()
        self.dataset_store = dataset_store
        self.dataset_builder = dataset_builder or TraceDatasetBuilder()

    async def run(
        self,
        task: CodingAgentTask,
        *,
        label: DatasetLabel | None = None,
    ) -> CodingAgentResult:
        # Run the agent first, then attach an evaluation summary to the saved
        # trace. dataclasses.replace creates a new frozen dataclass instance
        # with selected fields changed.
        result = await self.agent.run(task)
        evaluation = await self.evaluator.evaluate(result.trace)
        trace = replace(
            result.trace,
            final_output={
                **result.trace.final_output,
                "evaluation": {
                    "passed": evaluation.passed,
                    "summary": evaluation.summary,
                    "score": evaluation.score,
                    "details": evaluation.details,
                },
            },
            updated_at=datetime.now(UTC),
        )
        await self.agent.trace_store.save(trace)

        if label and self.dataset_store:
            # Labeled traces become examples for later supervised fine-tuning.
            await self.dataset_store.save(self.dataset_builder.build_example(trace, label))

        return replace(result, trace=trace)


def label_from_workflow_result(
    result: CodingAgentResult,
    *,
    outcome: OutcomeLabel | None = None,
    quality: QualityLabel | None = None,
    failure_modes: tuple[FailureMode, ...] | None = None,
    reviewer_notes: str | None = None,
) -> DatasetLabel:
    """Create a default label for a workflow result, allowing CLI overrides."""

    # Successful workflows default to accepted/good labels; failures default to
    # rejected/bad so the exported dataset can distinguish useful from bad runs.
    inferred_outcome = OutcomeLabel.ACCEPTED if result.ok else OutcomeLabel.REJECTED
    inferred_quality = QualityLabel.GOOD if result.ok else QualityLabel.BAD
    inferred_failures: tuple[FailureMode, ...] = ()
    if not result.ok:
        inferred_failures = (FailureMode.BAD_PATCH,)
        if result.verification_passed is False:
            inferred_failures = (FailureMode.TEST_FAILED,)
    return DatasetLabel(
        outcome=outcome or inferred_outcome,
        quality=quality or inferred_quality,
        failure_modes=failure_modes if failure_modes is not None else inferred_failures,
        reviewer_notes=reviewer_notes,
    )
