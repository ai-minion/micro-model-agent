"""Application services for running and labeling agent workflows."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from micro_model_agent.dataset.application.ports import DatasetExampleStore
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.acl import ExecutionToDatasetTranslator
from micro_model_agent.execution.application.ports import (
    CodingAgentResult,
    CodingAgentTask,
    CodingWorkflowRunner,
    WorkflowEvaluator,
)
from micro_model_agent.execution.domain.aggregate import WorkflowExecution
from micro_model_agent.execution.domain.services import WorkflowEvaluationService
from micro_model_agent.execution.domain.value_objects import WorkflowTrace
from micro_model_agent.shared.domain.event_bus import EventBus
from micro_model_agent.shared.domain.value_objects import EvaluationResult


class DefaultWorkflowEvaluator:
    """Adapter implementing the WorkflowEvaluator port via WorkflowEvaluationService.

    Kept for backward compatibility; new code should inject WorkflowEvaluationService
    directly from ``execution.domain.services``.
    """

    def __init__(self) -> None:
        self._service = WorkflowEvaluationService()

    async def evaluate(self, trace: WorkflowTrace) -> EvaluationResult:
        """Delegate to WorkflowEvaluationService (domain layer)."""
        return self._service.evaluate(trace)


class TraceDatasetBuilder:
    """Thin adapter that delegates to ExecutionToDatasetTranslator (dataset ACL).

    Kept for backward compatibility; prefer using ExecutionToDatasetTranslator
    from ``dataset.infrastructure.acl`` directly.
    """

    def __init__(self) -> None:
        self._translator = ExecutionToDatasetTranslator()

    def build_example(
        self,
        trace: WorkflowTrace,
        label: DatasetLabel,
        *,
        kind: DatasetExampleKind = DatasetExampleKind.REPAIR,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetExample:
        """Delegate to ExecutionToDatasetTranslator (dataset infrastructure ACL)."""
        return self._translator.translate(trace, label, kind=kind, metadata=metadata)


class RunAgentWorkflow:
    """Run the coding agent, evaluate the trace, and optionally store a labeled example.

    Phase 8 orchestration pattern:
    1. Run the agent via the ``CodingWorkflowRunner`` port (returns a result with a trace).
    2. Reconstruct a ``WorkflowExecution`` aggregate from the trace.
    3. Record the evaluation result on the aggregate.
    4. Persist via ``agent.trace_store.save`` (backward-compat) AND pull/publish domain events.
    5. Optionally build and persist a labeled ``DatasetExample``.
    """

    def __init__(
        self,
        agent: CodingWorkflowRunner,
        evaluator: WorkflowEvaluator | None = None,
        dataset_store: DatasetExampleStore | None = None,
        dataset_builder: TraceDatasetBuilder | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.agent = agent
        self.evaluator = evaluator or DefaultWorkflowEvaluator()
        self.dataset_store = dataset_store
        self.dataset_builder = dataset_builder or TraceDatasetBuilder()
        self.event_bus = event_bus

    async def run(
        self,
        task: CodingAgentTask,
        *,
        label: DatasetLabel | None = None,
    ) -> CodingAgentResult:
        # 1. Execute the agent.
        result = await self.agent.run(task)

        # 2. Reconstruct aggregate from the returned snapshot.
        execution = WorkflowExecution.from_snapshot(result.trace)

        # 3. Score the trace and embed the evaluation summary in final_output.
        evaluation = await self.evaluator.evaluate(result.trace)
        execution.final_output = {
            **execution.final_output,
            "evaluation": {
                "passed": evaluation.passed,
                "summary": evaluation.summary,
                "score": evaluation.score,
                "details": evaluation.details,
            },
        }
        execution.updated_at = datetime.now(UTC)

        # 4. Persist (backward-compat path via trace_store) and publish events.
        trace = execution.to_snapshot()
        await self.agent.trace_store.save(trace)
        if self.event_bus is not None:
            await self.event_bus.publish_all(execution.pull_events())

        # 5. Optionally persist a labeled training example.
        if label and self.dataset_store:
            await self.dataset_store.save(
                self.dataset_builder.build_example(trace, label)
            )

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
