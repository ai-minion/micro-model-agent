"""Dataset application event handlers.

These handlers subscribe to execution-context events via the shared EventBus
and react by building DatasetExample records in the default Dataset.

Cross-context dependency direction:
  execution events (Published Language) → dataset ACL → dataset domain
"""

from __future__ import annotations

from micro_model_agent.dataset.domain.aggregate import Dataset
from micro_model_agent.dataset.domain.repository import DatasetRepository
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.acl import ExecutionToDatasetTranslator
from micro_model_agent.execution.domain.events import WorkflowCompleted
from micro_model_agent.execution.domain.repository import WorkflowRepository


class OnWorkflowCompleted:
    """Subscribe to ``WorkflowCompleted`` and build a DatasetExample.

    The handler loads the full ``WorkflowExecution`` from the execution
    repository so the ACL translator has access to step details (tool calls,
    retrieved context, etc.).  It then infers a default label from the outcome
    and appends the example to the "default" Dataset.
    """

    DEFAULT_DATASET_NAME = "default"

    def __init__(
        self,
        *,
        workflow_repo: WorkflowRepository,
        dataset_repo: DatasetRepository,
        translator: ExecutionToDatasetTranslator | None = None,
    ) -> None:
        self.workflow_repo = workflow_repo
        self.dataset_repo = dataset_repo
        self.translator = translator or ExecutionToDatasetTranslator()

    async def handle(self, event: WorkflowCompleted) -> None:
        """Build and store a DatasetExample from a completed workflow execution."""

        execution = await self.workflow_repo.get(event.execution_id)
        if execution is None:
            return  # execution not found in this repository; skip gracefully

        trace = execution.to_snapshot()
        ok = bool(event.output.get("ok"))
        label = DatasetLabel(
            outcome=OutcomeLabel.ACCEPTED if ok else OutcomeLabel.REJECTED,
            quality=QualityLabel.GOOD if ok else QualityLabel.BAD,
        )
        example = self.translator.translate(
            trace,
            label,
            kind=DatasetExampleKind.REPAIR,
        )

        dataset = await self.dataset_repo.find_by_name(self.DEFAULT_DATASET_NAME)
        if dataset is None:
            dataset = Dataset(name=self.DEFAULT_DATASET_NAME)
            dataset.add_example(example)
            await self.dataset_repo.add(dataset)
        else:
            dataset.add_example(example)
            await self.dataset_repo.save(dataset)
