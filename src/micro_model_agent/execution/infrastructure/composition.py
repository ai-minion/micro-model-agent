"""Execution context runtime composition — wires the event pipeline.

This module provides:
- ``build_event_pipeline`` — creates an ``InProcessEventBus`` with all
  cross-context handlers subscribed (execution→dataset→evaluation→promotion).
- ``build_coding_workflow`` — replaces the old ``build_static_coding_workflow``
  helper with event-bus support.
"""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.execution.infrastructure.coding_agent import CodingAgent
from micro_model_agent.dataset.application.ports import DatasetExampleStore
from micro_model_agent.dataset.application.event_handlers import OnWorkflowCompleted
from micro_model_agent.dataset.infrastructure.repository import JsonlDatasetRepository
from micro_model_agent.evaluation.application.event_handlers import OnArtifactProduced
from micro_model_agent.evaluation.domain.events import ThresholdBreached, ThresholdMet
from micro_model_agent.evaluation.infrastructure.repository import (
    JsonlEvaluationReportRepository,
)
from micro_model_agent.execution.application.workflows import RunAgentWorkflow
from micro_model_agent.execution.domain.events import WorkflowCompleted
from micro_model_agent.execution.infrastructure.models.fake import StaticModelProvider
from micro_model_agent.execution.infrastructure.persistence_runtime import (
    workflow_trace_store,
)
from micro_model_agent.execution.infrastructure.repository import JsonlWorkflowRepository
from micro_model_agent.promotion.application.event_handlers import OnThresholdEvent
from micro_model_agent.promotion.infrastructure.repository import (
    JsonlModelRegistryRepository,
)
from micro_model_agent.repository_ops.infrastructure.command_runner import (
    AllowedTestCommand,
)
from micro_model_agent.repository_ops.infrastructure.tools_runtime import (
    build_builtin_tool_executor,
)
from micro_model_agent.shared.domain.in_process_event_bus import InProcessEventBus
from micro_model_agent.training.domain.events import ArtifactProduced
from collections.abc import Mapping, Sequence

__all__ = [
    "build_coding_workflow",
    "build_event_pipeline",
]


def build_event_pipeline(
    *,
    trace_dir: Path,
    dataset_root: Path,
    evaluation_root: Path,
    promotion_root: Path,
) -> InProcessEventBus:
    """Create an InProcessEventBus with all cross-context handlers wired.

    Handler chain::

        WorkflowCompleted  → OnWorkflowCompleted  (execution → dataset)
        ArtifactProduced   → OnArtifactProduced   (training  → evaluation)
        ThresholdMet       → OnThresholdEvent     (evaluation → promotion)
        ThresholdBreached  → OnThresholdEvent     (evaluation → promotion)
    """

    bus = InProcessEventBus()

    workflow_repo = JsonlWorkflowRepository(trace_dir / "traces.jsonl")
    dataset_repo = JsonlDatasetRepository(dataset_root)
    evaluation_repo = JsonlEvaluationReportRepository(evaluation_root)
    registry_repo = JsonlModelRegistryRepository(promotion_root)

    on_completed = OnWorkflowCompleted(
        workflow_repo=workflow_repo,
        dataset_repo=dataset_repo,
    )
    on_artifact = OnArtifactProduced(report_repo=evaluation_repo)
    on_threshold = OnThresholdEvent(registry_repo=registry_repo)

    bus.subscribe(WorkflowCompleted, on_completed.handle)
    bus.subscribe(ArtifactProduced, on_artifact.handle)
    bus.subscribe(ThresholdMet, on_threshold.handle_met)
    bus.subscribe(ThresholdBreached, on_threshold.handle_breached)

    return bus


def build_coding_workflow(
    *,
    repository_root: str | Path,
    patch: str,
    allowed_commands: Mapping[str, AllowedTestCommand | Sequence[str]],
    dataset_store: DatasetExampleStore | None = None,
    event_bus: InProcessEventBus | None = None,
) -> RunAgentWorkflow:
    """Build the fixed coding workflow, optionally wired to the event bus."""

    repository = Path(repository_root)
    agent = CodingAgent(
        model_provider=StaticModelProvider(patch),
        tool_executor=build_builtin_tool_executor(repository, allowed_commands),
        trace_store=workflow_trace_store(repository),
    )
    return RunAgentWorkflow(
        agent=agent,
        dataset_store=dataset_store,
        event_bus=event_bus,
    )
