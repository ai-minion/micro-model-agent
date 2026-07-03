"""WorkflowExecution — execution context aggregate root.

The aggregate enforces all lifecycle invariants for a coding-agent workflow run
and accumulates domain events that can be pulled and published after persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from micro_model_agent.execution.domain.events import (
    StepAdded,
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowStarted,
)
from micro_model_agent.execution.domain.exceptions import InvalidTransitionError
from micro_model_agent.execution.domain.value_objects import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)
from micro_model_agent.shared.domain.entity import Entity


@dataclass(slots=True)
class WorkflowStepEntity:
    """Mutable entity representing one step in the execution lifecycle."""

    name: str
    id: UUID
    status: WorkflowStatus = WorkflowStatus.PENDING
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    output: dict[str, Any] = field(default_factory=dict)

    def to_value_object(self) -> WorkflowStep:
        """Return the frozen value-object snapshot of this step."""

        return WorkflowStep(
            id=self.id,
            name=self.name,
            status=self.status,
            tool_call=self.tool_call,
            tool_result=self.tool_result,
            output=self.output,
        )


class WorkflowExecution(Entity):
    """Aggregate root for a single coding-agent workflow execution.

    Invariants enforced:
    - ``start()`` only allowed from PENDING.
    - ``add_step()`` only allowed while RUNNING.
    - ``complete()`` / ``fail()`` only allowed while RUNNING (or PENDING for fail).
    - Once in a terminal state the aggregate is immutable.
    """

    def __init__(self, goal: str, id: UUID | None = None) -> None:
        super().__init__(id)
        self.goal = goal
        self.status: WorkflowStatus = WorkflowStatus.PENDING
        self._steps: list[WorkflowStepEntity] = []
        self.final_output: dict[str, Any] = {}
        self.created_at: datetime = datetime.now(UTC)
        self.updated_at: datetime = datetime.now(UTC)

    # ------------------------------------------------------------------
    # State-transition commands
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Transition from PENDING → RUNNING."""

        if self.status != WorkflowStatus.PENDING:
            raise InvalidTransitionError(
                f"Cannot start a {self.status.value!r} workflow"
            )
        self.status = WorkflowStatus.RUNNING
        self._events.append(
            WorkflowStarted(execution_id=self.id, goal=self.goal)
        )

    def add_step(self, step: WorkflowStepEntity) -> None:
        """Append a step; only allowed while RUNNING."""

        if self.status != WorkflowStatus.RUNNING:
            raise InvalidTransitionError(
                "Steps can only be added to a running workflow"
            )
        self._steps.append(step)
        self._events.append(
            StepAdded(
                execution_id=self.id,
                step_id=step.id,
                step_name=step.name,
            )
        )

    def complete(self, final_output: dict[str, Any]) -> None:
        """Transition from RUNNING → SUCCEEDED."""

        if self.status != WorkflowStatus.RUNNING:
            raise InvalidTransitionError(
                f"Cannot complete a {self.status.value!r} workflow"
            )
        self.final_output = final_output
        self.status = WorkflowStatus.SUCCEEDED
        self.updated_at = datetime.now(UTC)
        self._events.append(
            WorkflowCompleted(execution_id=self.id, output=final_output)
        )

    def fail(self, reason: str) -> None:
        """Transition from RUNNING or PENDING → FAILED."""

        if self.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PENDING):
            raise InvalidTransitionError(
                f"Cannot fail a {self.status.value!r} workflow"
            )
        self.status = WorkflowStatus.FAILED
        self.updated_at = datetime.now(UTC)
        self._events.append(
            WorkflowFailed(execution_id=self.id, reason=reason)
        )

    # ------------------------------------------------------------------
    # Read-model projection
    # ------------------------------------------------------------------

    def to_snapshot(self) -> WorkflowTrace:
        """Return a frozen WorkflowTrace snapshot for backward-compatible callers.

        Old code that expects a ``WorkflowTrace`` continues to work; it just
        receives an immutable snapshot rather than the live aggregate.
        """

        return WorkflowTrace(
            id=self.id,
            goal=self.goal,
            status=self.status,
            steps=[step.to_value_object() for step in self._steps],
            final_output=dict(self.final_output),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_snapshot(cls, trace: WorkflowTrace) -> WorkflowExecution:
        """Reconstruct an aggregate from a persisted WorkflowTrace snapshot."""

        execution = cls(goal=trace.goal, id=trace.id)
        execution.status = trace.status
        execution.final_output = dict(trace.final_output)
        execution.created_at = trace.created_at
        execution.updated_at = trace.updated_at
        for step in trace.steps:
            execution._steps.append(
                WorkflowStepEntity(
                    id=step.id,
                    name=step.name,
                    status=step.status,
                    tool_call=step.tool_call,
                    tool_result=step.tool_result,
                    output=dict(step.output),
                )
            )
        # Reconstructed aggregates start with no pending events.
        return execution
