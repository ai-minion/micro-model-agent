"""Domain events for the execution bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from micro_model_agent.shared.domain.domain_event import DomainEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowStarted(DomainEvent):
    """Raised when a WorkflowExecution transitions from PENDING to RUNNING."""

    execution_id: UUID
    goal: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepAdded(DomainEvent):
    """Raised when a WorkflowStep is appended to a running execution."""

    execution_id: UUID
    step_id: UUID
    step_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowCompleted(DomainEvent):
    """Raised when a WorkflowExecution transitions to SUCCEEDED."""

    execution_id: UUID
    output: dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowFailed(DomainEvent):
    """Raised when a WorkflowExecution transitions to FAILED."""

    execution_id: UUID
    reason: str
