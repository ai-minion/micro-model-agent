"""Execution context value objects.

These are the canonical definitions for workflow-execution and tool-call types.
Canonical home: this module. ``domain/__init__.py`` re-exports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class WorkflowStatus(StrEnum):
    """Lifecycle status for workflow traces and steps."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Configuration that describes one kind of agent the framework can run."""

    name: str
    description: str
    default_model: str
    allowed_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Metadata about a model provider and the model it exposes."""

    provider: str
    model_name: str
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Description and JSON schemas for a tool the model is allowed to call."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    requires_approval: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One requested invocation of a named tool with JSON-style arguments."""

    tool_name: str
    arguments: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured output returned after a tool call finishes."""

    tool_call_id: UUID
    tool_name: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One step in an agent workflow, such as retrieval, patching, or testing."""

    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    output: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class WorkflowTrace:
    """Complete audit trail for a workflow run."""

    goal: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    steps: list[WorkflowStep] = field(default_factory=list)
    final_output: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
