"""Framework-independent domain contracts.

This module contains plain Python data containers for the core ideas in the
project: agents, tools, retrieval results, and workflow traces. The domain layer
does not import infrastructure code so these objects stay easy to test and reuse.
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


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Configuration that describes one kind of agent the framework can run."""

    name: str
    description: str
    default_model: str
    # default_factory creates a new list for each instance, avoiding shared
    # mutable defaults between AgentProfile objects.
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
    # IDs and timestamps are filled in automatically when a ToolCall is created.
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


@dataclass(frozen=True, slots=True)
class RepositoryProfile:
    """Basic information about the repository an agent is working in."""

    root_path: str
    name: str
    default_branch: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Question sent to a retriever when the agent needs repository context."""

    query: str
    intent: str = "general"
    limit: int = 10
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """One item returned from retrieval, with a score and extra metadata."""

    source_type: str
    title: str
    content: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Collection of retrieved items for a single query."""

    query: str
    intent: str
    results: list[RetrievedItem] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SemanticSearchResult(RetrievalResult):
    """Structured semantic retrieval result."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Pass/fail result plus optional score and details for later inspection."""

    passed: bool
    summary: str
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
