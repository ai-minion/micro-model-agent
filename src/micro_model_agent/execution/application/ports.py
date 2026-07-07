"""Execution context application ports.

Ports are abstract interfaces (Protocols) that the execution application layer
depends on.  Concrete adapters live in ``execution/infrastructure/``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from micro_model_agent.execution.domain.value_objects import (
    ToolCall,
    ToolResult,
    WorkflowTrace,
)
from micro_model_agent.shared.domain.value_objects import EvaluationResult


@dataclass(frozen=True, slots=True)
class CodingAgentTask:
    """Input for a coding workflow runner."""

    goal: str
    dry_run: bool = True
    require_approval: bool = True
    expected_changed_files: list[str] = field(default_factory=list)
    verification_command_name: str | None = None
    semantic_intent: str = "code"
    semantic_limit: int = 5


@dataclass(frozen=True, slots=True)
class CodingAgentResult:
    """Structured result returned by a coding workflow runner."""

    trace_id: UUID
    ok: bool
    summary: str
    patch_applied: bool
    verification_passed: bool | None
    changed_files: list[str]
    trace: WorkflowTrace


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class ModelProvider(Protocol):
    """Anything that can turn a prompt into model text."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a model completion from a chat messages list."""


class ToolExecutor(Protocol):
    """Anything that can run a typed tool call and return a typed result."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a typed tool call."""


class WorkflowEvaluator(Protocol):
    """Evaluator for judging whether a finished workflow succeeded."""

    async def evaluate(self, trace: WorkflowTrace) -> EvaluationResult:
        """Evaluate a completed workflow."""


class TraceStore(Protocol):
    """Persistence boundary for workflow traces."""

    async def save(self, trace: WorkflowTrace) -> None:
        """Persist a workflow trace."""

    async def get(self, trace_id: str) -> WorkflowTrace | None:
        """Load a workflow trace by id."""


class WorkflowTraceReader(Protocol):
    """Loads stored workflow traces for application workflows."""

    async def list_workflow_traces(self) -> list[WorkflowTrace]:
        """List stored workflow traces."""


class CodingWorkflowRunner(Protocol):
    """Application port for an implementation that can run a coding workflow."""

    trace_store: TraceStore

    async def run(self, task: CodingAgentTask) -> CodingAgentResult:
        """Run a coding workflow task."""
