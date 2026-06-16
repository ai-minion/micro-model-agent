"""Local JSONL workflow trace storage.

Workflow traces are append-only audit records. Saving the same trace ID again
adds a newer line, and loading returns the last matching record.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from micro_model_agent.domain.contracts import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)


def workflow_trace_to_record(trace: WorkflowTrace) -> dict[str, Any]:
    """Convert a workflow trace into a JSON-serializable record."""

    # Nested dataclasses are converted piece by piece so json.dumps only sees
    # dictionaries, lists, strings, booleans, and numbers.
    return {
        "id": str(trace.id),
        "goal": trace.goal,
        "status": trace.status.value,
        "steps": [_workflow_step_to_record(step) for step in trace.steps],
        "final_output": trace.final_output,
        "created_at": trace.created_at.isoformat(),
        "updated_at": trace.updated_at.isoformat(),
    }


def workflow_trace_from_record(record: dict[str, Any]) -> WorkflowTrace:
    """Convert a JSON record into a workflow trace."""

    # Rehydrate enums, UUIDs, and datetimes from the strings stored in JSON.
    return WorkflowTrace(
        id=UUID(record["id"]),
        goal=record["goal"],
        status=WorkflowStatus(record["status"]),
        steps=[_workflow_step_from_record(step) for step in record.get("steps", [])],
        final_output=dict(record.get("final_output", {})),
        created_at=datetime.fromisoformat(record["created_at"]),
        updated_at=datetime.fromisoformat(record["updated_at"]),
    )


def _workflow_step_to_record(step: WorkflowStep) -> dict[str, Any]:
    """Convert one workflow step into JSON-friendly primitives."""

    return {
        "id": str(step.id),
        "name": step.name,
        "status": step.status.value,
        "tool_call": _tool_call_to_record(step.tool_call) if step.tool_call else None,
        "tool_result": _tool_result_to_record(step.tool_result) if step.tool_result else None,
        "output": step.output,
    }


def _workflow_step_from_record(record: dict[str, Any]) -> WorkflowStep:
    """Rebuild one workflow step from a stored record."""

    tool_call = record.get("tool_call")
    tool_result = record.get("tool_result")
    return WorkflowStep(
        id=UUID(record["id"]),
        name=record["name"],
        status=WorkflowStatus(record["status"]),
        tool_call=_tool_call_from_record(tool_call) if tool_call else None,
        tool_result=_tool_result_from_record(tool_result) if tool_result else None,
        output=dict(record.get("output", {})),
    )


def _tool_call_to_record(tool_call: ToolCall) -> dict[str, Any]:
    """Convert a tool call to a record nested inside a workflow step."""

    return {
        "id": str(tool_call.id),
        "tool_name": tool_call.tool_name,
        "arguments": tool_call.arguments,
        "created_at": tool_call.created_at.isoformat(),
    }


def _tool_call_from_record(record: dict[str, Any]) -> ToolCall:
    """Rebuild a ToolCall from a stored nested record."""

    return ToolCall(
        id=UUID(record["id"]),
        tool_name=record["tool_name"],
        arguments=dict(record["arguments"]),
        created_at=datetime.fromisoformat(record["created_at"]),
    )


def _tool_result_to_record(tool_result: ToolResult) -> dict[str, Any]:
    """Convert a tool result to a record nested inside a workflow step."""

    return {
        "tool_call_id": str(tool_result.tool_call_id),
        "tool_name": tool_result.tool_name,
        "ok": tool_result.ok,
        "output": tool_result.output,
        "error": tool_result.error,
    }


def _tool_result_from_record(record: dict[str, Any]) -> ToolResult:
    """Rebuild a ToolResult from a stored nested record."""

    return ToolResult(
        tool_call_id=UUID(record["tool_call_id"]),
        tool_name=record["tool_name"],
        ok=bool(record["ok"]),
        output=dict(record.get("output", {})),
        error=record.get("error"),
    )


class JsonlTraceStore:
    """Append-only JSONL workflow trace store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def save(self, trace: WorkflowTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append instead of overwriting so trace history is preserved.
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(workflow_trace_to_record(trace), sort_keys=True))
            file.write("\n")

    async def get(self, trace_id: str) -> WorkflowTrace | None:
        if not self.path.exists():
            return None

        found: WorkflowTrace | None = None
        # Keep scanning after a match so the newest saved version wins.
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == trace_id:
                found = workflow_trace_from_record(record)
        return found

    async def list(self) -> list[WorkflowTrace]:
        """List the newest saved version of each trace in file order."""

        if not self.path.exists():
            return []

        traces_by_id: dict[str, WorkflowTrace] = {}
        order: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            trace_id = str(record["id"])
            if trace_id not in traces_by_id:
                order.append(trace_id)
            traces_by_id[trace_id] = workflow_trace_from_record(record)
        return [traces_by_id[trace_id] for trace_id in order]
