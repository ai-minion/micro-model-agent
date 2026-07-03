"""Local workflow trace storage.

Each trace is stored as a directory tree under the store's root:

    <root>/
      workflows.jsonl               # append-only index (no prompt/response blobs)
      <trace-id>/
        metadata.json               # trace-level: id, goal, status, run_metadata, timestamps
        <step-id>/                  # one folder per model turn
          metadata.json             # step-level: id, name, status, turn_index, tool_name
          request.txt               # raw prompt sent to the model
          response.txt              # raw model response
          workflow.json             # ordered event sequence for this turn

Saving the same trace ID again overwrites the per-trace files and appends a
new line to workflows.jsonl so the index always reflects the latest state.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from micro_model_agent.execution.domain.value_objects import (
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
        "run_metadata": trace.run_metadata,
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
        run_metadata=dict(record.get("run_metadata", {})),
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
    """Workflow trace store that writes a structured directory tree per trace.

    Layout::

        <root>/
          workflows.jsonl               # append-only index, no prompt/response blobs
          <trace-id>/
            metadata.json               # trace-level: id, goal, status, run_metadata, timestamps
            <step-id>/                  # one folder per model turn
              metadata.json             # step-level: id, name, status, turn_index, tool_name
              request.txt               # raw prompt sent to the model (if captured)
              response.txt              # raw model response (if captured)
              workflow.json             # ordered event sequence for this turn
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.root = self.path.parent

    async def save(self, trace: WorkflowTrace) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        trace_dir = self._trace_dir(str(trace.id))
        trace_dir.mkdir(parents=True, exist_ok=True)

        self._write_trace_artifacts(trace, trace_dir)

        index_record = self._index_record(trace)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(index_record, sort_keys=True))
            file.write("\n")

    async def get(self, trace_id: str) -> WorkflowTrace | None:
        meta_file = self._trace_dir(trace_id) / "metadata.json"
        if meta_file.exists():
            record = json.loads(meta_file.read_text(encoding="utf-8"))
            self._hydrate_steps(record, trace_id)
            return workflow_trace_from_record(record)

        if not self.path.exists():
            return None

        found: WorkflowTrace | None = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == trace_id:
                found = workflow_trace_from_record(record)
        return found

    async def list(self) -> list[WorkflowTrace]:
        """List the newest saved version of each trace in first-seen order."""

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
            # Prefer per-trace folder if available for the latest entry.
            meta_file = self._trace_dir(trace_id) / "metadata.json"
            if meta_file.exists():
                full_record = json.loads(meta_file.read_text(encoding="utf-8"))
                self._hydrate_steps(full_record, trace_id)
                traces_by_id[trace_id] = workflow_trace_from_record(full_record)
            else:
                traces_by_id[trace_id] = workflow_trace_from_record(record)
        return [traces_by_id[trace_id] for trace_id in order]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trace_dir(self, trace_id: str) -> Path:
        return self.root / trace_id

    def _write_trace_artifacts(self, trace: WorkflowTrace, trace_dir: Path) -> None:
        """Write (or overwrite) all per-trace artifact files for one save."""

        record = workflow_trace_to_record(trace)

        # Trace-level metadata — everything except the bulky steps list.
        metadata: dict[str, Any] = {
            "id": record["id"],
            "goal": record["goal"],
            "status": record["status"],
            "run_metadata": record.get("run_metadata", {}),
            "final_output": record.get("final_output", {}),
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "steps": [],  # filled with lightweight step summaries below
        }

        for turn_index, step in enumerate(record.get("steps", [])):
            if not isinstance(step, dict):
                continue
            step_id = str(step["id"])
            step_dir = trace_dir / step_id
            step_dir.mkdir(parents=True, exist_ok=True)

            output = dict(step.get("output") or {})
            prompt = output.pop("prompt", None)
            raw_response = output.pop("raw_response", None)

            # request.txt / response.txt
            if isinstance(prompt, str):
                (step_dir / "request.txt").write_text(prompt, encoding="utf-8")
            if isinstance(raw_response, str):
                (step_dir / "response.txt").write_text(raw_response, encoding="utf-8")

            # workflow.json — ordered event sequence for this turn
            workflow_events = _build_workflow_events(
                step_name=step.get("name", ""),
                output=output,
                prompt=prompt,
                raw_response=raw_response,
                tool_call=step.get("tool_call"),
                tool_result=step.get("tool_result"),
            )
            (step_dir / "workflow.json").write_text(
                json.dumps({"events": workflow_events}, sort_keys=True, indent=2),
                encoding="utf-8",
            )

            # step metadata.json
            step_meta: dict[str, Any] = {
                "id": step_id,
                "name": step.get("name"),
                "status": step.get("status"),
                "turn_index": turn_index,
            }
            tool_call = step.get("tool_call")
            if isinstance(tool_call, dict):
                step_meta["tool_name"] = tool_call.get("tool_name")
                step_meta["tool_call_id"] = tool_call.get("id")
            (step_dir / "metadata.json").write_text(
                json.dumps(step_meta, sort_keys=True, indent=2),
                encoding="utf-8",
            )

            # Collect lightweight step summary for trace metadata.json
            summary: dict[str, Any] = {
                "id": step_id,
                "name": step.get("name"),
                "status": step.get("status"),
                "turn_index": turn_index,
            }
            if isinstance(tool_call, dict):
                summary["tool_name"] = tool_call.get("tool_name")
            metadata["steps"].append(summary)

        # Write full record (with lightweight steps) as the canonical source.
        full_meta = {**metadata, "steps": record.get("steps", [])}
        _strip_prompt_fields(full_meta)
        (trace_dir / "metadata.json").write_text(
            json.dumps(full_meta, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def _index_record(self, trace: WorkflowTrace) -> dict[str, Any]:
        """Build a lean JSONL index line: no prompt/response blobs."""

        record = workflow_trace_to_record(trace)
        _strip_prompt_fields(record)
        return record

    def _hydrate_steps(self, record: dict[str, Any], trace_id: str) -> None:
        """Re-inject prompt and raw_response from sidecar files into a record."""

        for step in record.get("steps", []):
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id", ""))
            if not step_id:
                continue
            step_dir = self._trace_dir(trace_id) / step_id
            request_file = step_dir / "request.txt"
            response_file = step_dir / "response.txt"
            output = step.setdefault("output", {})
            if not isinstance(output, dict):
                continue
            if request_file.exists():
                output["prompt"] = request_file.read_text(encoding="utf-8")
            if response_file.exists():
                output["raw_response"] = response_file.read_text(encoding="utf-8")


def _strip_prompt_fields(record: dict[str, Any]) -> None:
    """Remove prompt/raw_response blobs from all step outputs in-place."""

    for step in record.get("steps", []):
        if not isinstance(step, dict):
            continue
        output = step.get("output")
        if isinstance(output, dict):
            output.pop("prompt", None)
            output.pop("raw_response", None)


def _build_workflow_events(
    *,
    step_name: str,
    output: dict[str, Any],
    prompt: str | None,
    raw_response: str | None,
    tool_call: dict[str, Any] | None,
    tool_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build the ordered event list for one turn's workflow.json."""

    events: list[dict[str, Any]] = []

    if prompt is not None:
        events.append({"type": "prompt", "content": prompt})

    if raw_response is not None:
        model_event: dict[str, Any] = {"type": "model_response", "raw": raw_response}
        if tool_call:
            model_event["parsed_kind"] = "tool_call"
            if output.get("reason"):
                model_event["reason"] = output["reason"]
        elif step_name == "final_response":
            model_event["parsed_kind"] = "final_response"
            if output.get("response"):
                model_event["response"] = output["response"]
        else:
            model_event["parsed_kind"] = "error"
        if output.get("error"):
            model_event["error"] = output["error"]
        if output.get("missing_required_tools"):
            model_event["missing_required_tools"] = output["missing_required_tools"]
        events.append(model_event)

    if isinstance(tool_call, dict):
        events.append({
            "type": "tool_call",
            "tool_name": tool_call.get("tool_name"),
            "arguments": tool_call.get("arguments", {}),
        })

    if isinstance(tool_result, dict):
        events.append({
            "type": "tool_result",
            "ok": tool_result.get("ok"),
            "output": tool_result.get("output", {}),
            "error": tool_result.get("error"),
        })

    return events


class LocalWorkflowTraceReader:
    """Filesystem adapter for listing stored workflow traces."""

    def __init__(self, path: str | Path) -> None:
        self.store = JsonlTraceStore(path)

    async def list_workflow_traces(self) -> list[WorkflowTrace]:
        """List stored workflow traces from a JSONL trace store."""

        return await self.store.list()
