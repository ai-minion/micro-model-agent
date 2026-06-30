"""Trace helpers for MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.infrastructure.comparison_trace import (
    ComparisonTraceSession,
    add_comparison_event,
    comparison_session_to_record,
    review_comparison_session,
    stop_comparison_session,
)
from micro_model_agent.infrastructure.composition import (
    comparison_trace_store as build_comparison_trace_store,
)
from micro_model_agent.infrastructure.composition import trace_dir as build_trace_dir
from micro_model_agent.infrastructure.composition import (
    workflow_trace_store as build_workflow_trace_store,
)
from micro_model_agent.infrastructure.trace_store import workflow_trace_to_record
from micro_model_agent.interfaces.mcp.compat import TRACE_DIR_NAME


async def read_trace(*, trace_id: str, repository_root: str = ".") -> dict[str, Any]:
    """Load a workflow trace captured by the local trace store."""

    trace_store = workflow_trace_store(Path(repository_root))
    trace = await trace_store.get(trace_id)
    if trace is None:
        return {"ok": False, "error": f"trace not found: {trace_id}"}
    return {"ok": True, "trace": workflow_trace_to_record(trace)}


async def start_comparison_trace(
    *,
    goal: str,
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
    context: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a comparison trace session around a consumer/local-model task."""

    repository = Path(repository_root)
    session = ComparisonTraceSession(
        goal=goal,
        repository_root=str(repository),
        context=context,
        metadata=metadata or {},
    )
    await comparison_trace_store(Path(comparison_repository_root or repository)).save(session)
    return {"ok": True, "session": comparison_session_to_record(session)}


async def record_comparison_event(
    *,
    session_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    actor: str = "consumer",
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
) -> dict[str, Any]:
    """Append one event to a comparison trace session."""

    session = await append_comparison_event(
        repository_root=Path(comparison_repository_root or repository_root),
        session_id=session_id,
        event_type=event_type,
        actor=actor,
        payload=payload or {},
    )
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    return {"ok": True, "session": comparison_session_to_record(session)}


async def stop_comparison_trace(
    *,
    session_id: str,
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
    actual_summary: str = "",
    changed_files: list[str] | None = None,
    tests: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Stop a comparison trace and attach the consumer's actual result."""

    store = comparison_trace_store(Path(comparison_repository_root or repository_root))
    session = await store.get(session_id)
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    stopped = stop_comparison_session(
        session,
        actual_result={
            "summary": actual_summary,
            "changed_files": changed_files or [],
            "tests": tests or [],
            "notes": notes,
        },
    )
    await store.save(stopped)
    return {"ok": True, "session": comparison_session_to_record(stopped)}


async def review_comparison_trace(
    *,
    session_id: str,
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
    local_model_quality: str = "unknown",
    local_model_notes: str = "",
    consumer_quality: str = "unknown",
    comparison_notes: str = "",
) -> dict[str, Any]:
    """Attach review details and return a compact comparison summary."""

    repository = Path(repository_root)
    store = comparison_trace_store(Path(comparison_repository_root or repository))
    session = await store.get(session_id)
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    local_traces = []
    comparison_root = Path(comparison_repository_root or repository)
    trace_roots = [comparison_root]
    session_root = Path(session.repository_root)
    if session_root not in trace_roots:
        trace_roots.append(session_root)
    for trace_id in session.local_trace_ids:
        trace = None
        for trace_root in trace_roots:
            trace_store = workflow_trace_store(trace_root)
            trace = await trace_store.get(trace_id)
            if trace is not None:
                break
        if trace is not None:
            local_traces.append(workflow_trace_to_record(trace))
    reviewed = review_comparison_session(
        session,
        review={
            "local_model_quality": local_model_quality,
            "local_model_notes": local_model_notes,
            "consumer_quality": consumer_quality,
            "comparison_notes": comparison_notes,
        },
    )
    await store.save(reviewed)
    return {
        "ok": True,
        "session": comparison_session_to_record(reviewed),
        "comparison": {
            "goal": reviewed.goal,
            "status": reviewed.status,
            "local_trace_ids": reviewed.local_trace_ids,
            "local_model": [
                {
                    "trace_id": trace["id"],
                    "status": trace["status"],
                    "response": trace["final_output"].get("response"),
                    "tool_calls_made": trace["final_output"].get("tool_calls_made"),
                }
                for trace in local_traces
            ],
            "consumer_actual": reviewed.actual_result,
            "review": reviewed.review,
        },
    }


def comparison_trace_store(repository_root: Path):
    """Return the comparison trace store for one repository."""

    return build_comparison_trace_store(repository_root, trace_dir_name=TRACE_DIR_NAME)


def workflow_trace_store(repository_root: Path):
    """Return the workflow trace store for one repository."""

    return build_workflow_trace_store(repository_root, trace_dir_name=TRACE_DIR_NAME)


def trace_dir(repository_root: Path) -> Path:
    """Return the centralized trace/log directory for one repository."""

    return build_trace_dir(repository_root, trace_dir_name=TRACE_DIR_NAME)


async def append_comparison_event(
    *,
    repository_root: Path,
    session_id: str,
    event_type: str,
    actor: str,
    payload: dict[str, Any],
) -> ComparisonTraceSession | None:
    """Append an event to a comparison trace session, if it exists."""

    store = comparison_trace_store(repository_root)
    session = await store.get(session_id)
    if session is None:
        return None
    updated = add_comparison_event(
        session,
        event_type=event_type,
        actor=actor,
        payload=payload,
    )
    await store.save(updated)
    return updated
