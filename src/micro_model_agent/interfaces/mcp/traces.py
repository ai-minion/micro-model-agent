"""Trace helpers for MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.interfaces.composition import (
    append_comparison_trace_event,
    review_comparison_trace_session,
    start_comparison_trace_session,
    stop_comparison_trace_session,
    workflow_trace_record,
)


async def read_trace(*, trace_id: str, repository_root: str = ".") -> dict[str, Any]:
    """Load a workflow trace captured by the local trace store."""

    trace = await workflow_trace_record(
        repository_root=repository_root,
        trace_id=trace_id,
    )
    if trace is None:
        return {"ok": False, "error": f"trace not found: {trace_id}"}
    return {"ok": True, "trace": trace}


async def start_comparison_trace(
    *,
    goal: str,
    repository_root: str = ".",
    task_repository_root: str | None = None,
    context: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a comparison trace session around a consumer/local-model task."""

    session = await start_comparison_trace_session(
        goal=goal,
        repository_root=task_repository_root or repository_root,
        comparison_repository_root=repository_root,
        context=context,
        metadata=metadata or {},
    )
    return {"ok": True, "session": session}


async def record_comparison_event(
    *,
    session_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    actor: str = "consumer",
    repository_root: str = ".",
) -> dict[str, Any]:
    """Append one event to a comparison trace session."""

    session = await append_comparison_event(
        repository_root=Path(repository_root),
        session_id=session_id,
        event_type=event_type,
        actor=actor,
        payload=payload or {},
    )
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    return {"ok": True, "session": session}


async def stop_comparison_trace(
    *,
    session_id: str,
    repository_root: str = ".",
    actual_summary: str = "",
    changed_files: list[str] | None = None,
    tests: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Stop a comparison trace and attach the consumer's actual result."""

    session = await stop_comparison_trace_session(
        repository_root=Path(repository_root),
        session_id=session_id,
        actual_result={
            "summary": actual_summary,
            "changed_files": changed_files or [],
            "tests": tests or [],
            "notes": notes,
        },
    )
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    return {"ok": True, "session": session}


async def review_comparison_trace(
    *,
    session_id: str,
    repository_root: str = ".",
    task_repository_root: str | None = None,
    local_model_quality: str = "unknown",
    local_model_notes: str = "",
    consumer_quality: str = "unknown",
    comparison_notes: str = "",
) -> dict[str, Any]:
    """Attach review details and return a compact comparison summary."""

    reviewed = await review_comparison_trace_session(
        repository_root=task_repository_root or repository_root,
        comparison_repository_root=repository_root,
        session_id=session_id,
        review={
            "local_model_quality": local_model_quality,
            "local_model_notes": local_model_notes,
            "consumer_quality": consumer_quality,
            "comparison_notes": comparison_notes,
        },
    )
    if reviewed is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    return {
        "ok": True,
        "session": reviewed["session"],
        "comparison": reviewed["comparison"],
    }


async def append_comparison_event(
    *,
    repository_root: Path,
    session_id: str,
    event_type: str,
    actor: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Append an event to a comparison trace session, if it exists."""

    return await append_comparison_trace_event(
        repository_root=repository_root,
        session_id=session_id,
        event_type=event_type,
        actor=actor,
        payload=payload,
    )
