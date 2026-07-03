"""Runtime helpers that compose persistence adapters into public records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from micro_model_agent.evaluation.infrastructure.comparison_trace import (
    ComparisonTraceSession,
    JsonlComparisonTraceStore,
    add_comparison_event,
    comparison_session_to_record,
    review_comparison_session,
    stop_comparison_session,
)
from micro_model_agent.execution.infrastructure.trace_store import (
    JsonlTraceStore,
    workflow_trace_to_record,
)
from micro_model_agent.repository_ops.infrastructure.workspace_registry import (
    JsonlWorkspaceRegistry,
    WorkspaceRecord,
    workspace_record_to_dict,
)

__all__ = [
    "DEFAULT_TRACE_DIR",
    "JsonlWorkspaceRegistry",
    "append_comparison_trace_event",
    "comparison_trace_store",
    "register_workspace_record",
    "registered_workspace_path",
    "review_comparison_trace_session",
    "start_comparison_trace_session",
    "stop_comparison_trace_session",
    "trace_dir",
    "workflow_trace_record",
    "workflow_trace_store",
    "workspace_registry",
]

DEFAULT_TRACE_DIR = Path(".traces")


def trace_dir(
    repository_root: str | Path,
    *,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> Path:
    """Return the centralized trace/log directory for one repository."""

    return Path(repository_root) / trace_dir_name


def workflow_trace_store(
    repository_root: str | Path,
    *,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> JsonlTraceStore:
    """Return the workflow trace store for one repository."""

    return JsonlTraceStore(
        trace_dir(repository_root, trace_dir_name=trace_dir_name) / "workflows.jsonl"
    )


def comparison_trace_store(
    repository_root: str | Path,
    *,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> JsonlComparisonTraceStore:
    """Return the comparison trace store for one repository."""

    return JsonlComparisonTraceStore(
        trace_dir(repository_root, trace_dir_name=trace_dir_name) / "comparison_sessions.jsonl"
    )


async def workflow_trace_record(
    *,
    repository_root: str | Path,
    trace_id: str,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> dict[str, Any] | None:
    """Load a workflow trace and return its JSON-ready record."""

    trace = await workflow_trace_store(
        repository_root,
        trace_dir_name=trace_dir_name,
    ).get(trace_id)
    if trace is None:
        return None
    return workflow_trace_to_record(trace)


async def start_comparison_trace_session(
    *,
    goal: str,
    repository_root: str | Path,
    comparison_repository_root: str | Path | None = None,
    context: str = "",
    metadata: dict[str, Any] | None = None,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> dict[str, Any]:
    """Create and persist a comparison trace session record."""

    repository = Path(repository_root)
    session = ComparisonTraceSession(
        goal=goal,
        repository_root=str(repository),
        context=context,
        metadata=metadata or {},
    )
    await comparison_trace_store(
        comparison_repository_root or repository,
        trace_dir_name=trace_dir_name,
    ).save(session)
    return comparison_session_to_record(session)


async def append_comparison_trace_event(
    *,
    repository_root: str | Path,
    session_id: str,
    event_type: str,
    actor: str,
    payload: dict[str, Any],
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> dict[str, Any] | None:
    """Append one event to a comparison trace session and return the record."""

    store = comparison_trace_store(repository_root, trace_dir_name=trace_dir_name)
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
    return comparison_session_to_record(updated)


async def stop_comparison_trace_session(
    *,
    repository_root: str | Path,
    session_id: str,
    actual_result: dict[str, Any],
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> dict[str, Any] | None:
    """Stop a comparison trace session and return the updated record."""

    store = comparison_trace_store(repository_root, trace_dir_name=trace_dir_name)
    session = await store.get(session_id)
    if session is None:
        return None
    stopped = stop_comparison_session(session, actual_result=actual_result)
    await store.save(stopped)
    return comparison_session_to_record(stopped)


async def review_comparison_trace_session(
    *,
    repository_root: str | Path,
    session_id: str,
    review: dict[str, Any],
    comparison_repository_root: str | Path | None = None,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> dict[str, Any] | None:
    """Attach review details and return the reviewed session plus trace summary."""

    repository = Path(repository_root)
    comparison_root = Path(comparison_repository_root or repository)
    store = comparison_trace_store(comparison_root, trace_dir_name=trace_dir_name)
    session = await store.get(session_id)
    if session is None:
        return None

    local_traces: list[dict[str, Any]] = []
    trace_roots = [comparison_root]
    session_root = Path(session.repository_root)
    if session_root not in trace_roots:
        trace_roots.append(session_root)
    for trace_id in session.local_trace_ids:
        for trace_root in trace_roots:
            trace_record = await workflow_trace_record(
                repository_root=trace_root,
                trace_id=trace_id,
                trace_dir_name=trace_dir_name,
            )
            if trace_record is not None:
                local_traces.append(trace_record)
                break

    reviewed = review_comparison_session(session, review=review)
    await store.save(reviewed)
    reviewed_record = comparison_session_to_record(reviewed)
    return {
        "session": reviewed_record,
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


def workspace_registry(registry_root: str | Path) -> JsonlWorkspaceRegistry:
    """Return the workspace registry for the server's default root."""

    return JsonlWorkspaceRegistry(Path(registry_root) / ".micro_model_agent" / "workspaces.jsonl")


async def register_workspace_record(
    *,
    registry_root: str | Path,
    workspace_path: str | Path,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a workspace record and return its JSON-ready representation."""

    record = WorkspaceRecord(
        path=str(workspace_path),
        name=name,
        metadata=dict(metadata or {}),
    )
    await workspace_registry(registry_root).save(record)
    return workspace_record_to_dict(record)


async def registered_workspace_path(
    *,
    registry_root: str | Path,
    workspace_id: str,
) -> Path | None:
    """Resolve a registered workspace id to its repository path."""

    workspace = await workspace_registry(registry_root).get(workspace_id)
    if workspace is None:
        return None
    return Path(workspace.path)
