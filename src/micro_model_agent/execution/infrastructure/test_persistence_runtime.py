"""Tests for persistence runtime composition helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.execution.domain.value_objects import (
    WorkflowStatus,
    WorkflowTrace,
)
from micro_model_agent.execution.infrastructure.persistence_runtime import (
    append_comparison_trace_event,
    comparison_trace_store,
    register_workspace_record,
    registered_workspace_path,
    review_comparison_trace_session,
    start_comparison_trace_session,
    stop_comparison_trace_session,
    trace_dir,
    workflow_trace_record,
    workflow_trace_store,
    workspace_registry,
)


def test_runtime_helpers_use_standard_trace_and_registry_paths(tmp_path: Path) -> None:
    assert trace_dir(tmp_path) == tmp_path / ".micro_model_agent" / "traces"
    assert workflow_trace_store(tmp_path).path == (
        tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"
    )
    assert comparison_trace_store(tmp_path).path == (
        tmp_path / ".micro_model_agent" / "traces" / "comparison_sessions.jsonl"
    )
    assert workspace_registry(tmp_path).path == (
        tmp_path / ".micro_model_agent" / "workspaces.jsonl"
    )


def test_workflow_trace_record_returns_json_ready_shape(tmp_path: Path) -> None:
    trace = WorkflowTrace(
        goal="Inspect status",
        status=WorkflowStatus.SUCCEEDED,
        final_output={"response": "ready", "tool_calls_made": 0},
    )

    asyncio.run(workflow_trace_store(tmp_path).save(trace))
    record = asyncio.run(
        workflow_trace_record(
            repository_root=tmp_path,
            trace_id=str(trace.id),
        )
    )

    assert record is not None
    assert record["id"] == str(trace.id)
    assert record["status"] == "succeeded"
    assert record["final_output"] == {"response": "ready", "tool_calls_made": 0}


def test_comparison_trace_runtime_returns_review_summary_across_roots(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    comparison_root = tmp_path / "comparison"
    trace = WorkflowTrace(
        goal="Inspect status",
        status=WorkflowStatus.SUCCEEDED,
        final_output={"response": "workspace ready", "tool_calls_made": 1},
    )
    asyncio.run(workflow_trace_store(workspace_root).save(trace))

    session = asyncio.run(
        start_comparison_trace_session(
            goal="Inspect status",
            repository_root=workspace_root,
            comparison_repository_root=comparison_root,
            context="shadow run",
            metadata={"source": "test"},
        )
    )
    updated = asyncio.run(
        append_comparison_trace_event(
            repository_root=comparison_root,
            session_id=session["id"],
            event_type="local_model_run",
            actor="micro_model_agent",
            payload={"trace_id": str(trace.id)},
        )
    )
    stopped = asyncio.run(
        stop_comparison_trace_session(
            repository_root=comparison_root,
            session_id=session["id"],
            actual_result={"summary": "human checked"},
        )
    )
    reviewed = asyncio.run(
        review_comparison_trace_session(
            repository_root=workspace_root,
            comparison_repository_root=comparison_root,
            session_id=session["id"],
            review={"quality": "good"},
        )
    )

    assert session["context"] == "shadow run"
    assert session["metadata"] == {"source": "test"}
    assert updated is not None
    assert updated["local_trace_ids"] == [str(trace.id)]
    assert stopped is not None
    assert stopped["status"] == "stopped"
    assert reviewed is not None
    assert reviewed["session"]["review"] == {"quality": "good"}
    assert reviewed["comparison"]["consumer_actual"] == {"summary": "human checked"}
    assert reviewed["comparison"]["local_model"] == [
        {
            "trace_id": str(trace.id),
            "status": "succeeded",
            "response": "workspace ready",
            "tool_calls_made": 1,
        }
    ]


def test_workspace_runtime_registers_and_resolves_paths(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    workspace_root = tmp_path / "workspace"

    record = asyncio.run(
        register_workspace_record(
            registry_root=registry_root,
            workspace_path=workspace_root,
            name="local",
            metadata={"kind": "fixture"},
        )
    )
    resolved = asyncio.run(
        registered_workspace_path(
            registry_root=registry_root,
            workspace_id=record["id"],
        )
    )

    assert record["path"] == str(workspace_root)
    assert record["name"] == "local"
    assert record["metadata"] == {"kind": "fixture"}
    assert resolved == workspace_root
