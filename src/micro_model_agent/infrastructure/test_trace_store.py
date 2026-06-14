"""Tests for local workflow trace storage."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.domain.contracts import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)
from micro_model_agent.infrastructure.trace_store import JsonlTraceStore


def test_jsonl_trace_store_round_trips_trace(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    tool_call = ToolCall(tool_name="repo.search", arguments={"query": "WorkflowTrace"})
    trace = WorkflowTrace(
        goal="Find workflow trace code",
        status=WorkflowStatus.SUCCEEDED,
        steps=[
            WorkflowStep(
                name="search",
                status=WorkflowStatus.SUCCEEDED,
                tool_call=tool_call,
                tool_result=ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name="repo.search",
                    ok=True,
                    output={"matches": []},
                ),
            )
        ],
        final_output={"ok": True},
    )

    asyncio.run(store.save(trace))
    loaded = asyncio.run(store.get(str(trace.id)))

    assert loaded is not None
    assert loaded.id == trace.id
    assert loaded.steps[0].tool_call is not None
    assert loaded.steps[0].tool_call.tool_name == "repo.search"
    assert loaded.final_output == {"ok": True}
