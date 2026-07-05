"""Tests for local workflow trace storage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.execution.domain.value_objects import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)
from micro_model_agent.execution.infrastructure.trace_store import JsonlTraceStore


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


def test_jsonl_trace_store_lists_latest_trace_versions(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    first = WorkflowTrace(
        goal="Read docs",
        status=WorkflowStatus.RUNNING,
        final_output={"ok": False},
    )
    updated = WorkflowTrace(
        id=first.id,
        goal=first.goal,
        status=WorkflowStatus.SUCCEEDED,
        final_output={"ok": True},
        created_at=first.created_at,
    )
    second = WorkflowTrace(goal="Search code", status=WorkflowStatus.SUCCEEDED)

    asyncio.run(store.save(first))
    asyncio.run(store.save(second))
    asyncio.run(store.save(updated))

    traces = asyncio.run(store.list())

    assert [trace.id for trace in traces] == [first.id, second.id]
    assert traces[0].status is WorkflowStatus.SUCCEEDED
    assert traces[0].final_output == {"ok": True}


def test_jsonl_trace_store_writes_trace_file_and_raw_io_sidecars(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / "traces" / "workflows.jsonl")
    trace = WorkflowTrace(
        goal="Read docs",
        status=WorkflowStatus.SUCCEEDED,
        steps=[
            WorkflowStep(
                name="final_response",
                status=WorkflowStatus.SUCCEEDED,
                output={
                    "prompt": [
                        {"role": "system", "content": "request"},
                        {"role": "user", "content": "{}"},
                    ],
                    "raw_response": '{"final_response":"done","ok":true}',
                    "response": "done",
                },
            )
        ],
        final_output={"ok": True, "response": "done"},
        run_metadata={"interface": "test"},
    )

    asyncio.run(store.save(trace))

    trace_dir = tmp_path / "traces" / str(trace.id)
    step_dir = trace_dir / str(trace.steps[0].id)
    metadata_file = trace_dir / "metadata.json"
    stored_meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    step_meta = json.loads((step_dir / "metadata.json").read_text(encoding="utf-8"))
    workflow = json.loads((step_dir / "workflow.json").read_text(encoding="utf-8"))
    index_record = json.loads((tmp_path / "traces" / "workflows.jsonl").read_text())

    assert metadata_file.exists()
    assert (step_dir / "request.txt").read_text(encoding="utf-8") == json.dumps(
        [{"role": "system", "content": "request"}, {"role": "user", "content": "{}"}]
    )
    assert (step_dir / "response.txt").read_text(encoding="utf-8") == (
        '{"final_response":"done","ok":true}'
    )

    # Metadata files must not embed raw blobs.
    assert "prompt" not in stored_meta["steps"][0].get("output", {})
    assert "raw_response" not in stored_meta["steps"][0].get("output", {})
    assert "prompt" not in index_record["steps"][0].get("output", {})
    assert "raw_response" not in index_record["steps"][0].get("output", {})

    # run_metadata is preserved.
    assert stored_meta["run_metadata"] == {"interface": "test"}
    assert index_record["run_metadata"] == {"interface": "test"}

    # Step metadata.
    assert step_meta["name"] == "final_response"
    assert step_meta["turn_index"] == 0

    # Workflow events.
    events = workflow["events"]
    assert events[0] == {
        "type": "prompt",
        "content": [{"role": "system", "content": "request"}, {"role": "user", "content": "{}"}],
    }
    assert events[1]["type"] == "model_response"
    assert events[1]["parsed_kind"] == "final_response"

    loaded = asyncio.run(store.get(str(trace.id)))

    assert loaded is not None
    assert loaded.steps[0].output["prompt"] == [
        {"role": "system", "content": "request"},
        {"role": "user", "content": "{}"},
    ]
    assert loaded.steps[0].output["raw_response"] == '{"final_response":"done","ok":true}'
    assert loaded.run_metadata == {"interface": "test"}
