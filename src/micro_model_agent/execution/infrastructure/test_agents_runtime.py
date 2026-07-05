"""Tests for concrete agent runtime composition helpers."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.execution.application.workflows import RunAgentWorkflow
from micro_model_agent.execution.infrastructure.agents_runtime import build_static_coding_workflow
from micro_model_agent.execution.infrastructure.trace_store import JsonlTraceStore
from micro_model_agent.repository_ops.infrastructure.command_runner import AllowedTestCommand


def test_agent_runtime_builds_static_coding_workflow(tmp_path: Path) -> None:
    workflow = build_static_coding_workflow(
        repository_root=tmp_path,
        patch="diff --git a/demo.txt b/demo.txt\n",
        allowed_commands={"unit": AllowedTestCommand(("python", "-m", "pytest"))},
    )

    assert isinstance(workflow, RunAgentWorkflow)
    assert isinstance(workflow.agent.trace_store, JsonlTraceStore)
    assert workflow.agent.trace_store.path == (
        tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"
    )
