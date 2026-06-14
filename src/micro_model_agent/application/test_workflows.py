"""Tests for workflow orchestration and trace-derived data capture."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from micro_model_agent.agents.coding_agent import CodingAgent, CodingAgentTask
from micro_model_agent.application.workflows import (
    RunAgentWorkflow,
    TraceDatasetBuilder,
    label_from_workflow_result,
)
from micro_model_agent.domain.datasets import OutcomeLabel, QualityLabel
from micro_model_agent.infrastructure.dataset_store import JsonlDatasetExampleStore
from micro_model_agent.infrastructure.fake_model_provider import StaticModelProvider
from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor
from micro_model_agent.infrastructure.trace_store import JsonlTraceStore


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)


def test_run_agent_workflow_evaluates_trace_and_stores_labeled_example(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    source = tmp_path / "hello.py"
    source.write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    patch = """diff --git a/hello.py b/hello.py
--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-print('hello')
+print('hello world')
"""
    trace_store = JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl")
    dataset_store = JsonlDatasetExampleStore(
        tmp_path / ".micro_model_agent" / "datasets" / "trace.jsonl"
    )
    agent = CodingAgent(
        model_provider=StaticModelProvider(patch),
        tool_executor=BuiltinToolExecutor(tmp_path, {}),
        trace_store=trace_store,
    )
    workflow = RunAgentWorkflow(agent=agent, dataset_store=dataset_store)

    initial = asyncio.run(
        agent.run(
            CodingAgentTask(
                goal="Update greeting",
                dry_run=True,
                expected_changed_files=["hello.py"],
            )
        )
    )
    label = label_from_workflow_result(initial)
    result = asyncio.run(
        workflow.run(
            CodingAgentTask(
                goal="Update greeting",
                dry_run=True,
                expected_changed_files=["hello.py"],
            ),
            label=label,
        )
    )
    examples = asyncio.run(dataset_store.list())

    assert result.ok is True
    assert result.trace.final_output["evaluation"]["passed"] is True
    assert examples[0].label.outcome is OutcomeLabel.ACCEPTED
    assert examples[0].label.quality is QualityLabel.GOOD
    assert examples[0].source.startswith("trace:")
    assert examples[0].target["changed_files"] == ["hello.py"]


def test_trace_dataset_builder_extracts_generated_patch(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    source = tmp_path / "hello.py"
    source.write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    patch = """diff --git a/hello.py b/hello.py
--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-print('hello')
+print('hi')
"""
    agent = CodingAgent(
        model_provider=StaticModelProvider(patch),
        tool_executor=BuiltinToolExecutor(tmp_path, {}),
        trace_store=JsonlTraceStore(tmp_path / "traces.jsonl"),
    )
    result = asyncio.run(
        agent.run(CodingAgentTask(goal="Say hi", dry_run=True, expected_changed_files=["hello.py"]))
    )
    example = TraceDatasetBuilder().build_example(result.trace, label_from_workflow_result(result))

    assert example.target["patch"] == patch
    assert example.metadata["trace_id"] == str(result.trace_id)
