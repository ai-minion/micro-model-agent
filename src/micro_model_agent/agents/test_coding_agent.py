"""End-to-end tests for the reference coding agent."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from micro_model_agent.agents.coding_agent import CodingAgent, CodingAgentTask
from micro_model_agent.execution.infrastructure.models.fake import StaticModelProvider
from micro_model_agent.execution.infrastructure.trace_store import JsonlTraceStore
from micro_model_agent.repository_ops.infrastructure.command_runner import AllowedTestCommand
from micro_model_agent.repository_ops.infrastructure.executor import BuiltinToolExecutor


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    _run_git(root, "init")
    _run_git(root, "config", "core.autocrlf", "false")
    _write_text(root / "app.py", "def value():\n    return 1\n")
    _run_git(root, "add", "app.py")


def _patch() -> str:
    return (
        "diff --git a/app.py b/app.py\n"
        "index 041b5f7..be082e7 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value():\n"
        "-    return 1\n"
        "+    return 2\n"
    )


def test_coding_agent_applies_patch_runs_verification_and_saves_trace(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    trace_store = JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl")
    executor = BuiltinToolExecutor(
        tmp_path,
        allowed_test_commands={
            "python-check": AllowedTestCommand(
                (
                    sys.executable,
                    "-c",
                    "from app import value; assert value() == 2; print('verified')",
                )
            )
        },
    )
    model = StaticModelProvider(_patch())
    agent = CodingAgent(model_provider=model, tool_executor=executor, trace_store=trace_store)

    result = asyncio.run(
        agent.run(
            CodingAgentTask(
                goal="Change value() to return 2",
                dry_run=False,
                require_approval=False,
                expected_changed_files=["app.py"],
                verification_command_name="python-check",
            )
        )
    )
    loaded_trace = asyncio.run(trace_store.get(str(result.trace_id)))

    assert result.ok is True
    assert result.patch_applied is True
    assert result.verification_passed is True
    assert result.changed_files == ["app.py"]
    assert "return 2" in (tmp_path / "app.py").read_text(encoding="utf-8")
    assert loaded_trace is not None
    assert [step.name for step in loaded_trace.steps] == [
        "retrieve_context",
        "generate_patch",
        "preview_or_apply_patch",
        "verify_changes",
        "summarize_diff",
    ]
    assert model.prompts


def test_coding_agent_dry_run_does_not_apply_patch(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    trace_store = JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl")
    executor = BuiltinToolExecutor(tmp_path, allowed_test_commands={})
    agent = CodingAgent(
        model_provider=StaticModelProvider(_patch()),
        tool_executor=executor,
        trace_store=trace_store,
    )

    result = asyncio.run(
        agent.run(
            CodingAgentTask(
                goal="Preview changing value() to return 2",
                dry_run=True,
                require_approval=True,
                expected_changed_files=["app.py"],
            )
        )
    )

    assert result.ok is True
    assert result.patch_applied is False
    assert "return 1" in (tmp_path / "app.py").read_text(encoding="utf-8")
