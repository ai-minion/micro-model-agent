"""End-to-end tests for the model-driven tool loop."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from micro_model_agent.agents.tool_loop_agent import ToolLoopAgent, ToolLoopAgentTask
from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor
from micro_model_agent.infrastructure.tools.catalog import builtin_tool_prompt_schemas
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.infrastructure.trace_store import JsonlTraceStore


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


def _model_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def test_tool_loop_agent_runs_tool_calls_and_returns_final_response(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                    "reason": "inspect the current implementation",
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.write_patch",
                    "arguments": {
                        "patch": _patch(),
                        "dry_run": False,
                        "require_approval": False,
                        "expected_changed_files": ["app.py"],
                    },
                    "reason": "apply the requested return value change",
                }
            ),
            _model_response(
                {
                    "tool_name": "test.run",
                    "arguments": {"command_name": "python-check"},
                    "reason": "verify the changed behavior",
                }
            ),
            _model_response(
                {
                    "tool_name": "git.diff",
                    "arguments": {"paths": ["app.py"]},
                    "reason": "inspect the final diff",
                }
            ),
            _model_response(
                {
                    "final_response": "Changed app.py so value() returns 2; verification passed.",
                    "ok": True,
                }
            ),
        ]
    )
    trace_store = JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl")
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
    agent = ToolLoopAgent(model_provider=model, tool_executor=executor, trace_store=trace_store)

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Change value() in app.py to return 2",
                tool_schemas=builtin_tool_prompt_schemas(),
            )
        )
    )
    loaded_trace = asyncio.run(trace_store.get(str(result.trace_id)))

    assert result.ok is True
    assert result.tool_calls_made == 4
    assert result.response == "Changed app.py so value() returns 2; verification passed."
    assert "return 2" in (tmp_path / "app.py").read_text(encoding="utf-8")
    assert loaded_trace is not None
    assert [step.name for step in loaded_trace.steps] == [
        "tool_call_1",
        "tool_call_2",
        "tool_call_3",
        "tool_call_4",
        "final_response",
    ]
    assert [
        step.tool_call.tool_name
        for step in loaded_trace.steps
        if step.tool_call is not None
    ] == ["repo.read", "repo.write_patch", "test.run", "git.diff"]
    assert "arguments_schema" in model.prompts[0]
    assert "return 1" in model.prompts[1]
    assert "verified" in model.prompts[3]
    assert loaded_trace.final_output == {
        "ok": True,
        "response": result.response,
        "tool_calls_made": 4,
    }


def test_tool_loop_agent_requires_tool_before_final_response(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response({"final_response": "value() probably returns 1.", "ok": True}),
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response({"final_response": "value() returns 1.", "ok": True}),
        ]
    )
    trace_store = JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl")
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=trace_store,
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py and tell me what value() returns.",
                tool_schemas=builtin_tool_prompt_schemas(),
            )
        )
    )

    assert result.ok is True
    assert result.response == "value() returns 1."
    assert result.tool_calls_made == 1
    assert [step.name for step in result.trace.steps] == [
        "model_turn_1",
        "tool_call_1",
        "final_response",
    ]
    assert result.trace.steps[0].output["error"] == "final_response_before_tool_call"


def test_tool_loop_agent_can_capture_prompts_and_run_metadata(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response({"final_response": "value() returns 1.", "ok": True}),
        ]
    )
    trace_store = JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl")
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=trace_store,
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py and tell me what value() returns.",
                available_tools=("repo.read",),
                capture_prompts=True,
                run_metadata={"interface": "test", "schema_prompt": True},
            )
        )
    )
    loaded_trace = asyncio.run(trace_store.get(str(result.trace_id)))

    assert loaded_trace is not None
    assert loaded_trace.steps[0].output["prompt"].startswith("<|system|>")
    assert loaded_trace.final_output["run_metadata"] == {
        "interface": "test",
        "schema_prompt": True,
    }


def test_tool_loop_agent_uses_first_json_object_from_overeager_response(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            (
                _model_response(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "app.py"}]},
                    }
                )
                + "\n"
                + _model_response({"final_response": "value() returns 1.", "ok": True})
            ),
            _model_response({"final_response": "value() returns 1.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py and tell me what value() returns.",
                tool_schemas=builtin_tool_prompt_schemas(),
            )
        )
    )

    assert result.ok is True
    assert result.tool_calls_made == 1
    assert [step.name for step in result.trace.steps] == ["tool_call_1", "final_response"]


def test_tool_loop_agent_rejects_tool_calls_after_budget(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response({"final_response": "value() returns 1.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py and tell me what value() returns.",
                max_tool_calls=1,
            )
        )
    )

    assert result.ok is True
    assert result.tool_calls_made == 1
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == "tool_call_after_budget_exhausted"


def test_tool_loop_agent_final_result_fails_after_failed_tool(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "missing.py"}]},
                }
            ),
            _model_response({"final_response": "read succeeded", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read missing.py.",
                available_tools=("repo.read",),
            )
        )
    )

    assert result.ok is False
    assert result.trace.status.value == "failed"
    assert result.trace.steps[-1].output["ok"] is False


def test_tool_loop_agent_requires_specific_tools_before_final_response(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response({"final_response": "done", "ok": True}),
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"query": "value", "kind": "text"},
                }
            ),
            _model_response({"final_response": "done after search", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read and search app.py.",
                available_tools=("repo.read", "repo.search"),
                required_tools=("repo.search",),
            )
        )
    )

    assert result.ok is True
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "tool_call_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == "final_response_before_required_tools"
