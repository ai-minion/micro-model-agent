"""End-to-end tests for the model-driven tool loop."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from micro_model_agent.execution.domain.value_objects import WorkflowStatus
from micro_model_agent.execution.infrastructure.models.fake import ScriptedModelProvider
from micro_model_agent.execution.infrastructure.tool_loop_agent import (
    ToolLoopAgent,
    ToolLoopAgentTask,
)
from micro_model_agent.execution.infrastructure.trace_store import JsonlTraceStore
from micro_model_agent.repository_ops.infrastructure.catalog import builtin_tool_prompt_schemas
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


def _return_patch(old: int, new: int) -> str:
    return (
        "diff --git a/app.py b/app.py\n"
        "index 041b5f7..be082e7 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value():\n"
        f"-    return {old}\n"
        f"+    return {new}\n"
    )


def _model_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _user_payload_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    user_msg = next(m for m in messages if m["role"] == "user")
    loaded = json.loads(user_msg["content"])
    assert isinstance(loaded, dict)
    return loaded


class SlowModelProvider:
    """Model provider that blocks until cancelled or timed out."""

    def __init__(self) -> None:
        self.prompts: list[list[dict[str, Any]]] = []
        self.tools: list[list[dict[str, Any]] | None] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        self.prompts.append(messages)
        self.tools.append(tools)
        await asyncio.sleep(3600)
        return _model_response({"final_response": "too late", "ok": True})


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
    assert model.tools[0] is not None
    assert model.tools[0][0]["function"]["name"] == "repo.search"
    assert "tool_schemas" not in model.prompts[0][1]["content"]
    assert "return 1" in model.prompts[1][1]["content"]
    assert "verified" in model.prompts[3][1]["content"]
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
    trace_store = JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl")
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
    trace_store = JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl")
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
                run_metadata={"interface": "test", "schema_prompt": True},
            )
        )
    )
    loaded_trace = asyncio.run(trace_store.get(str(result.trace_id)))

    assert loaded_trace is not None
    # Prompts are always captured to per-step request.json files.
    assert loaded_trace.steps[0].output["prompt"][0]["role"] == "system"
    assert loaded_trace.final_output["run_metadata"] == {
        "interface": "test",
        "schema_prompt": True,
    }
    # run_metadata is also on the trace itself.
    assert loaded_trace.run_metadata == {"interface": "test", "schema_prompt": True}
    # Per-step files exist on disk.
    step_dir = (
        tmp_path / ".traces" / str(result.trace_id) / str(loaded_trace.steps[0].id)
    )
    assert (step_dir / "request.json").exists()
    assert (step_dir / "workflow.json").exists()


def test_tool_loop_agent_prompt_includes_prior_tool_call_arguments(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"glob": "**/*.py", "limit": 10},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"glob": "**/*.py", "limit": 10},
                }
            ),
            _model_response({"final_response": "searched twice", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Search Python files twice.",
                available_tools=("repo.search",),
                max_tool_calls=None,
            )
        )
    )
    second_prompt = _user_payload_from_messages(model.prompts[1])
    third_prompt = _user_payload_from_messages(model.prompts[2])

    assert result.ok is True
    assert second_prompt["tool_history"][0]["tool_name"] == "repo.search"
    assert "small fixed turn budget" in model.prompts[0][0]["content"]
    assert second_prompt["tool_history"][0]["arguments"] == {
        "glob": "**/*.py",
        "limit": 10,
    }
    assert [entry["arguments"] for entry in third_prompt["tool_history"]] == [
        {"glob": "**/*.py", "limit": 10},
        {"glob": "**/*.py", "limit": 10},
    ]


def test_tool_loop_agent_compacts_tool_history_outputs_near_budget(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response({"final_response": "read app.py", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py.",
                available_tools=("repo.read",),
                max_tool_result_prompt_chars=260,
            )
        )
    )
    second_prompt = _user_payload_from_messages(model.prompts[1])
    history = second_prompt["tool_history"]

    assert history[0]["tool_name"] == "repo.read"
    assert history[0]["arguments"] == {"files": [{"path": "app.py"}]}
    assert history[0]["output"]["file_count"] == 1
    assert history[0]["output"]["paths"] == ["app.py"]
    assert "content" not in json.dumps(history[0]["output"])


def test_tool_loop_agent_hints_to_write_files_after_empty_creation_searches(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"glob": "**/*.py", "limit": 1},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.semantic_search",
                    "arguments": {"query": "minimal FastAPI ecommerce", "limit": 1},
                }
            ),
            _model_response({"final_response": "no files created", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Create a Python e-commerce skeleton in this empty repository.",
                available_tools=("repo.search", "repo.semantic_search", "repo.write_files"),
                max_tool_calls=None,
            )
        )
    )
    third_prompt = _user_payload_from_messages(model.prompts[2])

    assert "orchestration_hints" in third_prompt
    assert "stop searching and use repo.write_files" in third_prompt["orchestration_hints"][0]


def test_tool_loop_agent_stops_dry_run_creation_discovery_after_required_search(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"glob": "**/*.py", "limit": 1},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"glob": "**/*.md", "limit": 1},
                }
            ),
            _model_response({"final_response": "I propose creating hello.txt.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal=(
                    "Search the empty workspace, then propose creating hello.txt. "
                    "Do not apply changes."
                ),
                available_tools=("repo.search",),
                required_tools=("repo.search",),
                max_tool_calls=3,
            )
        )
    )
    second_prompt = _user_payload_from_messages(model.prompts[1])

    assert result.ok is True
    assert result.tool_calls_made == 1
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == (
        "tool_call_after_discovery_sufficient"
    )
    assert second_prompt["loop_budget"]["final_response_only"] is True


def test_tool_loop_agent_hints_to_write_files_after_patch_validation_failure(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_patch",
                    "arguments": {
                        "patch": "---\n+++ README.md\n@@\n+# Demo\n",
                        "dry_run": False,
                        "require_approval": False,
                    },
                }
            ),
            _model_response({"final_response": "patch failed", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Create README.md.",
                available_tools=("repo.write_patch", "repo.write_files"),
            )
        )
    )
    second_prompt = _user_payload_from_messages(model.prompts[1])

    assert result.ok is False
    assert "orchestration_hints" in second_prompt
    assert "use repo.write_files" in second_prompt["orchestration_hints"][0]


def test_tool_loop_agent_hints_to_write_files_after_write_files_validation_failure(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "paths": {"README.md": "# Demo\n"},
                        "dry_run": False,
                        "require_approval": False,
                    },
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "files": [{"path": "README.md", "content": "# Demo\n"}],
                        "dry_run": False,
                        "require_approval": False,
                    },
                }
            ),
            _model_response({"final_response": "Created README.md.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Create README.md.",
                available_tools=("repo.write_files",),
                required_tools=("repo.write_files",),
            )
        )
    )
    second_payload = _user_payload_from_messages(model.prompts[1])

    assert result.ok is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    assert "orchestration_hints" in second_payload
    assert "files" in second_payload["orchestration_hints"][0]
    assert "path" in second_payload["orchestration_hints"][0]
    assert "content" in second_payload["orchestration_hints"][0]


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
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
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
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
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


def test_tool_loop_agent_allows_final_response_after_last_tool_turn(tmp_path: Path) -> None:
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
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py and tell me what value() returns.",
                max_tool_calls=1,
                max_turns=1,
            )
        )
    )

    assert result.ok is True
    assert result.response == "value() returns 1."
    assert [step.name for step in result.trace.steps] == ["tool_call_1", "final_response"]


def test_tool_loop_agent_allows_final_response_after_max_turn_tool_call(
    tmp_path: Path,
) -> None:
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
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py and tell me what value() returns.",
                max_tool_calls=8,
                max_turns=1,
            )
        )
    )

    assert result.ok is True
    assert result.turns_used == 2
    assert [step.name for step in result.trace.steps] == ["tool_call_1", "final_response"]
    first_payload = _user_payload_from_messages(model.prompts[0])
    final_payload = _user_payload_from_messages(model.prompts[1])
    assert first_payload["loop_budget"] == {
        "turn_number": 1,
        "max_turns": 1,
        "turns_remaining_including_current": 1,
        "tool_calls_made": 0,
        "max_tool_calls": 8,
        "tool_calls_remaining": 8,
        "final_response_only": False,
    }
    assert final_payload["available_tools"] == []
    assert final_payload["loop_budget"]["turn_number"] == 2
    assert final_payload["loop_budget"]["final_response_only"] is True


def test_tool_loop_agent_saves_running_trace_after_each_step(tmp_path: Path) -> None:
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
    trace_path = tmp_path / ".traces" / "workflows.jsonl"
    trace_store = JsonlTraceStore(trace_path)
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=trace_store,
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py and tell me what value() returns.",
                max_tool_calls=1,
            )
        )
    )

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert result.ok is True
    assert [record["status"] for record in records] == [
        "running",
        "running",
        "succeeded",
    ]
    assert records[1]["steps"][0]["name"] == "tool_call_1"


def test_tool_loop_agent_times_out_model_turn_and_saves_trace(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    trace_store = JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl")
    agent = ToolLoopAgent(
        model_provider=SlowModelProvider(),
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=trace_store,
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py.",
                model_timeout_seconds=0.01,
            )
        )
    )
    trace = asyncio.run(trace_store.get(str(result.trace_id)))

    assert result.ok is False
    assert result.response == "model completion timed out"
    assert trace is not None
    assert trace.status is WorkflowStatus.FAILED
    assert trace.final_output["error"] == "model_completion_timeout"
    assert trace.steps[0].output["error"] == "model_completion_timeout"


def test_tool_loop_agent_saves_cancelled_trace(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    trace_store = JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl")
    agent = ToolLoopAgent(
        model_provider=SlowModelProvider(),
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=trace_store,
    )

    async def run_and_cancel() -> tuple[str, WorkflowStatus]:
        task = asyncio.create_task(agent.run(ToolLoopAgentTask(goal="Read app.py.")))
        await asyncio.sleep(0)
        task.cancel()
        result = await task
        return str(result.trace_id), result.trace.status

    trace_id, status = asyncio.run(run_and_cancel())
    trace = asyncio.run(trace_store.get(trace_id))

    assert status is WorkflowStatus.CANCELLED
    assert trace is not None
    assert trace.status is WorkflowStatus.CANCELLED
    assert trace.final_output["error"] == "cancelled"


def test_tool_loop_agent_blocks_duplicate_successful_write_files(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "dry_run": False,
                        "require_approval": False,
                        "files": [
                            {
                                "path": "README.md",
                                "content": "# Demo\n",
                            }
                        ],
                    },
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "dry_run": False,
                        "require_approval": False,
                        "files": [
                            {
                                "path": "README.md",
                                "content": "# Demo\n",
                            }
                        ],
                    },
                }
            ),
            _model_response({"final_response": "Created README.md.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Create README.md.",
                available_tools=("repo.write_files",),
                required_tools=("repo.write_files",),
                max_tool_calls=8,
            )
        )
    )

    assert result.ok is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == "duplicate_successful_write"
    second_payload = _user_payload_from_messages(model.prompts[1])
    assert second_payload["tool_history"][0]["arguments"] == {
        "dry_run": False,
        "file_count": 1,
        "paths": ["README.md"],
    }
    assert "# Demo" not in json.dumps(second_payload["tool_history"])
    assert "README.md" not in second_payload["orchestration_hints"][0]
    assert "Do not repeat the same write" in second_payload["orchestration_hints"][0]
    third_payload = _user_payload_from_messages(model.prompts[2])
    assert "tool_results" not in third_payload


def test_tool_loop_agent_allows_distinct_patch_to_same_file(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_patch",
                    "arguments": {
                        "patch": _return_patch(1, 2),
                        "dry_run": False,
                        "require_approval": False,
                        "expected_changed_files": ["app.py"],
                    },
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.write_patch",
                    "arguments": {
                        "patch": _return_patch(2, 3),
                        "dry_run": False,
                        "require_approval": False,
                        "expected_changed_files": ["app.py"],
                    },
                }
            ),
            _model_response({"final_response": "Updated app.py twice.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Update app.py in two steps.",
                available_tools=("repo.write_patch",),
                max_tool_calls=8,
            )
        )
    )

    assert result.ok is True
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "def value():\n    return 3\n"
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "tool_call_2",
        "final_response",
    ]


def test_tool_loop_agent_keeps_latest_read_write_history_per_path(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "files": [
                            {"path": "README.md", "content": "# Old\n"},
                            {"path": "app.py", "content": "print('old')\n"},
                        ],
                    },
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "README.md"}]},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "files": [
                            {"path": "README.md", "content": "# New\n"},
                        ],
                    },
                }
            ),
            _model_response({"final_response": "Updated README.md.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Create and update README.md.",
                available_tools=("repo.write_files", "repo.read"),
                max_tool_calls=8,
            )
        )
    )
    final_payload = _user_payload_from_messages(model.prompts[3])

    history = final_payload["tool_history"]
    assert [entry["tool_name"] for entry in history] == [
        "repo.write_files",
        "repo.write_files",
    ]
    assert history[0]["arguments"]["paths"] == ["app.py"]
    assert history[0]["output"]["changed_files"] == ["app.py"]
    assert history[1]["arguments"]["paths"] == ["README.md"]
    assert history[1]["output"]["changed_files"] == ["README.md"]


def test_tool_loop_agent_allows_final_response_after_duplicate_write_at_max_turn(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {"files": [{"path": "README.md", "content": "# Demo\n"}]},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {"files": [{"path": "README.md", "content": "# Demo\n"}]},
                }
            ),
            _model_response({"final_response": "Created README.md.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Create README.md.",
                available_tools=("repo.write_files",),
                required_tools=("repo.write_files",),
                max_turns=2,
            )
        )
    )

    assert result.ok is True
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == "duplicate_successful_write"


def test_tool_loop_agent_treats_write_files_as_required_write_patch(
    tmp_path: Path,
) -> None:
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "dry_run": False,
                        "require_approval": False,
                        "files": [{"path": "README.md", "content": "# Demo\n"}],
                    },
                }
            ),
            _model_response({"final_response": "Created README.md.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Create README.md.",
                available_tools=("repo.write_patch", "repo.write_files"),
                required_tools=("repo.write_patch",),
            )
        )
    )

    assert result.ok is True
    assert result.response == "Created README.md."


def test_tool_loop_agent_requires_verification_after_repair_write(
    tmp_path: Path,
) -> None:
    _write_text(tmp_path / "app.py", "VALUE = 1\n")
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "dry_run": False,
                        "require_approval": False,
                        "files": [{"path": "app.py", "content": "VALUE = 2\n"}],
                    },
                }
            ),
            _model_response({"final_response": "Fixed the pytest failure.", "ok": True}),
            _model_response(
                {
                    "tool_name": "test.run",
                    "arguments": {"command_name": "python-check"},
                }
            ),
            _model_response({"final_response": "Fixed and verified.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(
            tmp_path,
            allowed_test_commands={
                "python-check": AllowedTestCommand(
                    (
                        sys.executable,
                        "-c",
                        "from pathlib import Path; "
                        "assert Path('app.py').read_text() == 'VALUE = 2\\n'",
                    )
                )
            },
        ),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Fix the pytest import error so tests run successfully.",
                available_tools=("repo.write_files", "test.run"),
            )
        )
    )

    assert result.ok is True
    assert result.response == "Fixed and verified."
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "tool_call_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == "final_response_before_verification"
    third_payload = _user_payload_from_messages(model.prompts[2])
    assert "no test.run has passed" in third_payload["orchestration_hints"][0]


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
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
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


def test_tool_loop_agent_final_result_succeeds_after_repaired_failed_test(
    tmp_path: Path,
) -> None:
    _write_text(tmp_path / "app.py", "VALUE = 1\n")
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "test.run",
                    "arguments": {"command_name": "python-check"},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "dry_run": False,
                        "require_approval": False,
                        "files": [{"path": "app.py", "content": "VALUE = 2\n"}],
                    },
                }
            ),
            _model_response(
                {
                    "tool_name": "test.run",
                    "arguments": {"command_name": "python-check"},
                }
            ),
            _model_response({"final_response": "Fixed and verified.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(
            tmp_path,
            allowed_test_commands={
                "python-check": AllowedTestCommand(
                    (
                        sys.executable,
                        "-c",
                        "from pathlib import Path; "
                        "assert Path('app.py').read_text() == 'VALUE = 2\\n'",
                    )
                )
            },
        ),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Fix the failing test.",
                available_tools=("test.run", "repo.write_files"),
            )
        )
    )

    assert result.ok is True
    assert result.trace.status is WorkflowStatus.SUCCEEDED
    assert result.response == "Fixed and verified."


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
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
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


def test_tool_loop_agent_fires_on_turn_at_start_of_each_turn(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                    "reason": "read file",
                }
            ),
            _model_response({"final_response": "value() returns 1.", "ok": True}),
        ]
    )
    trace_store = JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl")
    executor = BuiltinToolExecutor(tmp_path, allowed_test_commands={})
    agent = ToolLoopAgent(model_provider=model, tool_executor=executor, trace_store=trace_store)

    on_turn_calls: list[tuple[int, int, str]] = []

    async def on_turn(turn_number: int, max_turns: int, message: str) -> None:
        on_turn_calls.append((turn_number, max_turns, message))

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py",
                available_tools=("repo.read",),
                max_turns=4,
                on_turn=on_turn,
            )
        )
    )

    assert result.ok is True
    thinking_calls = [
        (t, m)
        for t, m, msg in on_turn_calls
        if msg.startswith("thinking") or msg == "finalizing"
    ]
    tool_calls_fired = [
        (t, m, msg)
        for t, m, msg in on_turn_calls
        if not (msg.startswith("thinking") or msg == "finalizing")
    ]
    # "thinking" / "thinking (N tool calls made)" fires once per turn
    assert thinking_calls == [(1, 4), (2, 4)]
    # tool label fires once, before the tool executes
    assert tool_calls_fired == [(1, 4, "repo.read: app.py")]


def test_tool_loop_agent_on_turn_fires_before_tool_execution(tmp_path: Path) -> None:
    order: list[str] = []

    class OrderTrackingExecutor:
        async def execute(self, tool_call: Any) -> Any:
            order.append(f"execute:{tool_call.tool_name}")
            from micro_model_agent.execution.domain.value_objects import ToolResult

            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                ok=True,
                output={"content": "x = 1"},
            )

    async def on_turn(turn_number: int, max_turns: int, message: str) -> None:
        order.append(f"on_turn:{message}")

    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response({"final_response": "done", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=OrderTrackingExecutor(),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read x",
                available_tools=("repo.read",),
                max_turns=4,
                on_turn=on_turn,
            )
        )
    )

    # on_turn with the tool name must be recorded strictly before execute
    tool_on_turn_idx = next(i for i, s in enumerate(order) if s.startswith("on_turn:repo.read"))
    execute_idx = order.index("execute:repo.read")
    assert tool_on_turn_idx < execute_idx


def test_tool_loop_agent_propagates_on_turn_exception(tmp_path: Path) -> None:
    """The agent does not swallow callback exceptions; only the MCP layer does."""
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    async def failing_on_turn(turn_number: int, max_turns: int, message: str) -> None:
        raise RuntimeError("callback failure")

    model = ScriptedModelProvider(
        [_model_response({"final_response": "done", "ok": True})]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    try:
        asyncio.run(
            agent.run(
                ToolLoopAgentTask(
                    goal="Read x",
                    available_tools=("repo.read",),
                    require_tool_call=False,
                    on_turn=failing_on_turn,
                )
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "callback failure"
    else:
        raise AssertionError("expected on_turn exception to propagate")


def test_tool_loop_agent_blocks_duplicate_repo_read(
    tmp_path: Path,
) -> None:
    """A repeated repo.read with identical arguments is blocked with a dedup message."""
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
            _model_response({"final_response": "The file defines value().", "ok": True}),
        ]
    )
    _init_repo(tmp_path)
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Describe app.py",
                available_tools=("repo.read",),
                max_tool_calls=8,
            )
        )
    )

    assert result.ok is True
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == "duplicate_read_or_search"
    # The policy message goes to transcript (tool_results), not as a tool result.
    assert result.trace.steps[1].tool_result is not None
    assert result.trace.steps[1].tool_result.ok is False
    assert result.trace.steps[1].tool_result.error == "duplicate_read_or_search"


def test_tool_loop_agent_blocks_duplicate_repo_search(
    tmp_path: Path,
) -> None:
    """A repeated repo.search with identical arguments is blocked with a dedup message."""
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"glob": "**/*.py"},
                }
            ),
            _model_response(
                {
                    "tool_name": "repo.search",
                    "arguments": {"glob": "**/*.py"},
                }
            ),
            _model_response({"final_response": "Found app.py.", "ok": True}),
        ]
    )
    _init_repo(tmp_path)
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="List Python files.",
                available_tools=("repo.search",),
                max_tool_calls=8,
            )
        )
    )

    assert result.ok is True
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "model_turn_2",
        "final_response",
    ]
    assert result.trace.steps[1].output["error"] == "duplicate_read_or_search"
    assert result.trace.steps[1].tool_result is not None
    assert result.trace.steps[1].tool_result.ok is False


def test_tool_loop_agent_allows_reread_after_write(
    tmp_path: Path,
) -> None:
    """repo.read is NOT blocked if a write occurred between the first and second read."""
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
                    "tool_name": "repo.write_files",
                    "arguments": {
                        "dry_run": False,
                        "require_approval": False,
                        "files": [{"path": "app.py", "content": "def value():\n    return 2\n"}],
                    },
                }
            ),
            # Second read after write — should be allowed.
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            _model_response({"final_response": "Done.", "ok": True}),
        ]
    )
    _init_repo(tmp_path)
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Update app.py and verify.",
                available_tools=("repo.read", "repo.write_files"),
                max_tool_calls=8,
            )
        )
    )

    assert result.ok is True
    # All 3 tool calls should succeed (no dedup block on the second read).
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "tool_call_2",
        "tool_call_3",
        "final_response",
    ]


def test_tool_loop_agent_final_result_ok_after_unavailable_tool_call(
    tmp_path: Path,
) -> None:
    """final_response ok=True is allowed even when the model called an unavailable tool."""
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
                    "tool_name": "repo.verify",  # not in available_tools
                    "arguments": {"path": "app.py"},
                }
            ),
            _model_response({"final_response": "Read app.py successfully.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Read app.py.",
                available_tools=("repo.read",),
                max_tool_calls=8,
            )
        )
    )

    assert result.ok is True
    assert result.response == "Read app.py successfully."
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "tool_call_2",
        "final_response",
    ]


def test_tool_loop_agent_blocks_repeated_unavailable_tool(
    tmp_path: Path,
) -> None:
    """Calling an unavailable tool a second time is dedup-blocked with a stronger message."""
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            # First call to unavailable tool — gets "not available" response.
            _model_response(
                {
                    "tool_name": "repo.verify",
                    "arguments": {"path": "app.py"},
                }
            ),
            # Second call to the same unavailable tool — should be dedup-blocked.
            _model_response(
                {
                    "tool_name": "repo.verify",
                    "arguments": {"path": "app.py"},
                }
            ),
            _model_response({"final_response": "Done.", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    result = asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Describe app.py",
                available_tools=("repo.read",),
                max_tool_calls=8,
            )
        )
    )

    assert result.ok is True
    assert [step.name for step in result.trace.steps] == [
        "tool_call_1",
        "tool_call_2",
        "model_turn_3",
        "final_response",
    ]
    assert result.trace.steps[2].output["error"] == "repeated_unavailable_tool"
    assert result.trace.steps[2].tool_result is not None
    assert result.trace.steps[2].tool_result.ok is False
    assert result.trace.steps[2].tool_result.error == "repeated_unavailable_tool"


def test_tool_loop_agent_dedup_read_does_not_displace_original_content(
    tmp_path: Path,
) -> None:
    """After a dedup-blocked read, the original file content must still be in the prompt.

    Regression: compact_path_tool_history was treating the dedup block as the
    'latest' entry for src/stats.py, dropping the real read result with file
    content. The model was then told 'you already have this' but couldn't see
    what 'this' was, causing it to loop forever.
    """
    _init_repo(tmp_path)
    model = ScriptedModelProvider(
        [
            # Turn 1: read the file
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            # Turn 2: try to read the same file again (will be dedup-blocked)
            _model_response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "app.py"}]},
                }
            ),
            # Turn 3: final_response (model finally produces it)
            _model_response({"final_response": "app.py defines value().", "ok": True}),
        ]
    )
    agent = ToolLoopAgent(
        model_provider=model,
        tool_executor=BuiltinToolExecutor(tmp_path, allowed_test_commands={}),
        trace_store=JsonlTraceStore(tmp_path / ".traces" / "workflows.jsonl"),
    )

    asyncio.run(
        agent.run(
            ToolLoopAgentTask(
                goal="Describe app.py",
                available_tools=("repo.read",),
                max_tool_calls=8,
                capture_prompts=True,
            )
        )
    )

    # The prompt for turn 3 (after the dedup block) should still contain the
    # file content from the ORIGINAL read — not just the dedup block.
    turn3_payload = _user_payload_from_messages(model.prompts[2])
    history = turn3_payload["tool_history"]

    # The original read (tool_call_1) must be present with file content.
    original_read = next(
        (e for e in history if e["tool_name"] == "repo.read" and "files" in e.get("output", {})),
        None,
    )
    assert original_read is not None, "original read result with file content was compacted away"
    assert any(
        "def value" in f.get("content", "")
        for f in original_read["output"]["files"]
    ), "file content missing from original read in prompt"

    # The dedup block must appear in tool_results (transcript-based active
    # feedback), NOT in tool_history, so it doesn't displace real content.
    tool_results = turn3_payload.get("tool_results", [])
    dedup_result = next(
        (
            r for r in tool_results
            if r.get("tool_name") == "orchestration_policy"
            and r.get("ok") is False
            and "already retrieved" in r.get("error", "").lower()
        ),
        None,
    )
    assert dedup_result is not None, "dedup policy message missing from tool_results"
