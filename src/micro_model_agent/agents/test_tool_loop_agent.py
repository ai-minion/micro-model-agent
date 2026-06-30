"""End-to-end tests for the model-driven tool loop."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from micro_model_agent.agents.tool_loop_agent import ToolLoopAgent, ToolLoopAgentTask
from micro_model_agent.domain.contracts import WorkflowStatus
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


def _user_payload_from_prompt(prompt: str) -> dict[str, Any]:
    payload = prompt.split("<|user|>\n", 1)[1].split("\n<|assistant|>", 1)[0]
    loaded = json.loads(payload)
    assert isinstance(loaded, dict)
    return loaded


class SlowModelProvider:
    """Model provider that blocks until cancelled or timed out."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
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
    second_prompt = _user_payload_from_prompt(model.prompts[1])
    third_prompt = _user_payload_from_prompt(model.prompts[2])

    assert result.ok is True
    assert second_prompt["tool_history"][0]["tool_name"] == "repo.search"
    assert "small fixed turn budget" in model.prompts[0]
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
    second_prompt = _user_payload_from_prompt(model.prompts[1])
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
    third_prompt = _user_payload_from_prompt(model.prompts[2])

    assert "orchestration_hints" in third_prompt
    assert "stop searching and use repo.write_files" in third_prompt["orchestration_hints"][0]


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
    second_prompt = _user_payload_from_prompt(model.prompts[1])

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
    second_payload = _user_payload_from_prompt(model.prompts[1])

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
    first_payload = _user_payload_from_prompt(model.prompts[0])
    final_payload = _user_payload_from_prompt(model.prompts[1])
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
    second_prompt = model.prompts[1]
    second_payload = _user_payload_from_prompt(second_prompt)
    assert second_payload["tool_history"][0]["arguments"] == {
        "dry_run": False,
        "file_count": 1,
        "paths": ["README.md"],
    }
    assert "# Demo" not in json.dumps(second_payload["tool_history"])
    assert "README.md" not in second_payload["orchestration_hints"][0]
    assert "Do not repeat the same write" in second_payload["orchestration_hints"][0]
    third_payload = _user_payload_from_prompt(model.prompts[2])
    assert "tool_results" not in third_payload


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
    final_payload = _user_payload_from_prompt(model.prompts[3])

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
    third_payload = _user_payload_from_prompt(model.prompts[2])
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
