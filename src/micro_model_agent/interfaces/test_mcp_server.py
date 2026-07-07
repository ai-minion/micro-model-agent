"""Tests for the MCP server helpers."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from micro_model_agent.execution.application.tool_loop import run_profile_settings
from micro_model_agent.interfaces.mcp.tools.registry import register_mcp_tools
from micro_model_agent.interfaces.mcp.tools.run_loop import resolve_model_settings
from micro_model_agent.interfaces.mcp.workspace import path_from_user_input
from micro_model_agent.interfaces.mcp_server import (
    call_builtin_tool,
    create_mcp_server,
    init_repository,
    list_builtin_tools,
    read_trace,
    review_comparison_trace,
    run_agent_loop,
    start_comparison_trace,
    stop_comparison_trace,
)
from micro_model_agent.repository_ops.infrastructure.metadata import (
    initialize_repository,
    update_model_configuration,
)

DEFAULT_PUBLIC_TOOLS = {
    "micro_agent_init_workspace",
    "micro_agent_run_loop",
    "micro_agent_start_trace",
    "micro_agent_record_trace_event",
    "micro_agent_stop_trace",
    "micro_agent_review_trace",
}
DEFAULT_PROMPTS = {
    "compare_local_model_on_task",
    "collect_real_trace",
    "review_comparison_trace",
    "smoke_test_micro_agent",
}


def _model_response(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def test_mcp_server_exposes_only_loop_when_repository_is_initialized(tmp_path: Path) -> None:
    initialize_repository(tmp_path)

    server = create_mcp_server(repository_root=tmp_path)
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == DEFAULT_PUBLIC_TOOLS


def test_mcp_server_exposes_init_until_successful_initialization(tmp_path: Path) -> None:
    server = create_mcp_server(repository_root=tmp_path)
    initial_tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in initial_tools} == {
        *DEFAULT_PUBLIC_TOOLS,
        "micro_agent_init",
    }

    asyncio.run(server.call_tool("micro_agent_init", {}))
    final_tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in final_tools} == DEFAULT_PUBLIC_TOOLS
    assert (tmp_path / ".micro_model_agent" / "config.json").exists()


def test_mcp_server_debug_tools_are_opt_in(tmp_path: Path) -> None:
    initialize_repository(tmp_path)

    server = create_mcp_server(repository_root=tmp_path, expose_debug_tools=True)
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {
        *DEFAULT_PUBLIC_TOOLS,
        "micro_agent_builtin_tool",
        "micro_agent_read_trace",
        "micro_agent_list_builtin_tools",
    }


def test_mcp_server_advertises_tool_list_changes() -> None:
    server = create_mcp_server(expose_init_tool=False)

    options = server._mcp_server.create_initialization_options()

    assert options.capabilities.tools is not None
    assert options.capabilities.tools.listChanged is True


def test_mcp_server_exposes_workflow_prompts() -> None:
    server = create_mcp_server(expose_init_tool=False)

    prompts = asyncio.run(server.list_prompts())
    prompt_names = {prompt.name for prompt in prompts}
    rendered = asyncio.run(
        server.get_prompt(
            "compare_local_model_on_task",
            {
                "goal": "Create a README",
                "context": "Use the empty test workspace.",
            },
        )
    )

    assert prompt_names == DEFAULT_PROMPTS
    assert "micro_agent_start_trace" in str(rendered.messages)
    assert "Qwen2.5-Coder-7B-Instruct" in str(rendered.messages)


def test_mcp_tools_default_to_server_repository_root(tmp_path: Path) -> None:
    server = create_mcp_server(repository_root=tmp_path, expose_init_tool=False)

    started = asyncio.run(
        server.call_tool(
            "micro_agent_start_trace",
            {"goal": "Use the server default root."},
        )
    )
    _, data = started

    assert data["ok"] is True  # type: ignore[index]
    assert data["session"]["repository_root"] == str(tmp_path)  # type: ignore[index]


def test_mcp_workspace_id_selects_registered_workspace(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    workspace_root = tmp_path / "workspace"
    server = create_mcp_server(repository_root=registry_root, expose_init_tool=False)

    created = asyncio.run(
        server.call_tool(
            "micro_agent_init_workspace",
            {
                "path": str(workspace_root),
                "name": "chat-workspace",
            },
        )
    )
    _, created_data = created
    workspace_id = created_data["workspace"]["id"]  # type: ignore[index]  # type: ignore[index]
    started = asyncio.run(
        server.call_tool(
            "micro_agent_start_trace",
            {
                "workspace_id": workspace_id,
                "goal": "Use the registered workspace.",
            },
        )
    )
    _, started_data = started

    assert created_data["ok"] is True  # type: ignore[index]
    assert (workspace_root / ".micro_model_agent" / "config.json").exists()
    assert started_data["session"]["repository_root"] == str(workspace_root.resolve())  # type: ignore[index]


def test_path_from_user_input_maps_windows_paths_to_wsl_mounts(tmp_path: Path) -> None:
    mount_root = tmp_path / "mnt"

    assert path_from_user_input(
        "C:/Users/Rodger/workspace5",
        wsl_mount_root=mount_root,
    ) == mount_root / "c" / "Users" / "Rodger" / "workspace5"
    assert path_from_user_input(
        r"D:\Projects\code\workspace",
        wsl_mount_root=mount_root,
    ) == mount_root / "d" / "Projects" / "code" / "workspace"


def test_init_repository_is_idempotent(tmp_path: Path) -> None:
    first = init_repository(repository_root=str(tmp_path), default_model="qwen")
    second = init_repository(repository_root=str(tmp_path), default_model="ignored")

    assert first["ok"] is True
    assert first["already_initialized"] is False
    assert second["ok"] is True
    assert second["already_initialized"] is True
    assert second["config"]["model"]["default_model"] == "qwen"


def test_mcp_model_settings_use_selected_repository_config(tmp_path: Path) -> None:
    update = update_model_configuration(
        tmp_path,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=tmp_path / "training" / "runs" / "proof" / "adapter",
        selected_promotion={
            "artifact_id": "00000000-0000-4000-8000-000000000123",
            "artifact_name": "proof-adapter",
        },
    )

    settings = resolve_model_settings(
        repository_root=tmp_path,
        adapter_path=None,
        base_model=None,
        use_adapter=True,
    )

    assert update.ok is True
    assert settings == {
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "adapter_path": str(tmp_path / "training" / "runs" / "proof" / "adapter"),
        "selected_promotion_artifact_id": "00000000-0000-4000-8000-000000000123",
    }


def test_mcp_model_settings_prefer_explicit_args_over_selected_config(tmp_path: Path) -> None:
    update_model_configuration(
        tmp_path,
        base_model="configured-base",
        adapter_path="configured-adapter",
    )

    settings = resolve_model_settings(
        repository_root=tmp_path,
        adapter_path="explicit-adapter",
        base_model="explicit-base",
        use_adapter=True,
    )

    assert settings["base_model"] == "explicit-base"
    assert settings["adapter_path"] == "explicit-adapter"


def test_mcp_model_settings_can_disable_adapter_for_base_collection(
    tmp_path: Path,
) -> None:
    update_model_configuration(
        tmp_path,
        base_model="configured-base",
        adapter_path="configured-adapter",
    )

    settings = resolve_model_settings(
        repository_root=tmp_path,
        adapter_path=None,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        use_adapter=False,
    )

    assert settings["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert settings["adapter_path"] is None


def test_mcp_model_settings_use_base_model_when_adapter_is_unset(
    tmp_path: Path,
) -> None:
    settings = resolve_model_settings(
        repository_root=tmp_path,
        adapter_path=None,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        use_adapter=True,
    )

    assert settings["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert settings["adapter_path"] is None


def test_list_builtin_tools_marks_safe_defaults() -> None:
    result = list_builtin_tools()

    defaults = {
        tool["name"]
        for tool in result["tools"]
        if tool["default_enabled_for_mcp"]
    }
    assert {
        "repo.search",
        "repo.read",
        "repo.semantic_search",
        "repo.write_patch",
        "repo.write_files",
        "git.diff",
    } <= defaults


def test_call_builtin_tool_reads_repository_file(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nhello\n", encoding="utf-8")

    result = asyncio.run(
        call_builtin_tool(
            tool_name="repo.read",
            arguments={"files": [{"path": "README.md"}]},
            repository_root=str(tmp_path),
        )
    )

    assert result["ok"] is True
    assert result["tool_name"] == "repo.read"
    content = str(result["output"]["files"][0]["content"]).replace("\r\n", "\n")
    assert content == "# Demo\n\nhello\n"


def test_run_agent_loop_with_scripted_model_saves_trace(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nstatus: tiny\n", encoding="utf-8")

    result = asyncio.run(
        run_agent_loop(
            goal="Read README.md and summarize status.",
            repository_root=str(tmp_path),
            available_tools=["repo.read"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "README.md"}]},
                    }
                ),
                _model_response({"final_response": "status is tiny", "ok": True}),
            ],
        )
    )
    trace = asyncio.run(
        read_trace(trace_id=str(result["trace_id"]), repository_root=str(tmp_path))
    )

    assert result["ok"] is True
    assert result["response"] == "status is tiny"
    assert result["turns_used"] == 2
    assert result["tool_calls_made"] == 1
    assert result["loop_budget"] == {
        "max_turns": 4,
        "max_tool_calls": 1,
        "max_new_tokens": 350,
        "max_tool_result_prompt_chars": 2500,
    }
    assert trace["ok"] is True
    assert trace["trace"]["final_output"]["response"] == "status is tiny"


def test_run_agent_loop_applies_extended_profile_budget(tmp_path: Path) -> None:
    result = asyncio.run(
        run_agent_loop(
            goal="Create README.md.",
            repository_root=str(tmp_path),
            available_tools=["repo.write_files"],
            run_profile="extended",
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.write_files",
                        "arguments": {
                            "files": [{"path": "README.md", "content": "# Demo\n"}],
                        },
                    }
                ),
                _model_response({"final_response": "created README", "ok": True}),
            ],
        )
    )

    assert result["ok"] is True
    assert result["loop_budget"] == {
        "max_turns": 48,
        "max_tool_calls": None,
        "max_new_tokens": 32768,
        "max_tool_result_prompt_chars": 32000,
        "model_timeout_seconds": 600.0,
        "run_profile": "extended",
    }


def test_run_agent_loop_profile_preserves_explicit_smaller_tool_cap(
    tmp_path: Path,
) -> None:
    result = asyncio.run(
        run_agent_loop(
            goal="Search once and summarize.",
            repository_root=str(tmp_path),
            available_tools=["repo.search"],
            run_profile="quick",
            max_turns=5,
            max_tool_calls=3,
            scripted_responses=[
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
                _model_response(
                    {
                        "tool_name": "repo.search",
                        "arguments": {"glob": "**/*.txt", "limit": 1},
                    }
                ),
                _model_response(
                    {
                        "tool_name": "repo.search",
                        "arguments": {"glob": "**/*.json", "limit": 1},
                    }
                ),
                _model_response({"final_response": "searched once", "ok": True}),
            ],
        )
    )

    assert result["ok"] is True
    assert result["tool_calls_made"] == 3
    assert [step["name"] for step in result["steps"]] == [
        "tool_call_1",
        "tool_call_2",
        "tool_call_3",
        "model_turn_4",
        "final_response",
    ]
    assert result["steps"][3]["status"] == "failed"
    assert result["loop_budget"]["max_turns"] == 5
    assert result["loop_budget"]["max_tool_calls"] == 3
    assert result["loop_budget"]["run_profile"] == "quick"


def test_run_profile_settings_increase_by_tier() -> None:
    quick = run_profile_settings("quick")
    standard = run_profile_settings("standard")
    extended = run_profile_settings("extended")

    assert quick["max_tool_calls"] == 8
    assert quick["max_new_tokens"] == 512
    assert standard["max_tool_calls"] == 24
    assert standard["max_new_tokens"] == 8192
    assert extended["max_tool_calls"] is None
    assert extended["max_new_tokens"] == 32768
    assert quick["max_turns"] == 12
    assert standard["max_turns"] == 24
    assert extended["max_turns"] == 48
    assert quick["max_turns"] < standard["max_turns"] < extended["max_turns"]


def test_run_agent_loop_applies_write_files_when_patches_enabled(tmp_path: Path) -> None:
    result = asyncio.run(
        run_agent_loop(
            goal="Create a tiny project file.",
            repository_root=str(tmp_path),
            available_tools=["repo.write_files"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.write_files",
                        "arguments": {
                            "dry_run": False,
                            "files": [
                                {
                                    "path": "README.md",
                                    "content": "# Tiny\n\ncreated\n",
                                }
                            ],
                            "require_approval": True,
                        },
                    }
                ),
                _model_response({"final_response": "created README", "ok": True}),
            ],
        )
    )

    assert result["ok"] is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Tiny\n\ncreated\n"
    assert result["steps"][0]["tool_ok"] is True
    trace = asyncio.run(
        read_trace(trace_id=str(result["trace_id"]), repository_root=str(tmp_path))
    )
    tool_output = trace["trace"]["steps"][0]["tool_result"]["output"]
    assert tool_output["applied"] is True
    assert tool_output["dry_run"] is False


def test_run_agent_loop_normalizes_legacy_tool_names(tmp_path: Path) -> None:
    result = asyncio.run(
        run_agent_loop(
            goal="Create a tiny project file.",
            repository_root=str(tmp_path),
            available_tools=["shell", "apply_patch"],
            required_tools=["apply_patch"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.write_files",
                        "arguments": {
                            "files": [
                                {
                                    "path": "README.md",
                                    "content": "# Tiny\n\ncreated via alias\n",
                                }
                            ]
                        },
                    }
                ),
                _model_response({"final_response": "created README", "ok": True}),
            ],
        )
    )

    assert result["ok"] is True
    assert (
        tmp_path / "README.md"
    ).read_text(encoding="utf-8") == "# Tiny\n\ncreated via alias\n"
    trace = asyncio.run(
        read_trace(trace_id=str(result["trace_id"]), repository_root=str(tmp_path))
    )
    run_metadata = trace["trace"]["final_output"]["run_metadata"]
    assert run_metadata["available_tools"] == ["repo.write_files", "repo.write_patch"]
    assert run_metadata["required_tools"] == ["repo.write_files"]


def test_run_agent_loop_hides_git_diff_outside_git_repository(tmp_path: Path) -> None:
    result = asyncio.run(
        run_agent_loop(
            goal="Read README.md if it exists.",
            repository_root=str(tmp_path),
            available_tools=["repo.search", "git.diff"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.search",
                        "arguments": {"kind": "glob", "query": "README.md"},
                    }
                ),
                _model_response({"final_response": "No README.md found.", "ok": True}),
            ],
        )
    )

    trace = asyncio.run(
        read_trace(trace_id=str(result["trace_id"]), repository_root=str(tmp_path))
    )

    assert result["ok"] is True
    assert trace["trace"]["final_output"]["run_metadata"]["available_tools"] == [
        "repo.search"
    ]


def test_run_agent_loop_keeps_git_diff_inside_git_repository(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    result = asyncio.run(
        run_agent_loop(
            goal="Inspect git diff.",
            repository_root=str(tmp_path),
            available_tools=["git.diff"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "git.diff",
                        "arguments": {},
                    }
                ),
                _model_response({"final_response": "No diff.", "ok": True}),
            ],
        )
    )

    trace = asyncio.run(
        read_trace(trace_id=str(result["trace_id"]), repository_root=str(tmp_path))
    )

    assert result["ok"] is True
    assert trace["trace"]["final_output"]["run_metadata"]["available_tools"] == [
        "git.diff"
    ]


def test_run_agent_loop_adds_default_pytest_command_when_tests_allowed(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n",
        encoding="utf-8",
    )
    result = asyncio.run(
        run_agent_loop(
            goal="Run pytest.",
            repository_root=str(tmp_path),
            available_tools=["test.run"],
            allow_test_run=True,
            capture_prompts=True,
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "test.run",
                        "arguments": {"command_name": "pytest"},
                    }
                ),
                _model_response({"final_response": "pytest passed", "ok": True}),
            ],
        )
    )

    trace = asyncio.run(
        read_trace(trace_id=str(result["trace_id"]), repository_root=str(tmp_path))
    )
    first_prompt = trace["trace"]["steps"][0]["output"]["prompt"]
    assert isinstance(first_prompt, dict)
    user_message = next(
        message for message in first_prompt["messages"] if message.get("role") == "user"
    )
    payload = json.loads(user_message["content"])
    test_tool = next(
        tool for tool in first_prompt["tools"] if tool["function"]["name"] == "test.run"
    )

    assert result["ok"] is True
    assert trace["trace"]["final_output"]["run_metadata"]["allowed_test_commands"] == ["pytest"]
    assert "tool_schemas" not in payload
    assert test_tool["function"]["parameters"]["properties"]["command_name"]["enum"] == ["pytest"]


def test_run_agent_loop_uses_selected_adapter_config_with_scripted_smoke(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nstatus: promoted\n", encoding="utf-8")
    update_model_configuration(
        tmp_path,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=tmp_path / "training" / "runs" / "proof" / "adapter",
        selected_promotion={
            "artifact_id": "00000000-0000-4000-8000-000000000456",
            "artifact_name": "proof-adapter",
        },
    )

    result = asyncio.run(
        run_agent_loop(
            goal="Read README.md.",
            repository_root=str(tmp_path),
            available_tools=["repo.read", "test.run"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "README.md"}]},
                    }
                ),
                _model_response({"final_response": "status is promoted", "ok": True}),
            ],
        )
    )

    assert result["ok"] is True
    assert result["model"]["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert result["model"]["adapter_path"] == str(
        tmp_path / "training" / "runs" / "proof" / "adapter"
    )
    assert result["model"]["selected_promotion_artifact_id"] == (
        "00000000-0000-4000-8000-000000000456"
    )
    assert [step["tool_name"] for step in result["steps"] if step["tool_name"]] == ["repo.read"]


def test_comparison_trace_records_local_model_and_consumer_result(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nstatus: compare\n", encoding="utf-8")
    started = asyncio.run(
        start_comparison_trace(
            goal="Read README.md and summarize status.",
            repository_root=str(tmp_path),
            context="consumer will solve this too",
        )
    )
    session_id = started["session"]["id"]

    model_result = asyncio.run(
        run_agent_loop(
            goal="Read README.md and summarize status.",
            repository_root=str(tmp_path),
            available_tools=["repo.read"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "README.md"}]},
                    }
                ),
                _model_response({"final_response": "status is compare", "ok": True}),
            ],
            comparison_session_id=session_id,
        )
    )
    stopped = asyncio.run(
        stop_comparison_trace(
            session_id=session_id,
            repository_root=str(tmp_path),
            actual_summary="Codex read README.md and saw status compare.",
            changed_files=[],
            tests=[],
        )
    )
    reviewed = asyncio.run(
        review_comparison_trace(
            session_id=session_id,
            repository_root=str(tmp_path),
            local_model_quality="good",
            consumer_quality="good",
            comparison_notes="same answer",
        )
    )

    assert model_result["trace_id"] in stopped["session"]["local_trace_ids"]
    assert reviewed["comparison"]["local_model"][0]["response"] == "status is compare"
    assert reviewed["comparison"]["consumer_actual"]["summary"] == (
        "Codex read README.md and saw status compare."
    )
    assert reviewed["comparison"]["review"]["comparison_notes"] == "same answer"


def test_comparison_trace_can_use_central_store_for_workspace_task(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("# Demo\n\nstatus: central\n", encoding="utf-8")
    started = asyncio.run(
        start_comparison_trace(
            goal="Read README.md and summarize status.",
            repository_root=str(workspace_root),
            comparison_repository_root=str(registry_root),
            context="consumer will solve this too",
        )
    )
    session_id = started["session"]["id"]

    model_result = asyncio.run(
        run_agent_loop(
            goal="Read README.md and summarize status.",
            repository_root=str(workspace_root),
            comparison_repository_root=str(registry_root),
            available_tools=["repo.read"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "README.md"}]},
                    }
                ),
                _model_response({"final_response": "status is central", "ok": True}),
            ],
            comparison_session_id=session_id,
        )
    )
    stopped = asyncio.run(
        stop_comparison_trace(
            session_id=session_id,
            repository_root=str(workspace_root),
            comparison_repository_root=str(registry_root),
            actual_summary="Codex read README.md and saw status central.",
        )
    )
    central_workflow_trace = registry_root / ".micro_model_agent" / "traces" / "workflows.jsonl"
    reviewed = asyncio.run(
        review_comparison_trace(
            session_id=session_id,
            repository_root=str(workspace_root),
            comparison_repository_root=str(registry_root),
            local_model_quality="good",
            consumer_quality="good",
        )
    )

    assert (
        registry_root / ".micro_model_agent" / "traces" / "comparison_sessions.jsonl"
    ).exists()
    assert not (
        workspace_root / ".micro_model_agent" / "traces" / "comparison_sessions.jsonl"
    ).exists()
    assert central_workflow_trace.exists()
    assert not (
        workspace_root / ".micro_model_agent" / "traces" / "workflows.jsonl"
    ).exists()
    assert model_result["trace_id"] in stopped["session"]["local_trace_ids"]
    assert reviewed["comparison"]["local_model"][0]["response"] == "status is central"


def test_mcp_start_trace_uses_workspace_store_for_workspace_task(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    server = create_mcp_server(
        repository_root=registry_root,
        expose_init_tool=False,
        expose_debug_tools=False,
    )
    created = asyncio.run(
        server.call_tool(
            "micro_agent_init_workspace",
            {
                "path": str(workspace_root),
                "name": "workspace",
            },
        )
    )
    _, created_data = created
    workspace_id = created_data["workspace"]["id"]  # type: ignore[index]

    asyncio.run(
        server.call_tool(
            "micro_agent_start_trace",
            {
                "workspace_id": workspace_id,
                "goal": "Compare workspace trace storage.",
            },
        )
    )

    assert (
        workspace_root / ".micro_model_agent" / "traces" / "comparison_sessions.jsonl"
    ).exists()
    assert not (
        registry_root / ".micro_model_agent" / "traces" / "comparison_sessions.jsonl"
    ).exists()


def test_mcp_read_trace_uses_workspace_store_for_workspace_task(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("# Demo\n\nstatus: central\n", encoding="utf-8")
    server = create_mcp_server(
        repository_root=registry_root,
        expose_init_tool=False,
        expose_debug_tools=True,
    )
    created = asyncio.run(
        server.call_tool(
            "micro_agent_init_workspace",
            {
                "path": str(workspace_root),
                "name": "workspace",
            },
        )
    )
    _, created_data = created
    workspace_id = created_data["workspace"]["id"]  # type: ignore[index]
    ran_data = asyncio.run(
        run_agent_loop(
            goal="Read README.md and summarize status.",
            repository_root=str(workspace_root),
            comparison_repository_root=str(workspace_root),
            available_tools=["repo.read"],
            max_tool_calls=1,
            scripted_responses=[
                _model_response(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "README.md"}]},
                    }
                ),
                _model_response({"final_response": "status is central", "ok": True}),
            ],
        )
    )
    read = asyncio.run(
        server.call_tool(
            "micro_agent_read_trace",
            {
                "workspace_id": workspace_id,
                "trace_id": ran_data["trace_id"],
            },
        )
    )
    _, read_data = read

    assert read_data["ok"] is True  # type: ignore[index]
    assert read_data["trace"]["final_output"]["response"] == "status is central"  # type: ignore[index]
    assert (
        workspace_root / ".micro_model_agent" / "traces" / "workflows.jsonl"
    ).exists()
    assert not (
        registry_root / ".micro_model_agent" / "traces" / "workflows.jsonl"
    ).exists()


def test_mcp_run_loop_wires_on_turn_and_trace_root_to_handler(tmp_path: Path) -> None:
    """When micro_agent_run_loop runs through the server, handler wiring is project-scoped."""
    received: list[Any] = []

    async def mock_handler(**kwargs: Any) -> dict[str, Any]:
        received.append(kwargs)
        return {"ok": True, "response": "done"}

    server = FastMCP("test")
    register_mcp_tools(
        server,
        default_repository_root=str(tmp_path),
        expose_debug_tools=False,
        expose_init_tool=False,
        run_agent_loop_handler=mock_handler,
    )

    asyncio.run(server.call_tool("micro_agent_run_loop", {"goal": "test goal"}))

    assert len(received) == 1
    assert callable(received[0]["on_turn"])
    assert received[0]["comparison_repository_root"] == str(tmp_path)
