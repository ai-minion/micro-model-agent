"""Tests for the MCP server helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.infrastructure.repository_metadata import initialize_repository
from micro_model_agent.interfaces.mcp_server import (
    call_builtin_tool,
    create_mcp_server,
    init_repository,
    list_builtin_tools,
    read_trace,
    run_agent_loop,
)


def _model_response(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def test_mcp_server_exposes_only_loop_when_repository_is_initialized(tmp_path: Path) -> None:
    initialize_repository(tmp_path)

    server = create_mcp_server(repository_root=tmp_path)
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {"micro_agent_run_loop"}


def test_mcp_server_exposes_init_until_successful_initialization(tmp_path: Path) -> None:
    server = create_mcp_server(repository_root=tmp_path)
    initial_tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in initial_tools} == {
        "micro_agent_init",
        "micro_agent_run_loop",
    }

    asyncio.run(server.call_tool("micro_agent_init", {}))
    final_tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in final_tools} == {"micro_agent_run_loop"}
    assert (tmp_path / ".micro_model_agent" / "config.json").exists()


def test_mcp_server_debug_tools_are_opt_in(tmp_path: Path) -> None:
    initialize_repository(tmp_path)

    server = create_mcp_server(repository_root=tmp_path, expose_debug_tools=True)
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {
        "micro_agent_run_loop",
        "micro_agent_builtin_tool",
        "micro_agent_read_trace",
        "micro_agent_list_builtin_tools",
    }


def test_mcp_server_advertises_tool_list_changes() -> None:
    server = create_mcp_server(expose_init_tool=False)

    options = server._mcp_server.create_initialization_options()

    assert options.capabilities.tools is not None
    assert options.capabilities.tools.listChanged is True


def test_init_repository_is_idempotent(tmp_path: Path) -> None:
    first = init_repository(repository_root=str(tmp_path), default_model="qwen")
    second = init_repository(repository_root=str(tmp_path), default_model="ignored")

    assert first["ok"] is True
    assert first["already_initialized"] is False
    assert second["ok"] is True
    assert second["already_initialized"] is True
    assert second["config"]["model"]["default_model"] == "qwen"


def test_list_builtin_tools_marks_safe_defaults() -> None:
    result = list_builtin_tools()

    defaults = {
        tool["name"]
        for tool in result["tools"]
        if tool["default_enabled_for_mcp"]
    }
    assert {"repo.search", "repo.read", "repo.semantic_search", "git.diff"} <= defaults
    assert "repo.write_patch" not in defaults


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
    assert result["tool_calls_made"] == 1
    assert trace["ok"] is True
    assert trace["trace"]["final_output"]["response"] == "status is tiny"
