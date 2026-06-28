"""MCP server entrypoint for MicroModelAgent.

MCP exposes MicroModelAgent capabilities to external clients as tools. This file
builds the server, registers tools, and keeps patch application conservative by
default.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from mcp.server import NotificationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError as FastMcpToolError

from micro_model_agent.agents.tool_loop_agent import ToolLoopAgent, ToolLoopAgentTask
from micro_model_agent.application.ports import ModelProvider, ToolExecutor
from micro_model_agent.domain.contracts import ToolCall, ToolResult
from micro_model_agent.infrastructure.comparison_trace import (
    ComparisonTraceSession,
    JsonlComparisonTraceStore,
    add_comparison_event,
    comparison_session_to_record,
    review_comparison_session,
    stop_comparison_session,
)
from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
from micro_model_agent.infrastructure.repository_metadata import (
    initialize_repository,
    is_repository_initialized,
    load_repository_config,
)
from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor
from micro_model_agent.infrastructure.tools.catalog import (
    BUILTIN_TOOL_SPECS,
    builtin_tool_prompt_schemas,
)
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.infrastructure.trace_store import (
    JsonlTraceStore,
    workflow_trace_to_record,
)
from micro_model_agent.infrastructure.transformers_model_provider import (
    TransformersPeftModelProvider,
)
from micro_model_agent.infrastructure.workspace_registry import (
    JsonlWorkspaceRegistry,
    WorkspaceRecord,
    workspace_record_to_dict,
)

DEFAULT_MCP_AVAILABLE_TOOLS: tuple[str, ...] = (
    "repo.search",
    "repo.read",
    "repo.semantic_search",
    "repo.write_patch",
    "git.diff",
)
DEFAULT_7B_ADAPTER_PATH = (
    ".micro_model_agent/training/runs/qwen-coder-7b-tool-schema-20260613-205520/adapter"
)
MCP_DEBUG_TOOLS_ENV = "MICRO_MODEL_AGENT_MCP_DEBUG_TOOLS"
MCP_EXPOSE_INIT_ENV = "MICRO_MODEL_AGENT_MCP_EXPOSE_INIT"
MCP_REPOSITORY_ROOT_ENV = "MICRO_MODEL_AGENT_REPOSITORY_ROOT"
MCP_INIT_TOOL_NAME = "micro_agent_init"
type McpTransport = Literal["stdio", "sse", "streamable-http"]

_MODEL_CACHE: dict[tuple[str, str | None, int], TransformersPeftModelProvider] = {}


class PatchPolicyToolExecutor:
    """Tool executor wrapper that keeps patch writes dry-run unless explicitly enabled."""

    def __init__(self, wrapped: ToolExecutor, *, apply_patches: bool) -> None:
        self.wrapped = wrapped
        self.apply_patches = apply_patches

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.tool_name != "repo.write_patch" or self.apply_patches:
            return await self.wrapped.execute(tool_call)

        # MCP callers can preview patches by default, but real application is
        # forced back to dry-run unless apply_patches=True was requested.
        arguments = {
            **tool_call.arguments,
            "dry_run": True,
            "require_approval": True,
        }
        safe_call = replace(tool_call, arguments=arguments)
        return await self.wrapped.execute(safe_call)


async def run_agent_loop(
    *,
    goal: str,
    repository_root: str = ".",
    context: str = "",
    adapter_path: str | None = None,
    base_model: str | None = None,
    use_adapter: bool = True,
    available_tools: list[str] | None = None,
    required_tools: list[str] | None = None,
    max_turns: int = 4,
    max_tool_calls: int | None = 1,
    max_new_tokens: int = 350,
    max_tool_result_prompt_chars: int = 2500,
    schema_prompt: bool = True,
    capture_prompts: bool = False,
    apply_patches: bool = False,
    allow_test_run: bool = False,
    test_command_name: str | None = None,
    test_command_args: list[str] | None = None,
    scripted_responses: list[str] | None = None,
    comparison_session_id: str | None = None,
    comparison_repository_root: str | None = None,
    offline: bool = True,
) -> dict[str, Any]:
    """Run the model-driven tool loop and return a JSON-serializable result."""

    repository = Path(repository_root)
    # Tool availability is narrowed before the model sees it.
    allowed_tool_names = _allowed_tool_names(
        available_tools=available_tools,
        apply_patches=apply_patches,
        allow_test_run=allow_test_run or bool(test_command_name),
    )
    model_settings = _resolve_model_settings(
        repository_root=repository,
        adapter_path=adapter_path,
        base_model=base_model,
        use_adapter=use_adapter,
        allow_missing_base_model=bool(scripted_responses),
    )
    model_provider = _model_provider(
        adapter_path=model_settings["adapter_path"],
        base_model=model_settings["base_model"],
        max_new_tokens=max_new_tokens,
        scripted_responses=scripted_responses,
        offline=offline,
    )
    allowed_test_commands = _allowed_test_commands(test_command_name, test_command_args)
    # Wrap the normal executor so MCP-specific patch policy is enforced in one place.
    executor = PatchPolicyToolExecutor(
        BuiltinToolExecutor(repository, allowed_test_commands),
        apply_patches=apply_patches,
    )
    trace_store = JsonlTraceStore(repository / ".micro_model_agent" / "traces" / "workflows.jsonl")
    agent = ToolLoopAgent(
        model_provider=model_provider,
        tool_executor=executor,
        trace_store=trace_store,
    )

    result = await agent.run(
        ToolLoopAgentTask(
            goal=goal,
            available_tools=allowed_tool_names,
            required_tools=tuple(required_tools or ()),
            max_turns=max_turns,
            context=context,
            tool_schemas=builtin_tool_prompt_schemas(allowed_tool_names) if schema_prompt else {},
            max_tool_calls=max_tool_calls,
            max_tool_result_prompt_chars=max_tool_result_prompt_chars,
            capture_prompts=capture_prompts,
            run_metadata={
                "interface": "mcp",
                "schema_prompt": schema_prompt,
                "capture_prompts": capture_prompts,
                "apply_patches": apply_patches,
                "available_tools": list(allowed_tool_names),
                "required_tools": list(required_tools or ()),
                "model": model_settings,
            },
        )
    )
    if comparison_session_id:
        await _append_comparison_event(
            repository_root=Path(comparison_repository_root or repository),
            session_id=comparison_session_id,
            event_type="local_model_run",
            actor="micro_model_agent",
            payload={
                "trace_id": str(result.trace_id),
                "ok": result.ok,
                "response": result.response,
                "tool_calls_made": result.tool_calls_made,
                "model": model_settings,
            },
        )
    # Return a compact summary rather than the full trace. The full trace can be
    # loaded by debug tooling when exposed.
    return {
        "ok": result.ok,
        "response": result.response,
        "trace_id": str(result.trace_id),
        "tool_calls_made": result.tool_calls_made,
        "model": model_settings,
        "steps": [
            {
                "name": step.name,
                "status": step.status.value,
                "tool_name": step.tool_call.tool_name if step.tool_call else None,
                "tool_ok": step.tool_result.ok if step.tool_result else None,
                "tool_error": step.tool_result.error if step.tool_result else None,
            }
            for step in result.trace.steps
        ],
    }


async def call_builtin_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    repository_root: str = ".",
    apply_patches: bool = False,
    test_command_name: str | None = None,
    test_command_args: list[str] | None = None,
) -> dict[str, Any]:
    """Execute one built-in tool through MCP."""

    if tool_name not in BUILTIN_TOOL_SPECS:
        return {"ok": False, "error": f"unknown tool: {tool_name}"}

    # Direct tool execution uses the same validation and patch policy as the agent loop.
    executor = PatchPolicyToolExecutor(
        BuiltinToolExecutor(
            repository_root,
            _allowed_test_commands(test_command_name, test_command_args),
        ),
        apply_patches=apply_patches,
    )
    result = await executor.execute(ToolCall(tool_name=tool_name, arguments=arguments))
    return {
        "ok": result.ok,
        "tool_name": result.tool_name,
        "output": result.output,
        "error": result.error,
    }


async def read_trace(*, trace_id: str, repository_root: str = ".") -> dict[str, Any]:
    """Load a workflow trace captured by the local trace store."""

    trace_store = JsonlTraceStore(
        Path(repository_root) / ".micro_model_agent" / "traces" / "workflows.jsonl"
    )
    trace = await trace_store.get(trace_id)
    if trace is None:
        return {"ok": False, "error": f"trace not found: {trace_id}"}
    return {"ok": True, "trace": workflow_trace_to_record(trace)}


async def start_comparison_trace(
    *,
    goal: str,
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
    context: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a comparison trace session around a consumer/local-model task."""

    repository = Path(repository_root)
    session = ComparisonTraceSession(
        goal=goal,
        repository_root=str(repository),
        context=context,
        metadata=metadata or {},
    )
    await _comparison_trace_store(Path(comparison_repository_root or repository)).save(session)
    return {"ok": True, "session": comparison_session_to_record(session)}


async def record_comparison_event(
    *,
    session_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    actor: str = "consumer",
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
) -> dict[str, Any]:
    """Append one event to a comparison trace session."""

    session = await _append_comparison_event(
        repository_root=Path(comparison_repository_root or repository_root),
        session_id=session_id,
        event_type=event_type,
        actor=actor,
        payload=payload or {},
    )
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    return {"ok": True, "session": comparison_session_to_record(session)}


async def stop_comparison_trace(
    *,
    session_id: str,
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
    actual_summary: str = "",
    changed_files: list[str] | None = None,
    tests: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Stop a comparison trace and attach the consumer's actual result."""

    store = _comparison_trace_store(Path(comparison_repository_root or repository_root))
    session = await store.get(session_id)
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    stopped = stop_comparison_session(
        session,
        actual_result={
            "summary": actual_summary,
            "changed_files": changed_files or [],
            "tests": tests or [],
            "notes": notes,
        },
    )
    await store.save(stopped)
    return {"ok": True, "session": comparison_session_to_record(stopped)}


async def review_comparison_trace(
    *,
    session_id: str,
    repository_root: str = ".",
    comparison_repository_root: str | None = None,
    local_model_quality: str = "unknown",
    local_model_notes: str = "",
    consumer_quality: str = "unknown",
    comparison_notes: str = "",
) -> dict[str, Any]:
    """Attach review details and return a compact comparison summary."""

    repository = Path(repository_root)
    store = _comparison_trace_store(Path(comparison_repository_root or repository))
    session = await store.get(session_id)
    if session is None:
        return {"ok": False, "error": f"comparison trace not found: {session_id}"}
    local_traces = []
    trace_roots = [Path(session.repository_root)]
    comparison_root = Path(comparison_repository_root or repository)
    if comparison_root not in trace_roots:
        trace_roots.append(comparison_root)
    for trace_id in session.local_trace_ids:
        trace = None
        for trace_root in trace_roots:
            trace_store = JsonlTraceStore(
                trace_root / ".micro_model_agent" / "traces" / "workflows.jsonl"
            )
            trace = await trace_store.get(trace_id)
            if trace is not None:
                break
        if trace is not None:
            local_traces.append(workflow_trace_to_record(trace))
    reviewed = review_comparison_session(
        session,
        review={
            "local_model_quality": local_model_quality,
            "local_model_notes": local_model_notes,
            "consumer_quality": consumer_quality,
            "comparison_notes": comparison_notes,
        },
    )
    await store.save(reviewed)
    return {
        "ok": True,
        "session": comparison_session_to_record(reviewed),
        "comparison": {
            "goal": reviewed.goal,
            "status": reviewed.status,
            "local_trace_ids": reviewed.local_trace_ids,
            "local_model": [
                {
                    "trace_id": trace["id"],
                    "status": trace["status"],
                    "response": trace["final_output"].get("response"),
                    "tool_calls_made": trace["final_output"].get("tool_calls_made"),
                }
                for trace in local_traces
            ],
            "consumer_actual": reviewed.actual_result,
            "review": reviewed.review,
        },
    }


def list_builtin_tools() -> dict[str, Any]:
    """Return built-in tool names and descriptions."""

    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "default_enabled_for_mcp": spec.name in DEFAULT_MCP_AVAILABLE_TOOLS,
            }
            for spec in BUILTIN_TOOL_SPECS.values()
        ],
        "default_agent_tools": list(DEFAULT_MCP_AVAILABLE_TOOLS),
    }


async def init_workspace(
    *,
    registry_root: str | Path,
    path: str | None = None,
    name: str | None = None,
    create: bool = True,
    initialize: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or register a workspace and optionally initialize metadata there."""

    registry_base = Path(registry_root).resolve()
    if path:
        workspace_path = Path(path).expanduser().resolve()
    else:
        workspace_id = uuid4()
        directory_name = _path_safe_name(name or f"workspace-{workspace_id}")
        workspace_path = registry_base / ".micro_model_agent" / "workspaces" / directory_name

    if workspace_path.exists() and not workspace_path.is_dir():
        return {"ok": False, "error": f"workspace path is not a directory: {workspace_path}"}
    if not workspace_path.exists():
        if not create:
            return {"ok": False, "error": f"workspace path does not exist: {workspace_path}"}
        workspace_path.mkdir(parents=True, exist_ok=True)

    init_result = None
    if initialize:
        init_result = initialize_repository(workspace_path).to_dict()
        if init_result.get("ok") is not True:
            return {"ok": False, "error": init_result.get("error"), "init": init_result}

    record = WorkspaceRecord(
        path=str(workspace_path),
        name=name,
        metadata=metadata or {},
    )
    await _workspace_registry(registry_base).save(record)
    return {
        "ok": True,
        "workspace": workspace_record_to_dict(record),
        "repository_root": str(workspace_path),
        "init": init_result,
    }


def init_repository(
    *,
    repository_root: str = ".",
    default_model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    """Initialize repository metadata through MCP."""

    return initialize_repository(
        repository_root,
        default_model=default_model,
        base_model=base_model,
        adapter_path=adapter_path,
    ).to_dict()


def create_mcp_server(
    *,
    repository_root: str | Path = ".",
    expose_debug_tools: bool | None = None,
    expose_init_tool: bool | None = None,
) -> FastMCP:
    """Create the MicroModelAgent MCP server."""

    # FastMCP handles protocol details; this function only registers Python callables.
    server = FastMCP(
        "MicroModelAgent",
        instructions=(
            "Use MicroModelAgent to collect local-model coding traces. Preferred comparison "
            "workflow: call micro_agent_init_workspace when a chat needs its own directory, "
            "then call micro_agent_start_trace, call micro_agent_run_loop with workspace_id, "
            "comparison_session_id, base_model='Qwen/Qwen2.5-Coder-7B-Instruct', "
            "use_adapter=false, schema_prompt=true, capture_prompts=true, do the real work "
            "yourself, call micro_agent_stop_trace, and finish with micro_agent_review_trace. "
            "Patch writes are dry-run unless apply_patches is explicitly true."
        ),
    )
    _enable_tool_list_changed_capability(server)
    default_repository_root = str(Path(repository_root))

    @server.tool(
        name="micro_agent_init_workspace",
        description=(
            "Create or register a workspace directory and return a workspace_id for "
            "subsequent trace and local-model calls."
        ),
    )
    async def mcp_init_workspace(
        path: str | None = None,
        name: str | None = None,
        create: bool = True,
        initialize: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await init_workspace(
            registry_root=default_repository_root,
            path=path,
            name=name,
            create=create,
            initialize=initialize,
            metadata=metadata,
        )

    @server.tool(
        name="micro_agent_run_loop",
        description=(
            "Ask the local MicroModelAgent model to orchestrate repository tool calls and "
            "return a final response. Defaults target the cached Qwen 7B PEFT adapter."
        ),
    )
    async def mcp_run_agent_loop(
        goal: str,
        repository_root: str = default_repository_root,
        workspace_id: str | None = None,
        context: str = "",
        adapter_path: str | None = None,
        base_model: str | None = None,
        use_adapter: bool = True,
        available_tools: list[str] | None = None,
        required_tools: list[str] | None = None,
        max_turns: int = 4,
        max_tool_calls: int | None = 1,
        max_new_tokens: int = 350,
        max_tool_result_prompt_chars: int = 2500,
        schema_prompt: bool = True,
        capture_prompts: bool = False,
        apply_patches: bool = False,
        allow_test_run: bool = False,
        test_command_name: str | None = None,
        test_command_args: list[str] | None = None,
        comparison_session_id: str | None = None,
        offline: bool = True,
    ) -> dict[str, Any]:
        # This nested function is what MCP clients call as a tool.
        resolved_repository_root = await _resolve_workspace_root(
            registry_root=Path(default_repository_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await run_agent_loop(
            goal=goal,
            repository_root=str(resolved_repository_root),
            context=context,
            adapter_path=adapter_path,
            base_model=base_model,
            use_adapter=use_adapter,
            available_tools=available_tools,
            required_tools=required_tools,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_new_tokens=max_new_tokens,
            max_tool_result_prompt_chars=max_tool_result_prompt_chars,
            schema_prompt=schema_prompt,
            capture_prompts=capture_prompts,
            apply_patches=apply_patches,
            allow_test_run=allow_test_run,
            test_command_name=test_command_name,
            test_command_args=test_command_args,
            comparison_session_id=comparison_session_id,
            comparison_repository_root=default_repository_root,
            offline=offline,
        )

    @server.prompt(
        name="compare_local_model_on_task",
        title="Compare Local Model On Task",
        description=(
            "Reusable workflow for shadowing a Codex task with the local Qwen model "
            "and recording a comparison trace."
        ),
    )
    def compare_local_model_on_task(goal: str, context: str = "") -> str:
        return (
            "Use the MicroModelAgent MCP to compare the local model against your own work.\n\n"
            f"Goal: {goal}\n"
            f"Context: {context}\n\n"
            "Steps:\n"
            "1. If this chat needs its own directory, call micro_agent_init_workspace and keep "
            "the returned workspace_id.\n"
            "2. Call micro_agent_start_trace with the goal, context, and workspace_id if used.\n"
            "3. Call micro_agent_run_loop with the same goal, workspace_id if used, "
            "comparison_session_id from step 2, base_model='Qwen/Qwen2.5-Coder-7B-Instruct', "
            "use_adapter=false, schema_prompt=true, capture_prompts=true, and the relevant "
            "available_tools.\n"
            "4. Complete the task yourself using normal Codex tools.\n"
            "5. Call micro_agent_stop_trace with workspace_id if used, your actual summary, "
            "changed files, tests, "
            "and notes.\n"
            "6. Call micro_agent_review_trace with workspace_id if used, local_model_quality, "
            "consumer_quality, "
            "and concise comparison_notes.\n"
            "7. Report the session id and the main ways the local model matched or diverged."
        )

    @server.prompt(
        name="collect_real_trace",
        title="Collect Real Local-Model Trace",
        description="Run the base local model with explicit schemas and capture its trace.",
    )
    def collect_real_trace(goal: str, context: str = "") -> str:
        return (
            "Collect a real MicroModelAgent trace for later review.\n\n"
            f"Goal: {goal}\n"
            f"Context: {context}\n\n"
            "Call micro_agent_run_loop with base_model='Qwen/Qwen2.5-Coder-7B-Instruct', "
            "use_adapter=false, schema_prompt=true, capture_prompts=true, workspace_id if "
            "this chat initialized one, max_turns high enough for the task, and relevant "
            "available_tools. Use apply_patches=true only when real edits are intended in "
            "the configured test workspace. Record the returned trace_id for later review."
        )

    @server.prompt(
        name="review_comparison_trace",
        title="Review Comparison Trace",
        description="Review a comparison trace session after local-model and consumer work.",
    )
    def review_comparison_trace_prompt(session_id: str) -> str:
        return (
            "Review the MicroModelAgent comparison trace.\n\n"
            f"Session id: {session_id}\n\n"
            "Call micro_agent_review_trace. Set local_model_quality to good, mixed, bad, "
            "or unknown. Set consumer_quality similarly. In comparison_notes, describe "
            "whether the local model selected useful tools, read enough context, avoided "
            "bad assumptions, matched the actual changed files/tests, and produced a "
            "useful final answer."
        )

    @server.prompt(
        name="smoke_test_micro_agent",
        title="Smoke Test MicroModelAgent MCP",
        description="Simple prompt to verify the MCP server and repo tool loop are usable.",
    )
    def smoke_test_micro_agent() -> str:
        return (
            "Smoke-test the MicroModelAgent MCP. Call micro_agent_run_loop with goal "
            "'List the files in the current test workspace or explain that it is empty.', "
            "available_tools=['repo.search'], max_tool_calls=1, max_turns=4, "
            "base_model='Qwen/Qwen2.5-Coder-7B-Instruct', use_adapter=false, "
            "schema_prompt=true, capture_prompts=true, and offline=false if the base model "
            "is not cached."
        )

    @server.tool(
        name="micro_agent_start_trace",
        description=(
            "Start a comparison trace session before a consumer and the local model "
            "attempt the same repository task."
        ),
    )
    async def mcp_start_comparison_trace(
        goal: str,
        repository_root: str = default_repository_root,
        workspace_id: str | None = None,
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_repository_root = await _resolve_workspace_root(
            registry_root=Path(default_repository_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await start_comparison_trace(
            goal=goal,
            repository_root=str(resolved_repository_root),
            comparison_repository_root=default_repository_root,
            context=context,
            metadata=metadata,
        )

    @server.tool(
        name="micro_agent_record_trace_event",
        description="Record one consumer or local-model event in a comparison trace session.",
    )
    async def mcp_record_comparison_event(
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        actor: str = "consumer",
        repository_root: str = default_repository_root,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_repository_root = await _resolve_workspace_root(
            registry_root=Path(default_repository_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await record_comparison_event(
            session_id=session_id,
            event_type=event_type,
            payload=payload,
            actor=actor,
            repository_root=str(resolved_repository_root),
            comparison_repository_root=default_repository_root,
        )

    @server.tool(
        name="micro_agent_stop_trace",
        description=(
            "Stop a comparison trace session and attach the consumer's actual result."
        ),
    )
    async def mcp_stop_comparison_trace(
        session_id: str,
        repository_root: str = default_repository_root,
        workspace_id: str | None = None,
        actual_summary: str = "",
        changed_files: list[str] | None = None,
        tests: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        resolved_repository_root = await _resolve_workspace_root(
            registry_root=Path(default_repository_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await stop_comparison_trace(
            session_id=session_id,
            repository_root=str(resolved_repository_root),
            comparison_repository_root=default_repository_root,
            actual_summary=actual_summary,
            changed_files=changed_files,
            tests=tests,
            notes=notes,
        )

    @server.tool(
        name="micro_agent_review_trace",
        description=(
            "Review a comparison trace and summarize the local model result against "
            "the consumer's actual work."
        ),
    )
    async def mcp_review_comparison_trace(
        session_id: str,
        repository_root: str = default_repository_root,
        workspace_id: str | None = None,
        local_model_quality: str = "unknown",
        local_model_notes: str = "",
        consumer_quality: str = "unknown",
        comparison_notes: str = "",
    ) -> dict[str, Any]:
        resolved_repository_root = await _resolve_workspace_root(
            registry_root=Path(default_repository_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await review_comparison_trace(
            session_id=session_id,
            repository_root=str(resolved_repository_root),
            comparison_repository_root=default_repository_root,
            local_model_quality=local_model_quality,
            local_model_notes=local_model_notes,
            consumer_quality=consumer_quality,
            comparison_notes=comparison_notes,
        )

    if _should_expose_init_tool(default_repository_root, expose_init_tool):

        @server.tool(
            name=MCP_INIT_TOOL_NAME,
            description=(
                "Initialize MicroModelAgent metadata for this repository. The operation is "
                "idempotent and this tool hides itself after successful initialization."
            ),
        )
        async def mcp_init_repository(
            repository_root: str = default_repository_root,
            default_model: str | None = None,
            base_model: str | None = None,
            adapter_path: str | None = None,
        ) -> dict[str, Any]:
            result = init_repository(
                repository_root=repository_root,
                default_model=default_model,
                base_model=base_model,
                adapter_path=adapter_path,
            )
            if result.get("ok") is True:
                # Once initialization succeeds, hide the init tool and notify
                # clients that the tool list changed.
                await _remove_tool_and_notify(server, MCP_INIT_TOOL_NAME)
            return result

    if _should_expose_debug_tools(expose_debug_tools):
        # Debug tools are useful locally but are hidden unless explicitly enabled.

        @server.tool(
            name="micro_agent_builtin_tool",
            description="Execute one built-in MicroModelAgent repository tool directly.",
        )
        async def mcp_call_builtin_tool(
            tool_name: str,
            arguments: dict[str, Any],
            repository_root: str = default_repository_root,
            workspace_id: str | None = None,
            apply_patches: bool = False,
            test_command_name: str | None = None,
            test_command_args: list[str] | None = None,
        ) -> dict[str, Any]:
            resolved_repository_root = await _resolve_workspace_root(
                registry_root=Path(default_repository_root),
                repository_root=repository_root,
                workspace_id=workspace_id,
            )
            return await call_builtin_tool(
                tool_name=tool_name,
                arguments=arguments,
                repository_root=str(resolved_repository_root),
                apply_patches=apply_patches,
                test_command_name=test_command_name,
                test_command_args=test_command_args,
            )

        @server.tool(
            name="micro_agent_read_trace",
            description="Read a stored MicroModelAgent workflow trace by trace id.",
        )
        async def mcp_read_trace(
            trace_id: str,
            repository_root: str = default_repository_root,
            workspace_id: str | None = None,
        ) -> dict[str, Any]:
            resolved_repository_root = await _resolve_workspace_root(
                registry_root=Path(default_repository_root),
                repository_root=repository_root,
                workspace_id=workspace_id,
            )
            return await read_trace(
                trace_id=trace_id,
                repository_root=str(resolved_repository_root),
            )

        @server.tool(
            name="micro_agent_list_builtin_tools",
            description="List MicroModelAgent built-in tools and default MCP availability.",
        )
        def mcp_list_builtin_tools() -> dict[str, Any]:
            return list_builtin_tools()

    return server


def serve(transport: str = "stdio", *, repository_root: str | Path = ".") -> None:
    """Run the MCP server."""

    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(f"unknown MCP transport: {transport}")
    create_mcp_server(repository_root=repository_root).run(transport=cast(McpTransport, transport))


def main() -> None:
    """Console entrypoint for `python -m micro_model_agent.interfaces.mcp_server`."""

    parser = argparse.ArgumentParser(description="Serve MicroModelAgent over MCP.")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=("stdio", "sse", "streamable-http"),
        help="MCP transport.",
    )
    parser.add_argument(
        "--repository-root",
        default=os.environ.get(MCP_REPOSITORY_ROOT_ENV, "."),
        help=(
            "Default repository root for MCP tools. Also configurable with "
            f"{MCP_REPOSITORY_ROOT_ENV}."
        ),
    )
    args = parser.parse_args()
    serve(transport=args.transport, repository_root=args.repository_root)


def _model_provider(
    *,
    adapter_path: str | None,
    base_model: str | None,
    max_new_tokens: int,
    scripted_responses: list[str] | None,
    offline: bool,
) -> ModelProvider:
    if scripted_responses:
        return ScriptedModelProvider(scripted_responses)

    if offline:
        # Offline defaults keep local adapter runs from unexpectedly reaching
        # out to the Hugging Face hub.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    adapter = Path(adapter_path) if adapter_path else None
    resolved_base_model = base_model
    if resolved_base_model is None:
        if adapter is None:
            raise ValueError("base_model is required when adapter_path is not set")
        resolved_base_model = _base_model_from_adapter(adapter)

    cache_key = (resolved_base_model, str(adapter) if adapter else None, max_new_tokens)
    provider = _MODEL_CACHE.get(cache_key)
    if provider is None:
        # Cache providers so repeated MCP calls do not reload model weights.
        provider = TransformersPeftModelProvider(
            base_model=resolved_base_model,
            adapter_path=adapter,
            max_new_tokens=max_new_tokens,
        )
        _MODEL_CACHE[cache_key] = provider
    return provider


def _resolve_model_settings(
    *,
    repository_root: str | Path,
    adapter_path: str | None,
    base_model: str | None,
    use_adapter: bool,
    allow_missing_base_model: bool = False,
) -> dict[str, str | None]:
    """Resolve MCP model settings from explicit args, env, selected config, defaults."""

    repository_config = load_repository_config(repository_root) or {}
    model_config = repository_config.get("model")
    if not isinstance(model_config, dict):
        model_config = {}

    resolved_adapter_path = None
    if use_adapter:
        resolved_adapter_path = (
            adapter_path
            or os.environ.get("MICRO_MODEL_AGENT_ADAPTER_PATH")
            or _string_config_value(model_config, "adapter_path")
            or DEFAULT_7B_ADAPTER_PATH
        )
    resolved_base_model = (
        base_model
        or os.environ.get("MICRO_MODEL_AGENT_BASE_MODEL")
        or _string_config_value(model_config, "base_model")
    )
    if resolved_base_model is None and not allow_missing_base_model:
        if resolved_adapter_path is None:
            raise ValueError("--base-model is required when use_adapter is false")
        resolved_base_model = _base_model_from_adapter(Path(resolved_adapter_path))

    selected_promotion = model_config.get("selected_promotion")
    selected_artifact_id = None
    if isinstance(selected_promotion, dict):
        raw_artifact_id = selected_promotion.get("artifact_id")
        selected_artifact_id = raw_artifact_id if isinstance(raw_artifact_id, str) else None

    return {
        "base_model": resolved_base_model,
        "adapter_path": resolved_adapter_path,
        "selected_promotion_artifact_id": selected_artifact_id,
    }


def _string_config_value(config: dict[str, Any], key: str) -> str | None:
    """Read one string value from repository config."""

    value = config.get(key)
    return value if isinstance(value, str) and value else None


def _base_model_from_adapter(adapter_path: Path) -> str:
    """Read the base model name stored in a PEFT adapter config."""

    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter config does not exist: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError(f"adapter config does not contain base_model_name_or_path: {config_path}")
    return base_model


def _allowed_tool_names(
    *,
    available_tools: list[str] | None,
    apply_patches: bool,
    allow_test_run: bool,
) -> tuple[str, ...]:
    """Filter requested tool names according to MCP safety options."""

    requested = tuple(available_tools or DEFAULT_MCP_AVAILABLE_TOOLS)
    allowed: list[str] = []
    for tool_name in requested:
        if tool_name not in BUILTIN_TOOL_SPECS:
            raise ValueError(f"unknown built-in tool: {tool_name}")
        if tool_name == "repo.write_patch" and not apply_patches:
            # The wrapper will force this tool into dry-run mode.
            allowed.append(tool_name)
            continue
        if tool_name == "test.run" and not allow_test_run:
            continue
        allowed.append(tool_name)
    return tuple(dict.fromkeys(allowed))


def _allowed_test_commands(
    test_command_name: str | None,
    test_command_args: list[str] | None,
) -> dict[str, AllowedTestCommand]:
    """Build the allowlist consumed by the test.run tool."""

    if not test_command_name or not test_command_args:
        return {}
    return {test_command_name: AllowedTestCommand(tuple(test_command_args))}


def _comparison_trace_store(repository_root: Path) -> JsonlComparisonTraceStore:
    """Return the comparison trace store for one repository."""

    return JsonlComparisonTraceStore(
        repository_root / ".micro_model_agent" / "traces" / "comparison_sessions.jsonl"
    )


def _workspace_registry(registry_root: Path) -> JsonlWorkspaceRegistry:
    """Return the workspace registry for the server's default root."""

    return JsonlWorkspaceRegistry(registry_root / ".micro_model_agent" / "workspaces.jsonl")


async def _resolve_workspace_root(
    *,
    registry_root: Path,
    repository_root: str,
    workspace_id: str | None,
) -> Path:
    """Resolve a workspace id or direct repository root into a filesystem path."""

    if not workspace_id:
        return Path(repository_root)
    workspace = await _workspace_registry(registry_root.resolve()).get(workspace_id)
    if workspace is None:
        raise FastMcpToolError(f"workspace_id not found: {workspace_id}")
    return Path(workspace.path)


async def _append_comparison_event(
    *,
    repository_root: Path,
    session_id: str,
    event_type: str,
    actor: str,
    payload: dict[str, Any],
) -> ComparisonTraceSession | None:
    """Append an event to a comparison trace session, if it exists."""

    store = _comparison_trace_store(repository_root)
    session = await store.get(session_id)
    if session is None:
        return None
    updated = add_comparison_event(
        session,
        event_type=event_type,
        actor=actor,
        payload=payload,
    )
    await store.save(updated)
    return updated


def _path_safe_name(value: str) -> str:
    """Return a conservative directory name for generated workspaces."""

    safe = "".join(character if character.isalnum() else "-" for character in value.lower())
    return "-".join(part for part in safe.split("-") if part) or "workspace"


def _should_expose_debug_tools(explicit: bool | None) -> bool:
    """Decide whether debug tools should be registered."""

    if explicit is not None:
        return explicit
    return _truthy_env(MCP_DEBUG_TOOLS_ENV)


def _should_expose_init_tool(repository_root: str | Path, explicit: bool | None) -> bool:
    """Show the init tool only when requested or when the repository is uninitialized."""

    if explicit is not None:
        return explicit
    if _truthy_env(MCP_EXPOSE_INIT_ENV):
        return True
    return not is_repository_initialized(repository_root)


def _truthy_env(name: str) -> bool:
    """Interpret common truthy environment variable values."""

    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enable_tool_list_changed_capability(server: FastMCP) -> None:
    """Advertise dynamic tool-list updates for FastMCP versions without a public knob."""

    # Some FastMCP versions do not expose this capability directly, so the code
    # wraps the low-level initialization function.
    lowlevel_server = cast(Any, server)._mcp_server
    original_create_initialization_options = lowlevel_server.create_initialization_options

    def create_initialization_options(
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> Any:
        if notification_options is None:
            notification_options = NotificationOptions(tools_changed=True)
        else:
            notification_options.tools_changed = True
        return original_create_initialization_options(
            notification_options,
            experimental_capabilities,
        )

    lowlevel_server.create_initialization_options = create_initialization_options


async def _remove_tool_and_notify(server: FastMCP, tool_name: str) -> None:
    """Remove a tool and notify connected clients when possible."""

    try:
        server.remove_tool(tool_name)
    except FastMcpToolError:
        return

    try:
        context = server.get_context()
        await context.request_context.session.send_tool_list_changed()
    except (AttributeError, LookupError, RuntimeError, ValueError):
        return


if __name__ == "__main__":
    main()
