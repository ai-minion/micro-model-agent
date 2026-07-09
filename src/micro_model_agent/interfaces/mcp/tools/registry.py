"""FastMCP tool registration for the MCP interface."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError as FastMcpToolError

from micro_model_agent.interfaces.mcp.compat import (
    CANONICAL_TOOL_NAMES_TEXT,
    MCP_INIT_TOOL_NAME,
    RunProfile,
)
from micro_model_agent.interfaces.mcp.policy.exposure import (
    should_expose_debug_tools,
    should_expose_init_tool,
)
from micro_model_agent.interfaces.mcp.tools.builtin import (
    call_builtin_tool,
    list_builtin_tools,
)
from micro_model_agent.interfaces.mcp.traces import (
    read_trace,
    record_comparison_event,
    review_comparison_trace,
    start_comparison_trace,
    stop_comparison_trace,
)
from micro_model_agent.interfaces.mcp.workspace import (
    init_repository,
    init_workspace,
    resolve_workspace_root,
)

RunAgentLoopHandler = Callable[..., Awaitable[dict[str, Any]]]


def register_mcp_tools(
    server: FastMCP,
    *,
    default_repository_root: str,
    registry_root: str | None = None,
    expose_debug_tools: bool | None,
    expose_init_tool: bool | None,
    run_agent_loop_handler: RunAgentLoopHandler,
) -> None:
    """Register MicroModelAgent's public and conditional MCP tools."""

    resolved_registry_root = registry_root or default_repository_root
    trace_repository_root = str(Path(resolved_registry_root))

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
            registry_root=resolved_registry_root,
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
            "return a final response. Defaults use configured model settings and run "
            "without a fine-tuning adapter when no adapter path is configured. "
            "Each run has a limited turn/tool budget; use run_profile='quick', 'standard', "
            "or 'extended' to choose the size, and split broad tasks into multiple focused "
            "loops instead of asking one loop to build, repair, test, and document everything. "
            f"Canonical available_tools names are: {CANONICAL_TOOL_NAMES_TEXT}. "
            "Use these names instead of Codex tool names such as shell or apply_patch."
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
        model_timeout_seconds: float | None = None,
        run_profile: RunProfile | None = None,
        expose_tool_schemas: bool = True,
        allow_test_run: bool = False,
        test_command_name: str | None = None,
        test_command_args: list[str] | None = None,
        comparison_session_id: str | None = None,
        offline: bool = True,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_repository_root = await resolve_workspace_root(
            registry_root=Path(resolved_registry_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )

        progress_ctx = ctx

        async def _on_turn(turn_number: int, max_turns: int, message: str) -> None:
            if progress_ctx is None:
                return
            label = (
                f"[turn {turn_number}/{max_turns}] {message}"
                if message
                else f"[turn {turn_number}/{max_turns}]"
            )
            try:
                await progress_ctx.report_progress(turn_number - 1, max_turns, message=label)
                await progress_ctx.info(label)
            except Exception:
                pass

        return await run_agent_loop_handler(
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
            model_timeout_seconds=model_timeout_seconds,
            run_profile=run_profile,
            expose_tool_schemas=expose_tool_schemas,
            apply_patches=True,
            allow_test_run=allow_test_run,
            test_command_name=test_command_name,
            test_command_args=test_command_args,
            comparison_session_id=comparison_session_id,
            server_repository_root=trace_repository_root,
            offline=offline,
            on_turn=_on_turn if ctx is not None else None,
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
        resolved_repository_root = await resolve_workspace_root(
            registry_root=Path(resolved_registry_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await start_comparison_trace(
            goal=goal,
            repository_root=trace_repository_root,
            task_repository_root=str(resolved_repository_root),
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
        await resolve_workspace_root(
            registry_root=Path(resolved_registry_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await record_comparison_event(
            session_id=session_id,
            event_type=event_type,
            payload=payload,
            actor=actor,
            repository_root=trace_repository_root,
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
        await resolve_workspace_root(
            registry_root=Path(resolved_registry_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await stop_comparison_trace(
            session_id=session_id,
            repository_root=trace_repository_root,
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
        resolved_repository_root = await resolve_workspace_root(
            registry_root=Path(resolved_registry_root),
            repository_root=repository_root,
            workspace_id=workspace_id,
        )
        return await review_comparison_trace(
            session_id=session_id,
            repository_root=trace_repository_root,
            task_repository_root=str(resolved_repository_root),
            local_model_quality=local_model_quality,
            local_model_notes=local_model_notes,
            consumer_quality=consumer_quality,
            comparison_notes=comparison_notes,
        )

    if should_expose_init_tool(resolved_registry_root, expose_init_tool):

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
                await remove_tool_and_notify(server, MCP_INIT_TOOL_NAME)
            return result

    if should_expose_debug_tools(expose_debug_tools):

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
            resolved_repository_root = await resolve_workspace_root(
                registry_root=Path(resolved_registry_root),
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
            await resolve_workspace_root(
                registry_root=Path(resolved_registry_root),
                repository_root=repository_root,
                workspace_id=workspace_id,
            )
            return await read_trace(
                trace_id=trace_id,
                repository_root=trace_repository_root,
            )

        @server.tool(
            name="micro_agent_list_builtin_tools",
            description="List MicroModelAgent built-in tools and default MCP availability.",
        )
        def mcp_list_builtin_tools() -> dict[str, Any]:
            return list_builtin_tools()


async def remove_tool_and_notify(server: FastMCP, tool_name: str) -> None:
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
