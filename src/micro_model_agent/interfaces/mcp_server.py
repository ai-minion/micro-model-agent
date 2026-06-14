"""MCP server entrypoint for MicroModelAgent.

MCP exposes MicroModelAgent capabilities to external clients as tools. This file
builds the server, registers tools, and keeps patch application conservative by
default.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from mcp.server import NotificationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError as FastMcpToolError

from micro_model_agent.agents.tool_loop_agent import ToolLoopAgent, ToolLoopAgentTask
from micro_model_agent.application.ports import ModelProvider, ToolExecutor
from micro_model_agent.domain.contracts import ToolCall, ToolResult
from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
from micro_model_agent.infrastructure.repository_metadata import (
    initialize_repository,
    is_repository_initialized,
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

DEFAULT_MCP_AVAILABLE_TOOLS: tuple[str, ...] = (
    "repo.search",
    "repo.read",
    "repo.semantic_search",
    "git.diff",
)
DEFAULT_7B_ADAPTER_PATH = (
    ".micro_model_agent/training/runs/qwen-coder-7b-tool-schema-20260613-205520/adapter"
)
MCP_DEBUG_TOOLS_ENV = "MICRO_MODEL_AGENT_MCP_DEBUG_TOOLS"
MCP_EXPOSE_INIT_ENV = "MICRO_MODEL_AGENT_MCP_EXPOSE_INIT"
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
    available_tools: list[str] | None = None,
    required_tools: list[str] | None = None,
    max_turns: int = 4,
    max_tool_calls: int | None = 1,
    max_new_tokens: int = 350,
    max_tool_result_prompt_chars: int = 2500,
    schema_prompt: bool = False,
    apply_patches: bool = False,
    allow_test_run: bool = False,
    test_command_name: str | None = None,
    test_command_args: list[str] | None = None,
    scripted_responses: list[str] | None = None,
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
    model_provider = _model_provider(
        adapter_path=adapter_path,
        base_model=base_model,
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
        )
    )
    # Return a compact summary rather than the full trace. The full trace can be
    # loaded by debug tooling when exposed.
    return {
        "ok": result.ok,
        "response": result.response,
        "trace_id": str(result.trace_id),
        "tool_calls_made": result.tool_calls_made,
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
            "Run a local MicroModelAgent coding workflow through constrained repository "
            "tools. Patch writes are dry-run unless apply_patches is explicitly true."
        ),
    )
    _enable_tool_list_changed_capability(server)
    default_repository_root = str(Path(repository_root))

    @server.tool(
        name="micro_agent_run_loop",
        description=(
            "Ask the local MicroModelAgent model to orchestrate repository tool calls and "
            "return a final response. Defaults target the cached Qwen 7B PEFT adapter."
        ),
    )
    async def mcp_run_agent_loop(
        goal: str,
        repository_root: str = ".",
        context: str = "",
        adapter_path: str | None = None,
        base_model: str | None = None,
        available_tools: list[str] | None = None,
        required_tools: list[str] | None = None,
        max_turns: int = 4,
        max_tool_calls: int | None = 1,
        max_new_tokens: int = 350,
        max_tool_result_prompt_chars: int = 2500,
        schema_prompt: bool = False,
        apply_patches: bool = False,
        allow_test_run: bool = False,
        test_command_name: str | None = None,
        test_command_args: list[str] | None = None,
        offline: bool = True,
    ) -> dict[str, Any]:
        # This nested function is what MCP clients call as a tool.
        return await run_agent_loop(
            goal=goal,
            repository_root=repository_root,
            context=context,
            adapter_path=adapter_path,
            base_model=base_model,
            available_tools=available_tools,
            required_tools=required_tools,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_new_tokens=max_new_tokens,
            max_tool_result_prompt_chars=max_tool_result_prompt_chars,
            schema_prompt=schema_prompt,
            apply_patches=apply_patches,
            allow_test_run=allow_test_run,
            test_command_name=test_command_name,
            test_command_args=test_command_args,
            offline=offline,
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
            repository_root: str = ".",
            apply_patches: bool = False,
            test_command_name: str | None = None,
            test_command_args: list[str] | None = None,
        ) -> dict[str, Any]:
            return await call_builtin_tool(
                tool_name=tool_name,
                arguments=arguments,
                repository_root=repository_root,
                apply_patches=apply_patches,
                test_command_name=test_command_name,
                test_command_args=test_command_args,
            )

        @server.tool(
            name="micro_agent_read_trace",
            description="Read a stored MicroModelAgent workflow trace by trace id.",
        )
        async def mcp_read_trace(trace_id: str, repository_root: str = ".") -> dict[str, Any]:
            return await read_trace(trace_id=trace_id, repository_root=repository_root)

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

    serve()


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

    resolved_adapter_path = adapter_path or os.environ.get(
        "MICRO_MODEL_AGENT_ADAPTER_PATH",
        DEFAULT_7B_ADAPTER_PATH,
    )
    adapter = Path(resolved_adapter_path)
    resolved_base_model = base_model or os.environ.get("MICRO_MODEL_AGENT_BASE_MODEL")
    if resolved_base_model is None:
        resolved_base_model = _base_model_from_adapter(adapter)

    cache_key = (resolved_base_model, str(adapter), max_new_tokens)
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
