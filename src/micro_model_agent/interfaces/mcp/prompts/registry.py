"""MCP prompt registration."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from micro_model_agent.interfaces.mcp.compat import CANONICAL_TOOL_NAMES_TEXT


def register_workflow_prompts(server: FastMCP) -> None:
    """Register the public workflow prompts exposed by the MCP server."""

    @server.prompt(
        name="compare_local_model_on_task",
        title="Compare Local Model On Task",
        description=(
            "Reusable workflow for shadowing a Codex task with the local Qwen model "
            "and recording a comparison trace."
        ),
    )
    def compare_local_model_on_task(goal: str, context: str = "") -> str:
        return compare_local_model_on_task_prompt(goal=goal, context=context)

    @server.prompt(
        name="collect_real_trace",
        title="Collect Real Local-Model Trace",
        description="Run the base local model with explicit schemas and capture its trace.",
    )
    def collect_real_trace(goal: str, context: str = "") -> str:
        return collect_real_trace_prompt(goal=goal, context=context)

    @server.prompt(
        name="review_comparison_trace",
        title="Review Comparison Trace",
        description="Review a comparison trace session after local-model and consumer work.",
    )
    def review_comparison_trace_prompt_adapter(session_id: str) -> str:
        return review_comparison_trace_prompt(session_id=session_id)

    @server.prompt(
        name="smoke_test_micro_agent",
        title="Smoke Test MicroModelAgent MCP",
        description="Simple prompt to verify the MCP server and repo tool loop are usable.",
    )
    def smoke_test_micro_agent() -> str:
        return smoke_test_micro_agent_prompt()


def compare_local_model_on_task_prompt(goal: str, context: str = "") -> str:
    """Build the comparison workflow prompt text."""

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
        "use_adapter=false, expose_tool_schemas=true, and the relevant "
        f"available_tools. Use only MicroModelAgent tool names: {CANONICAL_TOOL_NAMES_TEXT}. "
        "Do not pass Codex tool names such as shell or apply_patch.\n"
        "4. Complete the task yourself using normal Codex tools.\n"
        "5. Call micro_agent_stop_trace with workspace_id if used, your actual summary, "
        "changed files, tests, "
        "and notes.\n"
        "6. Call micro_agent_review_trace with workspace_id if used, local_model_quality, "
        "consumer_quality, "
        "and concise comparison_notes.\n"
        "7. Report the session id and the main ways the local model matched or diverged."
    )


def collect_real_trace_prompt(goal: str, context: str = "") -> str:
    """Build the trace-collection prompt text."""

    return (
        "Collect a real MicroModelAgent trace for later review.\n\n"
        f"Goal: {goal}\n"
        f"Context: {context}\n\n"
        "Call micro_agent_run_loop with base_model='Qwen/Qwen2.5-Coder-7B-Instruct', "
        "use_adapter=false, expose_tool_schemas=true, workspace_id if "
        "this chat initialized one, max_turns high enough for the task, and relevant "
        f"available_tools from this canonical list: {CANONICAL_TOOL_NAMES_TEXT}. "
        "Prefer repo.write_files for greenfield file creation and repo.write_patch for "
        "precise edits. Record the returned trace_id for later review."
    )


def review_comparison_trace_prompt(session_id: str) -> str:
    """Build the comparison-review prompt text."""

    return (
        "Review the MicroModelAgent comparison trace.\n\n"
        f"Session id: {session_id}\n\n"
        "Call micro_agent_review_trace. Set local_model_quality to good, mixed, bad, "
        "or unknown. Set consumer_quality similarly. In comparison_notes, describe "
        "whether the local model selected useful tools, read enough context, avoided "
        "bad assumptions, matched the actual changed files/tests, and produced a "
        "useful final answer."
    )


def smoke_test_micro_agent_prompt() -> str:
    """Build the smoke-test prompt text."""

    return (
        "Smoke-test the MicroModelAgent MCP. Call micro_agent_run_loop with goal "
        "'List the files in the current test workspace or explain that it is empty.', "
        "available_tools=['repo.search'], max_tool_calls=1, max_turns=4, "
        "base_model='Qwen/Qwen2.5-Coder-7B-Instruct', use_adapter=false, "
        "expose_tool_schemas=true, and offline=false if the base model "
        "is not cached."
    )
