"""Tests for MCP prompt builder functions."""

from __future__ import annotations

from micro_model_agent.interfaces.mcp.compat import CANONICAL_TOOL_NAMES_TEXT
from micro_model_agent.interfaces.mcp.prompts.registry import (
    collect_real_trace_prompt,
    compare_local_model_on_task_prompt,
    review_comparison_trace_prompt,
)


def test_compare_local_model_prompt_embeds_goal_and_context() -> None:
    result = compare_local_model_on_task_prompt(
        goal="refactor the parser", context="Python 3.12"
    )
    assert "refactor the parser" in result
    assert "Python 3.12" in result


def test_collect_real_trace_prompt_lists_canonical_tools() -> None:
    result = collect_real_trace_prompt(goal="add a feature")
    for tool in CANONICAL_TOOL_NAMES_TEXT.split(", "):
        assert tool in result, f"canonical tool {tool!r} missing from prompt"


def test_review_comparison_trace_prompt_includes_session_id() -> None:
    result = review_comparison_trace_prompt(session_id="abc-123")
    assert "abc-123" in result
