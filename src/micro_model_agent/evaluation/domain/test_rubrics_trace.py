"""Tests for the trace evaluation rubrics domain scoring logic."""

from __future__ import annotations

import json

import pytest

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.evaluation.domain.rubrics_trace import (
    TraceExampleScore,
    expected_trace_final_response,
    expected_trace_patch,
    expected_trace_tool_names,
    normalize_trace_text,
    score_trace_example,
    trace_category,
    trace_id,
    trace_patch_match,
    trace_similarity,
    trace_tool_names_from_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label() -> DatasetLabel:
    return DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)


def _example(
    target: dict | None = None,
    metadata: dict | None = None,
) -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "fix the null pointer"},
        target=target or {},
        label=_label(),
        metadata=metadata or {},
    )


def _resp(**kwargs) -> str:
    return json.dumps(kwargs)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_expected_trace_final_response_from_final_response() -> None:
    ex = _example(target={"final_response": "The fix applied cleanly."})
    assert expected_trace_final_response(ex) == "The fix applied cleanly."


def test_expected_trace_final_response_from_summary_fallback() -> None:
    ex = _example(target={"summary": "Applied patch successfully."})
    assert expected_trace_final_response(ex) == "Applied patch successfully."


def test_expected_trace_final_response_none_when_absent() -> None:
    ex = _example(target={})
    assert expected_trace_final_response(ex) is None


def test_expected_trace_patch_present() -> None:
    ex = _example(target={"patch": "--- a/f.py\n+++ b/f.py"})
    assert expected_trace_patch(ex) == "--- a/f.py\n+++ b/f.py"


def test_expected_trace_patch_none_when_absent() -> None:
    ex = _example(target={"final_response": "done"})
    assert expected_trace_patch(ex) is None


def test_expected_trace_tool_names_from_metadata() -> None:
    # expected_trace_tool_names reads from example.input["tool_history"]
    ex = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={
            "goal": "fix",
            "tool_history": [
                {"tool_call": {"tool_name": "repo.read"}},
                {"tool_call": {"tool_name": "repo.write_patch"}},
            ],
        },
        target={},
        label=_label(),
    )
    names = expected_trace_tool_names(ex)
    assert "repo.read" in names
    assert "repo.write_patch" in names


def test_expected_trace_tool_names_empty_when_absent() -> None:
    ex = _example()
    assert expected_trace_tool_names(ex) == []


def test_trace_tool_names_from_response() -> None:
    response = {
        "tool_history": [
            {"tool_call": {"tool_name": "repo.read"}},
            {"tool_call": {"tool_name": "repo.write_patch"}},
        ]
    }
    names = trace_tool_names_from_response(response)
    assert "repo.read" in names
    assert "repo.write_patch" in names


def test_trace_similarity_identical_strings() -> None:
    assert trace_similarity("hello world", "hello world") == 1.0


def test_trace_similarity_empty_strings() -> None:
    result = trace_similarity("", "")
    assert result >= 0.0


def test_trace_similarity_different_strings() -> None:
    score = trace_similarity("fix the bug", "completely unrelated text")
    assert 0.0 <= score <= 1.0
    assert score < 1.0


def test_normalize_trace_text_normalizes_whitespace() -> None:
    # normalize_trace_text collapses internal whitespace but doesn't lowercase
    assert normalize_trace_text("hello   world") == "hello world"


def test_normalize_trace_text_strips_whitespace() -> None:
    assert normalize_trace_text("  hello  ") == "hello"


def test_trace_id_from_metadata() -> None:
    ex = _example(metadata={"trace_id": "abc-123"})
    assert trace_id(ex) == "abc-123"


def test_trace_id_none_when_absent() -> None:
    ex = _example()
    assert trace_id(ex) is None


def test_trace_category_from_metadata() -> None:
    ex = _example(metadata={"category": "repair_basic"})
    assert trace_category(ex) == "repair_basic"


# ---------------------------------------------------------------------------
# score_trace_example
# ---------------------------------------------------------------------------


def test_score_final_response_match() -> None:
    ex = _example(target={"final_response": "The patch was applied."})
    response = _resp(final_response="The patch was applied.")
    score = score_trace_example(ex, response)
    assert score.parse_success is True
    assert score.final_response_match is True
    assert score.score == 1.0


def test_score_final_response_mismatch() -> None:
    ex = _example(target={"final_response": "Applied successfully."})
    response = _resp(final_response="I don't know what happened.")
    score = score_trace_example(ex, response)
    # trace_similarity may still give partial credit — score >= 0
    assert score.parse_success is True
    assert 0.0 <= score.score <= 1.0


def test_score_patch_match() -> None:
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-bug\n+fix"
    ex = _example(target={"patch": patch})
    response = _resp(patch=patch)
    score = score_trace_example(ex, response)
    assert score.patch_match is True
    assert score.score == 1.0


def test_score_tool_history_match() -> None:
    # expected_trace_tool_names reads from input.tool_history
    ex = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={
            "goal": "fix",
            "tool_history": [
                {"tool_call": {"tool_name": "repo.read"}},
                {"tool_call": {"tool_name": "repo.write_patch"}},
            ],
        },
        target={},
        label=_label(),
    )
    response = _resp(
        tool_history=[
            {"tool_call": {"tool_name": "repo.read"}},
            {"tool_call": {"tool_name": "repo.write_patch"}},
        ]
    )
    score = score_trace_example(ex, response)
    assert score.tool_history_match is True
    assert score.score == 1.0


def test_score_unparseable_response_gives_zero() -> None:
    ex = _example(target={"final_response": "done"})
    score = score_trace_example(ex, "I cannot do that.")
    assert score.parse_success is False
    assert score.score == 0.0


def test_score_empty_target_gives_zero_with_error() -> None:
    ex = _example(target={})
    score = score_trace_example(ex, _resp())
    assert score.score == 0.0
    assert any("no scorable target" in e for e in score.errors)


def test_score_returns_trace_example_score_type() -> None:
    ex = _example(target={"final_response": "done"})
    result = score_trace_example(ex, _resp(final_response="done"))
    assert isinstance(result, TraceExampleScore)


def test_score_example_id_matches() -> None:
    ex = _example(target={"final_response": "done"})
    score = score_trace_example(ex, _resp(final_response="done"))
    assert score.example_id == str(ex.id)
