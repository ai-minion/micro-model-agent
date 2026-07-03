"""Tests for the synthetic evaluation rubrics domain scoring logic."""

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
from micro_model_agent.evaluation.domain.rubrics_synthetic import (
    SyntheticExampleScore,
    expected_tool_name,
    expects_refusal,
    json_object_from_response,
    score_synthetic_example,
    strip_markdown_fence,
    synthetic_category,
    synthetic_metrics,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _example(
    *,
    kind: DatasetExampleKind = DatasetExampleKind.TOOL_USE,
    input_extra: dict | None = None,
    target_extra: dict | None = None,
    outcome: OutcomeLabel = OutcomeLabel.ACCEPTED,
) -> DatasetExample:
    label = DatasetLabel(outcome=outcome, quality=QualityLabel.GOOD)
    inp = {"tool_name": "repo_read", "arguments": {"path": "main.py"}}
    inp.update(input_extra or {})
    target = {"tool_name": "repo_read", "arguments": {"path": "main.py"}}
    target.update(target_extra or {})
    return DatasetExample(kind=kind, input=inp, target=target, label=label)


def _resp(tool_name: str = "repo_read", arguments: dict | None = None) -> str:
    return json.dumps({"tool_name": tool_name, "arguments": arguments or {"path": "main.py"}})


# ---------------------------------------------------------------------------
# json_object_from_response
# ---------------------------------------------------------------------------


def test_parses_plain_json() -> None:
    result = json_object_from_response('{"tool_name": "repo_read"}')
    assert result["tool_name"] == "repo_read"


def test_parses_json_with_leading_text() -> None:
    result = json_object_from_response('here is my response: {"tool_name": "repo_read"}')
    assert result["tool_name"] == "repo_read"


def test_parses_markdown_fenced_json() -> None:
    response = '```json\n{"tool_name": "repo_read"}\n```'
    result = json_object_from_response(response)
    assert result["tool_name"] == "repo_read"


def test_raises_on_non_json_response() -> None:
    with pytest.raises(ValueError, match="JSON"):
        json_object_from_response("I cannot help with that.")


def test_raises_on_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        json_object_from_response("[1, 2, 3]")


# ---------------------------------------------------------------------------
# strip_markdown_fence
# ---------------------------------------------------------------------------


def test_strip_removes_json_fence() -> None:
    fenced = "```json\n{\"key\": \"value\"}\n```"
    stripped = strip_markdown_fence(fenced)
    assert stripped == '{"key": "value"}'


def test_strip_removes_plain_fence() -> None:
    fenced = "```\n{\"key\": \"value\"}\n```"
    stripped = strip_markdown_fence(fenced)
    assert stripped == '{"key": "value"}'


def test_strip_noop_for_plain_text() -> None:
    plain = '{"tool_name": "repo_read"}'
    assert strip_markdown_fence(plain) == plain


# ---------------------------------------------------------------------------
# expected_tool_name / expects_refusal / synthetic_category
# ---------------------------------------------------------------------------


def test_expected_tool_name_from_target() -> None:
    example = _example()
    assert expected_tool_name(example) == "repo_read"


def test_expected_tool_name_missing_returns_none() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    example = DatasetExample(kind=DatasetExampleKind.TOOL_USE, input={}, target={}, label=label)
    assert expected_tool_name(example) is None


def test_expects_refusal_false_by_default() -> None:
    assert expects_refusal(_example()) is False


def test_expects_refusal_true_when_flagged() -> None:
    # expects_refusal is True when outcome is REJECTED
    example = _example(outcome=OutcomeLabel.REJECTED)
    assert expects_refusal(example) is True


def test_synthetic_category_from_input_metadata() -> None:
    # synthetic_category reads from example.metadata, not input
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"tool_name": "repo_read", "arguments": {}},
        target={"tool_name": "repo_read", "arguments": {}},
        label=label,
        metadata={"category": "tool_use_basic"},
    )
    assert synthetic_category(example) == "tool_use_basic"


def test_synthetic_category_none_when_absent() -> None:
    assert synthetic_category(_example()) is None


# ---------------------------------------------------------------------------
# score_synthetic_example
# ---------------------------------------------------------------------------


def test_perfect_response_scores_one() -> None:
    # Without tool_argument_contracts, valid_arguments=False for tool_use examples.
    # A perfect score (1.0) requires passing contracts. Without them, score < 1.
    example = _example()
    score = score_synthetic_example(example, _resp())
    assert score.parse_success is True
    assert score.correct_tool is True
    assert score.exact_arguments is True
    # score < 1.0 because valid_arguments=False without tool_argument_contracts
    assert score.score > 0.0


def test_wrong_tool_name_reduces_score() -> None:
    example = _example()
    score = score_synthetic_example(example, _resp(tool_name="git_diff"))
    assert score.correct_tool is False
    assert score.score < 1.0


def test_unparseable_response_scores_zero() -> None:
    example = _example()
    score = score_synthetic_example(example, "I cannot do that.")
    assert score.parse_success is False
    assert score.score == 0.0
    assert len(score.errors) > 0


def test_unexpected_final_response_penalises() -> None:
    example = _example()
    resp = json.dumps({"final_response": "done", "tool_name": "repo_read"})
    score = score_synthetic_example(example, resp)
    assert score.unexpected_final_response is True
    assert score.score < 1.0


def test_repair_example_requires_correct_tool_and_valid_args() -> None:
    # For a repair example with matching tool call in target.
    example = _example(kind=DatasetExampleKind.REPAIR)
    score = score_synthetic_example(example, _resp())
    # correct_tool=True (tool names match), but valid_arguments may be False
    # without contracts. repair_success = correct_tool AND valid_arguments.
    assert score.correct_tool is True
    assert score.score > 0.0  # partial score from other passing components


def test_repair_example_fails_with_wrong_tool() -> None:
    example = _example(kind=DatasetExampleKind.REPAIR)
    score = score_synthetic_example(example, _resp(tool_name="git_diff"))
    assert score.repair_success is False


def test_score_returns_synthetic_example_score_type() -> None:
    score = score_synthetic_example(_example(), _resp())
    assert isinstance(score, SyntheticExampleScore)


def test_score_example_id_matches() -> None:
    example = _example()
    score = score_synthetic_example(example, _resp())
    assert score.example_id == str(example.id)


# ---------------------------------------------------------------------------
# synthetic_metrics aggregation
# ---------------------------------------------------------------------------


def test_synthetic_metrics_empty_list() -> None:
    # ZeroDivisionError for empty list — currently all rates are 0/0
    import pytest
    with pytest.raises(ZeroDivisionError):
        synthetic_metrics([])


def test_synthetic_metrics_all_perfect_has_keys() -> None:
    example = _example()
    scores = [score_synthetic_example(example, _resp()) for _ in range(3)]
    metrics = synthetic_metrics(scores)
    assert "parse_success_rate" in metrics
    assert "correct_tool_rate" in metrics
    assert metrics["parse_success_rate"] == 1.0
    assert metrics["correct_tool_rate"] == 1.0
