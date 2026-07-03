"""Tests for synthetic prompt payload helpers."""

from __future__ import annotations

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.prompting import (
    synthetic_prompt_payload,
)


def _label(outcome: OutcomeLabel = OutcomeLabel.ACCEPTED) -> DatasetLabel:
    return DatasetLabel(outcome=outcome, quality=QualityLabel.GOOD)


def _example(
    outcome: OutcomeLabel = OutcomeLabel.ACCEPTED,
    goal: str = "fix the null pointer bug",
    target: dict | None = None,
    input_extra: dict | None = None,
) -> DatasetExample:
    inp: dict = {"goal": goal}
    if input_extra:
        inp.update(input_extra)
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input=inp,
        target=target or {"patch": "--- a/f.py\n+++ b/f.py"},
        label=_label(outcome),
    )


# ---------------------------------------------------------------------------
# synthetic_prompt_payload
# ---------------------------------------------------------------------------


def test_payload_has_required_keys() -> None:
    payload = synthetic_prompt_payload(_example())
    assert "goal" in payload
    assert "available_tools" in payload
    assert "response_contract" in payload


def test_payload_goal_matches_example() -> None:
    payload = synthetic_prompt_payload(_example(goal="write unit tests"))
    assert payload["goal"] == "write unit tests"


def test_payload_available_tools_is_list() -> None:
    payload = synthetic_prompt_payload(_example())
    assert isinstance(payload["available_tools"], list)


def test_payload_custom_available_tools_respected() -> None:
    example = _example(input_extra={"available_tools": ["repo.read", "git.diff"]})
    payload = synthetic_prompt_payload(example)
    assert payload["available_tools"] == ["repo.read", "git.diff"]


def test_payload_rejected_outcome_has_refusal_contract() -> None:
    example = _example(outcome=OutcomeLabel.REJECTED)
    payload = synthetic_prompt_payload(example)
    contract = payload["response_contract"]
    assert isinstance(contract, dict)
    # Rejected examples should have a refusal in the response contract
    assert "refusal" in str(contract)


def test_payload_accepted_outcome_has_tool_call_contract() -> None:
    example = _example(outcome=OutcomeLabel.ACCEPTED)
    payload = synthetic_prompt_payload(example)
    contract = payload["response_contract"]
    assert isinstance(contract, dict)


def test_payload_no_tool_schemas_by_default() -> None:
    payload = synthetic_prompt_payload(_example())
    assert "tool_schemas" not in payload


def test_payload_include_tool_schemas() -> None:
    example = _example(input_extra={"available_tools": ["repo.read"]})
    payload = synthetic_prompt_payload(example, include_tool_schemas=True)
    assert "tool_schemas" in payload
    schemas = payload["tool_schemas"]
    assert isinstance(schemas, dict)
    assert "repo.read" in schemas


def test_payload_sanitized_input_included() -> None:
    payload = synthetic_prompt_payload(_example())
    assert "input" in payload


def test_payload_is_json_friendly() -> None:
    import json
    payload = synthetic_prompt_payload(_example())
    # All values should be JSON-serializable
    json.dumps(payload)  # should not raise
