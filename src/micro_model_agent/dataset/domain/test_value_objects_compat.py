"""Tests for dataset domain contracts."""

from __future__ import annotations

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)


def test_dataset_example_records_good_tool_use_example() -> None:
    label = DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)

    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "Find repository docs", "tools": ["repo.search"]},
        target={"tool_name": "repo.search", "arguments": {"query": "docs"}},
        label=label,
        tool_schema_version="v1",
    )

    assert example.kind is DatasetExampleKind.TOOL_USE
    assert example.label.quality is QualityLabel.GOOD
    assert example.tool_schema_version == "v1"


def test_dataset_example_can_label_bad_failure_modes() -> None:
    label = DatasetLabel(
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
        failure_modes=(FailureMode.UNSAFE_PATH_REQUESTED,),
    )

    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "Read a secret file"},
        target={"tool_name": "repo.read", "arguments": {"path": "../secret"}},
        label=label,
    )

    assert example.label.failure_modes == (FailureMode.UNSAFE_PATH_REQUESTED,)
