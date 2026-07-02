"""Tests for pure evaluation rubric helpers."""

from __future__ import annotations

import json
from typing import Any

from micro_model_agent.application.evaluation_synthetic_rubric import SyntheticRubric
from micro_model_agent.application.evaluation_trace_rubric import TraceRubric
from micro_model_agent.application.evaluation_workspace_staged_rubric import (
    WorkspaceStagedRubric,
)
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)


class _RepoReadContract:
    @classmethod
    def model_validate(cls, obj: Any) -> Any:
        if not isinstance(obj.get("files"), list):
            raise ValueError("files must be a list")
        return obj


def test_synthetic_rubric_scores_without_model_or_filesystem() -> None:
    example = _example(
        input={"goal": "Read the README", "available_tools": ["repo.read"]},
        target={
            "tool_name": "repo.read",
            "arguments": {"files": [{"path": "README.md"}]},
        },
        source="synthetic:test",
    )
    rubric = SyntheticRubric(
        tool_argument_contracts={"repo.read": _RepoReadContract},
        default_available_tools=("repo.read",),
    )

    score = rubric.score_example(example, json.dumps(example.target))

    assert score.score == 1.0
    assert score.tool_profile["available_tools"] == ["repo.read"]


def test_trace_rubric_scores_without_model_or_filesystem() -> None:
    example = _example(
        input={
            "goal": "Replay the trace",
            "tool_history": [{"tool_call": {"tool_name": "repo.read"}}],
        },
        target={"final_response": "Read README.md", "patch": "diff --git a/README.md"},
        source="trace:test",
    )
    rubric = TraceRubric(default_available_tools=("repo.read",))

    score = rubric.score_example(
        example,
        json.dumps(
            {
                "final_response": "Read README.md",
                "patch": "diff --git a/README.md",
                "tool_history": [{"tool_name": "repo.read"}],
            }
        ),
    )

    assert score.score == 1.0
    assert score.tool_history_match is True


def test_workspace_staged_rubric_scores_without_model_or_filesystem() -> None:
    example = _example(
        input={"goal": "Plan a dry-run fix", "available_tools": ["repo.search"]},
        target={
            "stages": {
                "read_search": {
                    "required_files": ["src/app.py"],
                    "required_queries": ["failing test"],
                }
            }
        },
        source="workspace:test",
    )
    rubric = WorkspaceStagedRubric(default_available_tools=("repo.search",))

    score = rubric.score_example(
        example,
        json.dumps(
            {
                "read_search": {
                    "files": ["src/app.py"],
                    "queries": ["failing test"],
                }
            }
        ),
    )

    assert score.score == 1.0
    assert score.stages[0].passed is True


def _example(
    *,
    input: dict[str, object],
    target: dict[str, object],
    source: str,
) -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.EVALUATION,
        input=input,
        target=target,
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        source=source,
    )
