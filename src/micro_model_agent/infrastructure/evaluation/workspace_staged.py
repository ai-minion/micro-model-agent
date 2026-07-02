"""Staged workspace reasoning evaluation for coding-agent behavior."""

from __future__ import annotations

from typing import cast

from micro_model_agent.application.evaluation_rubrics.workspace_staged import (
    STAGE_NAMES,
    WorkspaceStagedExampleScore,
    WorkspaceStagedRubric,
    WorkspaceStageScore,
)
from micro_model_agent.application.evaluation_workflows import (
    WORKSPACE_STAGED_SYSTEM_PROMPT as APPLICATION_WORKSPACE_STAGED_SYSTEM_PROMPT,
)
from micro_model_agent.application.evaluation_workflows import (
    WorkspaceStagedEvaluationSuite as ApplicationWorkspaceStagedEvaluationSuite,
)
from micro_model_agent.application.evaluation_workflows import (
    WorkspaceStagedExampleScorer,
)
from micro_model_agent.application.evaluation_workflows import (
    workspace_staged_prompt_payload as application_workspace_staged_prompt_payload,
)
from micro_model_agent.domain.datasets import DatasetExample
from micro_model_agent.infrastructure.evaluation.workspace_staged_review import (
    LocalWorkspaceStagedReviewBuilder,
    LocalWorkspaceStagedReviewQueueWriter,
    build_workspace_staged_review_records,
)
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS

__all__ = [
    "LocalWorkspaceStagedReviewBuilder",
    "LocalWorkspaceStagedReviewQueueWriter",
    "STAGE_NAMES",
    "WorkspaceStagedEvaluationSuite",
    "WorkspaceStagedExampleScore",
    "WorkspaceStageScore",
    "build_workspace_staged_review_records",
    "is_workspace_staged_example",
    "workspace_staged_prompt_payload",
]

WORKSPACE_STAGED_SYSTEM_PROMPT = APPLICATION_WORKSPACE_STAGED_SYSTEM_PROMPT


class WorkspaceStagedEvaluationSuite(ApplicationWorkspaceStagedEvaluationSuite):
    """Staged workspace evaluator wired to infrastructure rubric contracts."""

    def __init__(self, pass_threshold: float = 0.8, rubric_version: str = "legacy") -> None:
        default_available_tools = tuple(TOOL_ARGUMENT_CONTRACTS)
        self.rubric = WorkspaceStagedRubric(
            rubric_version=rubric_version,
            default_available_tools=default_available_tools,
        )
        super().__init__(
            score_example=cast(
                WorkspaceStagedExampleScorer,
                self.rubric.score_example,
            ),
            pass_threshold=pass_threshold,
            rubric_version=rubric_version,
            default_available_tools=default_available_tools,
        )


def workspace_staged_prompt_payload(example: DatasetExample) -> dict[str, object]:
    """Build the staged workspace prompt payload shared by eval and SFT export."""

    return application_workspace_staged_prompt_payload(
        example,
        default_available_tools=tuple(TOOL_ARGUMENT_CONTRACTS),
    )


def is_workspace_staged_example(example: DatasetExample) -> bool:
    """Return whether an evaluation example uses the staged workspace surface."""

    return isinstance(example.input.get("workspace_files"), dict) or isinstance(
        example.target.get("gold_response"),
        dict,
    )
