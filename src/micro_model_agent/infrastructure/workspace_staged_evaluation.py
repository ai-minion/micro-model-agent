"""Staged workspace reasoning evaluation for coding-agent behavior."""

from __future__ import annotations

import json

from micro_model_agent.application.ports import ModelProvider
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import DatasetExample
from micro_model_agent.infrastructure.dataset_metadata import tool_profile_for_example
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS
from micro_model_agent.infrastructure.workspace_staged_review import (
    LocalWorkspaceStagedReviewBuilder,
    LocalWorkspaceStagedReviewQueueWriter,
    build_workspace_staged_review_records,
)
from micro_model_agent.infrastructure.workspace_staged_rubrics import (
    STAGE_NAMES,
    WorkspaceStagedExampleScore,
    WorkspaceStagedRubric,
    WorkspaceStageScore,
)

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

WORKSPACE_STAGED_SYSTEM_PROMPT = (
    "You are MicroModelAgent evaluating a dry-run coding task. "
    "Respond with exactly one JSON object and no markdown. "
    "Do not claim that patches were applied. "
    "Use these top-level keys: read_search, diagnosis, patch_proposal, "
    "test_selection, final_summary. "
    "read_search should name files and searches needed. "
    "diagnosis should explain the likely cause and plan before patching. "
    "patch_proposal should include changed_files and a dry-run patch sketch. "
    "test_selection should list focused commands. "
    "final_summary should summarize the dry-run proposal, files, tests, and risks."
)


class WorkspaceStagedEvaluationSuite:
    """Evaluate workspace reasoning in staged, dry-run-only responses."""

    def __init__(self, pass_threshold: float = 0.8, rubric_version: str = "legacy") -> None:
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        if rubric_version not in {"legacy", "v2", "auto"}:
            raise ValueError("rubric_version must be legacy, v2, or auto")
        self.pass_threshold = pass_threshold
        self.rubric_version = rubric_version
        self.rubric = WorkspaceStagedRubric(rubric_version=rubric_version)

    async def evaluate_model(
        self,
        model_provider: ModelProvider,
        examples: list[DatasetExample],
    ) -> EvaluationResult:
        """Score model completions against staged workspace expectations."""

        if not examples:
            return EvaluationResult(
                passed=False,
                summary="staged workspace eval has no examples",
                score=0.0,
                details={"example_count": 0, "errors": ["dataset contains no examples"]},
            )

        scores: list[WorkspaceStagedExampleScore] = []
        for example in examples:
            raw_response = await model_provider.complete(self._prompt_for_example(example))
            scores.append(self.rubric.score_example(example, raw_response))

        overall_score = sum(score.score for score in scores) / len(scores)
        stage_metrics = self.rubric.stage_metrics(scores)
        return EvaluationResult(
            passed=overall_score >= self.pass_threshold,
            summary=(
                f"staged workspace eval scored {overall_score:.2f} "
                f"over {len(scores)} example(s)"
            ),
            score=overall_score,
            details={
                "example_count": len(scores),
                "pass_threshold": self.pass_threshold,
                "rubric_version": self.rubric_version,
                "metrics": {
                    "parse_success_rate": self.rubric.parse_success_rate(scores),
                    **stage_metrics,
                },
                "category_metrics": self.rubric.category_metrics(scores),
                "examples": [score.as_record() for score in scores],
            },
        )

    def _prompt_for_example(self, example: DatasetExample) -> str:
        return (
            f"<|system|>\n{WORKSPACE_STAGED_SYSTEM_PROMPT}\n"
            f"<|user|>\n{json.dumps(workspace_staged_prompt_payload(example), sort_keys=True)}\n"
            "<|assistant|>\n"
        )


def workspace_staged_prompt_payload(example: DatasetExample) -> dict[str, object]:
    """Build the staged workspace prompt payload shared by eval and SFT export."""

    return {
        "goal": example.input.get("goal", ""),
        "repository_context": example.input.get("repository_context", {}),
        "available_tools": tool_profile_for_example(
            example,
            default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
        )["available_tools"],
        "workspace_files": example.input.get("workspace_files", {}),
        "candidate_files": example.input.get("candidate_files", []),
        "observations": example.input.get("observations", []),
        "constraints": example.input.get("constraints", []),
    }


def is_workspace_staged_example(example: DatasetExample) -> bool:
    """Return whether an evaluation example uses the staged workspace surface."""

    return isinstance(example.input.get("workspace_files"), dict) or isinstance(
        example.target.get("gold_response"),
        dict,
    )
