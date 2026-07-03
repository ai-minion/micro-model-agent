"""Application workflows for staged workspace behavior evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from micro_model_agent.dataset.application.ports import (
    DatasetExampleReader,
    DatasetToolProfileSummarizer,
)
from micro_model_agent.dataset.domain.value_objects import DatasetExample
from micro_model_agent.evaluation.application._meta import _with_evaluation_metadata
from micro_model_agent.evaluation.application.ports import (
    EvaluationResultReader,
    EvaluationResultWriter,
    ModelBehaviorEvaluationSuite,
    WorkspaceStagedReviewBuilder,
    WorkspaceStagedReviewQueueWriter,
)
from micro_model_agent.execution.application.ports import ModelProvider
from micro_model_agent.shared.domain.value_objects import EvaluationResult


@dataclass(frozen=True, slots=True)
class RunWorkspaceStagedEvaluationRequest:
    """Request for staged workspace behavior evaluation."""

    run_id: str
    run_dir: Path
    dataset_path: Path
    provider_kind: str
    model_provider: ModelProvider | None
    model: str | None = None
    base_model: str | None = None
    adapter_path: Path | None = None
    max_examples: int | None = None
    output_path: Path | None = None
    default_available_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunWorkspaceStagedEvaluationResult:
    """Result returned after staged workspace behavior evaluation."""

    report_path: Path
    evaluation: EvaluationResult


@dataclass(frozen=True, slots=True)
class RunWorkspaceStagedReviewRequest:
    """Request for building a staged workspace review queue."""

    dataset_path: Path
    report_paths: tuple[Path, ...]
    output_path: Path
    simple_failure_threshold: float = 0.4
    auto_accept_threshold: float = 0.95


@dataclass(frozen=True, slots=True)
class RunWorkspaceStagedReviewBuildResult:
    """Result returned after building staged workspace review records."""

    output_path: Path
    records: list[dict[str, Any]]
    auto_triage_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class RunWorkspaceStagedReviewWriteRequest:
    """Request for writing staged workspace review records."""

    output_path: Path
    records: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RunWorkspaceStagedReviewWriteResult:
    """Result returned after writing staged workspace review records."""

    output_path: Path
    record_count: int
    auto_triage_counts: dict[str, int]


class WorkspaceStagedExampleScore(Protocol):
    """Score shape consumed by the application workspace-staged suite."""

    category: str | None
    score: float
    parse_success: bool

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-ready score record."""


type WorkspaceStagedExampleScorer = Callable[
    [DatasetExample, str],
    WorkspaceStagedExampleScore,
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

WORKSPACE_STAGED_STAGE_NAMES = (
    "read_search",
    "diagnosis",
    "patch_proposal",
    "test_selection",
    "final_summary",
)


class WorkspaceStagedEvaluationSuite:
    """Run staged workspace examples against a model provider."""

    def __init__(
        self,
        *,
        score_example: WorkspaceStagedExampleScorer,
        pass_threshold: float = 0.8,
        rubric_version: str = "legacy",
        default_available_tools: Sequence[str] = (),
    ) -> None:
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        if rubric_version not in {"legacy", "v2", "auto"}:
            raise ValueError("rubric_version must be legacy, v2, or auto")
        self.score_example = score_example
        self.pass_threshold = pass_threshold
        self.rubric_version = rubric_version
        self.default_available_tools = tuple(default_available_tools)

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
            scores.append(self.score_example(example, raw_response))

        overall_score = sum(score.score for score in scores) / len(scores)
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
                    "parse_success_rate": _workspace_staged_parse_success_rate(scores),
                    **_workspace_staged_stage_metrics(scores),
                },
                "category_metrics": _workspace_staged_category_metrics(scores),
                "examples": [score.as_record() for score in scores],
            },
        )

    def _prompt_for_example(self, example: DatasetExample) -> str:
        """Build the model prompt for one staged workspace example."""

        return (
            f"<|system|>\n{WORKSPACE_STAGED_SYSTEM_PROMPT}\n"
            f"<|user|>\n{json.dumps(self._prompt_payload(example), sort_keys=True)}\n"
            "<|assistant|>\n"
        )

    def _prompt_payload(self, example: DatasetExample) -> dict[str, object]:
        return workspace_staged_prompt_payload(
            example,
            default_available_tools=self.default_available_tools,
        )


class RunWorkspaceStagedEvaluationWorkflow:
    """Evaluate staged workspace examples against a runnable model provider."""

    def __init__(
        self,
        *,
        example_reader: DatasetExampleReader,
        behavior_suite: ModelBehaviorEvaluationSuite,
        tool_profile_summarizer: DatasetToolProfileSummarizer,
        evaluation_writer: EvaluationResultWriter,
    ) -> None:
        self.example_reader = example_reader
        self.behavior_suite = behavior_suite
        self.tool_profile_summarizer = tool_profile_summarizer
        self.evaluation_writer = evaluation_writer

    async def run(
        self,
        request: RunWorkspaceStagedEvaluationRequest,
    ) -> RunWorkspaceStagedEvaluationResult:
        """Run staged workspace behavior evaluation."""

        if request.model_provider is None:
            raise ValueError(
                "workspace-staged eval requires a runnable model, adapter, or scripted response"
            )

        examples = self.example_reader.load_dataset_examples(request.dataset_path)
        if request.max_examples is not None:
            examples = examples[: request.max_examples]
        evaluation = await self.behavior_suite.evaluate_model(
            request.model_provider,
            examples,
        )
        evaluation = _with_evaluation_metadata(
            evaluation,
            run_id=request.run_id,
            dataset_path=request.dataset_path,
            examples=examples,
            provider_kind=request.provider_kind,
            model=request.model,
            base_model=request.base_model,
            adapter_path=request.adapter_path,
            tool_profile_summarizer=self.tool_profile_summarizer,
            default_available_tools=request.default_available_tools,
        )
        report_path = self.evaluation_writer.write_evaluation_result(
            request.run_dir,
            evaluation,
            request.output_path,
        )
        return RunWorkspaceStagedEvaluationResult(
            report_path=report_path,
            evaluation=evaluation,
        )


class RunWorkspaceStagedReviewWorkflow:
    """Build and write staged workspace review queue records."""

    def __init__(
        self,
        *,
        example_reader: DatasetExampleReader,
        evaluation_reader: EvaluationResultReader,
        review_builder: WorkspaceStagedReviewBuilder,
        review_writer: WorkspaceStagedReviewQueueWriter,
    ) -> None:
        self.example_reader = example_reader
        self.evaluation_reader = evaluation_reader
        self.review_builder = review_builder
        self.review_writer = review_writer

    def build(
        self,
        request: RunWorkspaceStagedReviewRequest,
    ) -> RunWorkspaceStagedReviewBuildResult:
        """Build staged workspace review records without writing them."""

        if not request.report_paths:
            raise ValueError("at least one --report is required")

        examples = self.example_reader.load_dataset_examples(request.dataset_path)
        reports = [
            (path, self.evaluation_reader.load_evaluation_result(path))
            for path in request.report_paths
        ]
        records = self.review_builder.build_workspace_staged_review_records(
            examples=examples,
            reports=reports,
            simple_failure_threshold=request.simple_failure_threshold,
            auto_accept_threshold=request.auto_accept_threshold,
        )
        return RunWorkspaceStagedReviewBuildResult(
            output_path=request.output_path,
            records=records,
            auto_triage_counts=_auto_triage_counts(records),
        )

    def write(
        self,
        request: RunWorkspaceStagedReviewWriteRequest,
    ) -> RunWorkspaceStagedReviewWriteResult:
        """Write staged workspace review records."""

        self.review_writer.write_workspace_staged_review_records(
            request.output_path,
            request.records,
        )
        return RunWorkspaceStagedReviewWriteResult(
            output_path=request.output_path,
            record_count=len(request.records),
            auto_triage_counts=_auto_triage_counts(request.records),
        )

    def run(
        self,
        request: RunWorkspaceStagedReviewRequest,
    ) -> RunWorkspaceStagedReviewWriteResult:
        """Build and write staged workspace review records."""

        build_result = self.build(request)
        return self.write(
            RunWorkspaceStagedReviewWriteRequest(
                output_path=build_result.output_path,
                records=build_result.records,
            )
        )


def workspace_staged_prompt_payload(
    example: DatasetExample,
    *,
    default_available_tools: Sequence[str] = (),
) -> dict[str, object]:
    """Build the staged workspace prompt payload shared by eval and SFT export."""

    return {
        "goal": example.input.get("goal", ""),
        "repository_context": example.input.get("repository_context", {}),
        "available_tools": _workspace_staged_available_tools(
            example,
            default_available_tools=default_available_tools,
        ),
        "workspace_files": example.input.get("workspace_files", {}),
        "candidate_files": example.input.get("candidate_files", []),
        "observations": example.input.get("observations", []),
        "constraints": example.input.get("constraints", []),
    }


def _auto_triage_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        auto_triage = record.get("auto_triage")
        if not isinstance(auto_triage, dict):
            continue
        decision = str(auto_triage.get("decision"))
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def _workspace_staged_available_tools(
    example: DatasetExample,
    *,
    default_available_tools: Sequence[str],
) -> list[str]:
    input_tools = example.input.get("available_tools")
    if isinstance(input_tools, list) and all(isinstance(tool, str) for tool in input_tools):
        return list(dict.fromkeys(input_tools))

    metadata_profile = example.metadata.get("tool_profile")
    if isinstance(metadata_profile, dict):
        metadata_tools = metadata_profile.get("available_tools")
        if isinstance(metadata_tools, list) and all(
            isinstance(tool, str) for tool in metadata_tools
        ):
            return list(dict.fromkeys(metadata_tools))

    return list(dict.fromkeys(default_available_tools))


def _workspace_staged_parse_success_rate(
    scores: list[WorkspaceStagedExampleScore],
) -> float:
    return sum(1.0 for score in scores if score.parse_success) / len(scores)


def _workspace_staged_stage_metrics(
    scores: list[WorkspaceStagedExampleScore],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    records = [score.as_record() for score in scores]
    for stage_name in WORKSPACE_STAGED_STAGE_NAMES:
        stage_scores = [
            stage
            for record in records
            for stage in _workspace_staged_record_stages(record)
            if stage.get("name") == stage_name
        ]
        if not stage_scores:
            continue
        metrics[f"{stage_name}_score"] = sum(
            _workspace_staged_stage_score(stage) for stage in stage_scores
        ) / len(stage_scores)
        metrics[f"{stage_name}_pass_rate"] = sum(
            1.0 for stage in stage_scores if stage.get("passed") is True
        ) / len(stage_scores)
    return metrics


def _workspace_staged_category_metrics(
    scores: list[WorkspaceStagedExampleScore],
) -> dict[str, dict[str, float]]:
    grouped_scores: dict[str, list[WorkspaceStagedExampleScore]] = {}
    for score in scores:
        grouped_scores.setdefault(score.category or "uncategorized", []).append(score)

    return {
        category: {
            "example_count": float(len(category_scores)),
            "score": sum(score.score for score in category_scores) / len(category_scores),
            "parse_success_rate": _workspace_staged_parse_success_rate(category_scores),
            **_workspace_staged_stage_metrics(category_scores),
        }
        for category, category_scores in sorted(grouped_scores.items())
    }


def _workspace_staged_record_stages(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    stages = record.get("stages")
    if not isinstance(stages, list):
        return []
    return [stage for stage in stages if isinstance(stage, dict)]


def _workspace_staged_stage_score(stage: dict[str, Any]) -> float:
    value = stage.get("score")
    return float(value) if isinstance(value, int | float) else 0.0
