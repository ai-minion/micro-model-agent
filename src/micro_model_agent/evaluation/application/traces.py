"""Application workflows for trace-derived behavior evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from micro_model_agent.evaluation.application._meta import _with_evaluation_metadata
from micro_model_agent.evaluation.application.ports import (
    EvaluationResultWriter,
    ModelBehaviorEvaluationSuite,
)
from micro_model_agent.dataset.application.ports import (
    DatasetExampleReader,
    DatasetToolProfileSummarizer,
)
from micro_model_agent.execution.application.ports import ModelProvider
from micro_model_agent.shared.domain.value_objects import EvaluationResult
from micro_model_agent.dataset.domain.value_objects import DatasetExample


@dataclass(frozen=True, slots=True)
class RunTraceEvaluationRequest:
    """Request for trace-derived behavior evaluation."""

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
class RunTraceEvaluationResult:
    """Result returned after trace-derived behavior evaluation."""

    report_path: Path
    evaluation: EvaluationResult


class TraceBehaviorExampleScore(Protocol):
    """Score shape consumed by the application trace evaluation suite."""

    category: str | None
    score: float
    parse_success: bool
    final_response_match: bool
    patch_match: bool
    tool_history_match: bool

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-ready score record."""


type TraceExampleScorer = Callable[
    [DatasetExample, str],
    TraceBehaviorExampleScore,
]


class TraceBehaviorEvaluationSuite:
    """Run held-out trace-derived examples against a model provider."""

    def __init__(
        self,
        *,
        score_example: TraceExampleScorer,
        pass_threshold: float = 0.8,
        default_available_tools: Sequence[str] = (),
    ) -> None:
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        self.score_example = score_example
        self.pass_threshold = pass_threshold
        self.default_available_tools = tuple(default_available_tools)

    async def evaluate_model(
        self,
        model_provider: ModelProvider,
        examples: list[DatasetExample],
    ) -> EvaluationResult:
        """Score model completions against held-out trace expectations."""

        if not examples:
            return EvaluationResult(
                passed=False,
                summary="trace behavior eval has no examples",
                score=0.0,
                details={"example_count": 0, "errors": ["dataset contains no examples"]},
            )

        scores: list[TraceBehaviorExampleScore] = []
        for example in examples:
            raw_response = await model_provider.complete(self._prompt_for_example(example))
            scores.append(self.score_example(example, raw_response))

        overall_score = sum(score.score for score in scores) / len(scores)
        return EvaluationResult(
            passed=overall_score >= self.pass_threshold,
            summary=f"trace behavior eval scored {overall_score:.2f} over {len(scores)} example(s)",
            score=overall_score,
            details={
                "example_count": len(scores),
                "pass_threshold": self.pass_threshold,
                "metrics": _trace_metrics(scores),
                "category_metrics": _trace_category_metrics(scores),
                "examples": [score.as_record() for score in scores],
            },
        )

    def _prompt_for_example(self, example: DatasetExample) -> str:
        """Build the model prompt for one held-out trace example."""

        payload = {
            "goal": example.input.get("goal", ""),
            "available_tools": _trace_available_tools(
                example,
                default_available_tools=self.default_available_tools,
            ),
            "retrieved_context": example.input.get("retrieved_context", {}),
            "tool_history": example.input.get("tool_history", []),
            "steps": example.input.get("steps", []),
        }
        system_prompt = (
            "You are MicroModelAgent replaying a held-out workflow trace. "
            "Respond with exactly one JSON object and no markdown. "
            "Include final_response when the task is complete, "
            "patch when a code change is required, "
            "and tool_history when tool calls were part of the workflow."
        )
        return (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{json.dumps(payload, sort_keys=True)}\n"
            "<|assistant|>\n"
        )


class RunTraceEvaluationWorkflow:
    """Evaluate trace-derived examples against a runnable model provider."""

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

    async def run(self, request: RunTraceEvaluationRequest) -> RunTraceEvaluationResult:
        """Run trace-derived behavior evaluation."""

        if request.model_provider is None:
            raise ValueError("trace eval requires a runnable model, adapter, or scripted response")

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
        return RunTraceEvaluationResult(
            report_path=report_path,
            evaluation=evaluation,
        )


def _trace_available_tools(
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


def _trace_metrics(
    scores: list[TraceBehaviorExampleScore],
) -> dict[str, float]:
    """Aggregate trace example score flags into report metrics."""

    total = len(scores)
    return {
        "parse_success_rate": _trace_rate(scores, "parse_success", total),
        "final_response_match_rate": _trace_rate(
            scores,
            "final_response_match",
            total,
        ),
        "patch_match_rate": _trace_rate(scores, "patch_match", total),
        "tool_history_match_rate": _trace_rate(scores, "tool_history_match", total),
    }


def _trace_category_metrics(
    scores: list[TraceBehaviorExampleScore],
) -> dict[str, dict[str, float]]:
    """Aggregate trace example score flags by category."""

    grouped_scores: dict[str, list[TraceBehaviorExampleScore]] = {}
    for score in scores:
        grouped_scores.setdefault(score.category or "uncategorized", []).append(score)

    return {
        category: {
            "example_count": float(len(category_scores)),
            "score": sum(score.score for score in category_scores) / len(category_scores),
            **_trace_metrics(category_scores),
        }
        for category, category_scores in sorted(grouped_scores.items())
    }


def _trace_rate(
    scores: list[TraceBehaviorExampleScore],
    field_name: str,
    total: int,
) -> float:
    return sum(1.0 for score in scores if bool(getattr(score, field_name))) / total
