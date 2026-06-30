"""Behavioral evaluation for trace-derived MicroModelAgent examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from micro_model_agent.application.ports import ModelProvider
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import DatasetExample
from micro_model_agent.infrastructure.dataset_metadata import tool_profile_for_example
from micro_model_agent.infrastructure.evaluation_response_parsing import (
    json_object_from_response,
)
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS


@dataclass(frozen=True, slots=True)
class TraceExampleScore:
    """Scorecard for one held-out trace-derived example."""

    example_id: str
    trace_id: str | None
    category: str | None
    raw_response: str
    parsed_response: dict[str, Any] | None
    tool_profile: dict[str, Any]
    score: float
    parse_success: bool
    final_response_match: bool
    patch_match: bool
    tool_history_match: bool
    errors: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-ready detail record for reports."""

        return {
            "example_id": self.example_id,
            "trace_id": self.trace_id,
            "category": self.category,
            "score": self.score,
            "parse_success": self.parse_success,
            "final_response_match": self.final_response_match,
            "patch_match": self.patch_match,
            "tool_history_match": self.tool_history_match,
            "errors": list(self.errors),
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "tool_profile": self.tool_profile,
        }


class TraceBehaviorEvaluationSuite:
    """Run held-out trace-derived examples against a model provider."""

    def __init__(self, pass_threshold: float = 0.8) -> None:
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        self.pass_threshold = pass_threshold

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

        scores: list[TraceExampleScore] = []
        for example in examples:
            raw_response = await model_provider.complete(self._prompt_for_example(example))
            scores.append(self._score_example(example, raw_response))

        overall_score = sum(score.score for score in scores) / len(scores)
        return EvaluationResult(
            passed=overall_score >= self.pass_threshold,
            summary=f"trace behavior eval scored {overall_score:.2f} over {len(scores)} example(s)",
            score=overall_score,
            details={
                "example_count": len(scores),
                "pass_threshold": self.pass_threshold,
                "metrics": self._metrics(scores),
                "category_metrics": self._category_metrics(scores),
                "examples": [score.as_record() for score in scores],
            },
        )

    def _prompt_for_example(self, example: DatasetExample) -> str:
        """Build the model prompt for one held-out trace example."""

        payload = {
            "goal": example.input.get("goal", ""),
            "available_tools": tool_profile_for_example(
                example,
                default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
            )["available_tools"],
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

    def _score_example(
        self,
        example: DatasetExample,
        raw_response: str,
    ) -> TraceExampleScore:
        errors: list[str] = []
        try:
            response = json_object_from_response(raw_response)
        except ValueError as exc:
            return TraceExampleScore(
                example_id=str(example.id),
                trace_id=self._trace_id(example),
                category=self._category(example),
                raw_response=raw_response,
                parsed_response=None,
                tool_profile=tool_profile_for_example(
                    example,
                    default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
                ),
                score=0.0,
                parse_success=False,
                final_response_match=False,
                patch_match=False,
                tool_history_match=False,
                errors=(str(exc),),
            )

        checks: list[bool] = []
        final_response_match = self._final_response_match(example, response)
        if self._expected_final_response(example) is not None:
            checks.append(final_response_match)
            if not final_response_match:
                errors.append("final_response did not match expected trace response")

        patch_match = self._patch_match(example, response)
        if self._expected_patch(example) is not None:
            checks.append(patch_match)
            if not patch_match:
                errors.append("patch did not match expected trace patch")

        tool_history_match = self._tool_history_match(example, response)
        if self._expected_tool_names(example):
            checks.append(tool_history_match)
            if not tool_history_match:
                errors.append("tool_history did not include expected tool calls")

        if not checks:
            errors.append("trace example has no scorable target fields")

        score = sum(1.0 for check in checks if check) / len(checks) if checks else 0.0
        return TraceExampleScore(
            example_id=str(example.id),
            trace_id=self._trace_id(example),
            category=self._category(example),
            raw_response=raw_response,
            parsed_response=response,
            tool_profile=tool_profile_for_example(
                example,
                default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
            ),
            score=score,
            parse_success=True,
            final_response_match=final_response_match,
            patch_match=patch_match,
            tool_history_match=tool_history_match,
            errors=tuple(errors),
        )

    def _expected_final_response(self, example: DatasetExample) -> str | None:
        value = example.target.get("final_response") or example.target.get("summary")
        return value if isinstance(value, str) and value.strip() else None

    def _expected_patch(self, example: DatasetExample) -> str | None:
        value = example.target.get("patch")
        return value if isinstance(value, str) and value.strip() else None

    def _final_response_match(self, example: DatasetExample, response: dict[str, Any]) -> bool:
        expected = self._expected_final_response(example)
        if expected is None:
            return True
        actual = response.get("final_response", response.get("response"))
        if not isinstance(actual, str):
            return False
        return self._similarity(expected, actual) >= 0.8

    def _patch_match(self, example: DatasetExample, response: dict[str, Any]) -> bool:
        expected = self._expected_patch(example)
        if expected is None:
            return True
        actual = response.get("patch")
        if not isinstance(actual, str):
            return False
        return self._normalize_text(expected) == self._normalize_text(actual)

    def _tool_history_match(self, example: DatasetExample, response: dict[str, Any]) -> bool:
        expected_tools = self._expected_tool_names(example)
        if not expected_tools:
            return True
        actual_tools = self._tool_names_from_response(response)
        return actual_tools[: len(expected_tools)] == expected_tools

    def _expected_tool_names(self, example: DatasetExample) -> list[str]:
        names: list[str] = []
        history = example.input.get("tool_history")
        if not isinstance(history, list):
            return names
        for item in history:
            if not isinstance(item, dict):
                continue
            tool_call = item.get("tool_call")
            if isinstance(tool_call, dict) and isinstance(tool_call.get("tool_name"), str):
                names.append(tool_call["tool_name"])
        return names

    def _tool_names_from_response(self, response: dict[str, Any]) -> list[str]:
        history = response.get("tool_history")
        if isinstance(history, list):
            names: list[str] = []
            for item in history:
                if not isinstance(item, dict):
                    continue
                tool_name = item.get("tool_name")
                tool_call = item.get("tool_call")
                if isinstance(tool_name, str):
                    names.append(tool_name)
                elif isinstance(tool_call, dict) and isinstance(tool_call.get("tool_name"), str):
                    names.append(tool_call["tool_name"])
            return names

        tool_name = response.get("tool_name")
        return [tool_name] if isinstance(tool_name, str) else []

    def _metrics(self, scores: list[TraceExampleScore]) -> dict[str, float]:
        total = len(scores)
        return {
            "parse_success_rate": self._rate(scores, "parse_success", total),
            "final_response_match_rate": self._rate(scores, "final_response_match", total),
            "patch_match_rate": self._rate(scores, "patch_match", total),
            "tool_history_match_rate": self._rate(scores, "tool_history_match", total),
        }

    def _rate(self, scores: list[TraceExampleScore], field_name: str, total: int) -> float:
        return sum(1.0 for score in scores if bool(getattr(score, field_name))) / total

    def _category_metrics(self, scores: list[TraceExampleScore]) -> dict[str, dict[str, float]]:
        grouped_scores: dict[str, list[TraceExampleScore]] = {}
        for score in scores:
            grouped_scores.setdefault(score.category or "uncategorized", []).append(score)

        return {
            category: {
                "example_count": float(len(category_scores)),
                "score": sum(score.score for score in category_scores) / len(category_scores),
                **self._metrics(category_scores),
            }
            for category, category_scores in sorted(grouped_scores.items())
        }

    def _similarity(self, expected: str, actual: str) -> float:
        return SequenceMatcher(
            None,
            self._normalize_text(expected).casefold(),
            self._normalize_text(actual).casefold(),
        ).ratio()

    def _normalize_text(self, value: str) -> str:
        return " ".join(value.split())

    def _trace_id(self, example: DatasetExample) -> str | None:
        trace_id = example.metadata.get("trace_id")
        return trace_id if isinstance(trace_id, str) else None

    def _category(self, example: DatasetExample) -> str | None:
        category = example.metadata.get("category")
        return category if isinstance(category, str) else None
