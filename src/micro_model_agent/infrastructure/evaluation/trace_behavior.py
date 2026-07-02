"""Compatibility adapters for trace-derived evaluation."""

from __future__ import annotations

from typing import cast

from micro_model_agent.application.evaluation_rubrics.trace import (
    TraceExampleScore,
    TraceRubric,
    expected_trace_final_response,
    expected_trace_patch,
    expected_trace_tool_names,
    normalize_trace_text,
    score_trace_example,
    trace_category,
    trace_final_response_match,
    trace_id,
    trace_patch_match,
    trace_similarity,
    trace_tool_history_match,
    trace_tool_names_from_response,
)
from micro_model_agent.application.evaluation_workflows import (
    TraceBehaviorEvaluationSuite as ApplicationTraceBehaviorEvaluationSuite,
)
from micro_model_agent.application.evaluation_workflows import TraceExampleScorer
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS

__all__ = [
    "TraceBehaviorEvaluationSuite",
    "TraceExampleScore",
    "TraceRubric",
    "expected_trace_final_response",
    "expected_trace_patch",
    "expected_trace_tool_names",
    "normalize_trace_text",
    "score_trace_example",
    "trace_category",
    "trace_final_response_match",
    "trace_id",
    "trace_patch_match",
    "trace_similarity",
    "trace_tool_history_match",
    "trace_tool_names_from_response",
]


class TraceBehaviorEvaluationSuite(ApplicationTraceBehaviorEvaluationSuite):
    """Trace behavior evaluator wired to infrastructure scoring defaults."""

    def __init__(self, pass_threshold: float = 0.8) -> None:
        default_available_tools = tuple(TOOL_ARGUMENT_CONTRACTS)
        self.rubric = TraceRubric(default_available_tools=default_available_tools)
        super().__init__(
            score_example=cast(TraceExampleScorer, self.rubric.score_example),
            pass_threshold=pass_threshold,
            default_available_tools=default_available_tools,
        )
