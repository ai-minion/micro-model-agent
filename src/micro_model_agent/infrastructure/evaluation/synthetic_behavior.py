"""Compatibility adapters for behavioral synthetic evaluation."""

from __future__ import annotations

from typing import cast

from micro_model_agent.application.evaluation import (
    SyntheticBehaviorEvaluationSuite as ApplicationSyntheticBehaviorEvaluationSuite,
)
from micro_model_agent.application.evaluation import (
    SyntheticExampleScorer,
)
from micro_model_agent.application.evaluation_synthetic_rubric import (
    SyntheticExampleScore,
    SyntheticRubric,
)
from micro_model_agent.infrastructure.evaluation.trace_behavior import (
    TraceBehaviorEvaluationSuite,
    TraceExampleScore,
)
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS

__all__ = [
    "SyntheticBehaviorEvaluationSuite",
    "SyntheticExampleScore",
    "TraceBehaviorEvaluationSuite",
    "TraceExampleScore",
]


class SyntheticBehaviorEvaluationSuite(ApplicationSyntheticBehaviorEvaluationSuite):
    """Synthetic behavior evaluator wired to infrastructure scoring contracts."""

    def __init__(self, pass_threshold: float = 0.8) -> None:
        default_available_tools = tuple(TOOL_ARGUMENT_CONTRACTS)
        self.rubric = SyntheticRubric(
            tool_argument_contracts=TOOL_ARGUMENT_CONTRACTS,
            default_available_tools=default_available_tools,
        )
        super().__init__(
            score_example=cast(SyntheticExampleScorer, self.rubric.score_example),
            pass_threshold=pass_threshold,
            default_available_tools=default_available_tools,
        )
