"""Tests for behavioral synthetic evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.infrastructure.dataset_store import load_dataset_examples
from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
from micro_model_agent.infrastructure.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.synthetic_evaluation import SyntheticBehaviorEvaluationSuite


def _response(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def test_behavioral_synthetic_evaluator_scores_expected_tool_calls() -> None:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(4))
    model = ScriptedModelProvider([_response(example.target) for example in examples])

    result = asyncio.run(SyntheticBehaviorEvaluationSuite().evaluate_model(model, examples))

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["metrics"]["parse_success_rate"] == 1.0
    assert result.details["metrics"]["correct_tool_rate"] == 1.0
    assert result.details["metrics"]["valid_argument_rate"] == 1.0
    assert result.details["metrics"]["safe_refusal_rate"] == 1.0


def test_behavioral_synthetic_evaluator_fails_unsafe_response_without_refusal() -> None:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(3))
    unsafe_example = examples[2]
    model = ScriptedModelProvider(
        [
            _response(
                {
                    "tool_name": "repo.read",
                    "arguments": {"files": [{"path": "docs/architecture.md"}]},
                }
            )
        ]
    )

    result = asyncio.run(SyntheticBehaviorEvaluationSuite().evaluate_model(model, [unsafe_example]))

    assert result.passed is False
    assert result.details["unsafe_failure_count"] == 1
    assert result.details["examples"][0]["safe_refusal"] is False


def test_behavioral_synthetic_evaluator_scores_safe_refusal_without_tool_call() -> None:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(3))
    unsafe_example = examples[2]
    model = ScriptedModelProvider([_response({"refusal": "I cannot help with that."})])

    result = asyncio.run(SyntheticBehaviorEvaluationSuite().evaluate_model(model, [unsafe_example]))

    assert result.passed is True
    assert result.score == 1.0
    detail = result.details["examples"][0]
    assert detail["correct_tool"] is True
    assert detail["valid_arguments"] is True
    assert detail["exact_arguments"] is True
    assert detail["safe_refusal"] is True


def test_behavioral_synthetic_evaluator_records_parse_errors() -> None:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(1))
    model = ScriptedModelProvider(["not json"])

    result = asyncio.run(SyntheticBehaviorEvaluationSuite().evaluate_model(model, examples))

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["metrics"]["parse_success_rate"] == 0.0
    assert "model response must be a JSON object" in result.details["examples"][0]["errors"][0]


def test_behavioral_synthetic_evaluator_reports_category_metrics() -> None:
    examples = load_dataset_examples(Path("examples/synthetic-data/held-out.behavior.jsonl"))
    model = ScriptedModelProvider([_response(example.target) for example in examples])

    result = asyncio.run(SyntheticBehaviorEvaluationSuite().evaluate_model(model, examples))

    assert result.passed is True
    assert result.details["example_count"] == 8
    category_metrics = result.details["category_metrics"]
    assert category_metrics["valid_tool_call"]["example_count"] == 1.0
    assert category_metrics["documentation_grounded_retrieval"]["score"] == 1.0
    assert set(category_metrics) >= {
        "bad_json",
        "final_response_misuse",
        "invalid_arguments",
        "repair_behavior",
        "safe_refusal",
        "wrong_tool",
    }
