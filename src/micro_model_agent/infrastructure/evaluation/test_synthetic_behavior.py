"""Tests for behavioral synthetic evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.domain.datasets import DatasetExample, OutcomeLabel
from micro_model_agent.infrastructure.datasets.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.evaluation.synthetic_behavior import (
    SyntheticBehaviorEvaluationSuite,
    TraceBehaviorEvaluationSuite,
)
from micro_model_agent.infrastructure.models.fake import ScriptedModelProvider
from micro_model_agent.infrastructure.persistence.dataset_store import load_dataset_examples


def _response(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def _first_rejected_example(count: int = 12) -> DatasetExample:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(count))
    return next(example for example in examples if example.label.outcome is OutcomeLabel.REJECTED)


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
    assert result.details["examples"][0]["tool_profile"]["available_tools"] == examples[
        0
    ].input.get("available_tools", [])


def test_behavioral_synthetic_evaluator_fails_unsafe_response_without_refusal() -> None:
    unsafe_example = _first_rejected_example()
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
    unsafe_example = _first_rejected_example()
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
    assert result.details["example_count"] == 15
    assert result.details["metrics"]["repair_success_rate"] == 1.0
    assert result.details["unsafe_failure_count"] == 0
    category_metrics = result.details["category_metrics"]
    assert category_metrics["valid_tool_call"]["example_count"] == 1.0
    assert category_metrics["documentation_grounded_retrieval"]["score"] == 1.0
    assert set(category_metrics) >= {
        "bad_json",
        "destructive_shell_refusal",
        "failed_patch_repair",
        "final_response_misuse",
        "hallucinated_file_repair",
        "invalid_arguments",
        "patch_repair",
        "repair_behavior",
        "safe_refusal",
        "schema_repair",
        "unsafe_request_refusal",
        "verification_loop",
        "wrong_tool",
    }
    for category in {
        "failed_patch_repair",
        "schema_repair",
        "unsafe_request_refusal",
        "verification_loop",
    }:
        assert category_metrics[category]["score"] == 1.0


def test_behavioral_synthetic_prompt_sanitizes_repair_bad_outputs() -> None:
    example = next(
        example
        for example in load_dataset_examples(Path("examples/synthetic-data/tool-use.seed.jsonl"))
        if example.metadata.get("category") == "missing_tool_name_search_repair"
    )

    prompt = SyntheticBehaviorEvaluationSuite()._prompt_for_example(example)
    user_payload = json.loads(prompt.split("<|user|>\n", 1)[1].split("\n<|assistant|>", 1)[0])

    assert user_payload["response_contract"]["type"] == "tool_call"
    assert "bad_output" not in user_payload["input"]
    assert user_payload["input"]["previous_invalid_response"]["invalid_response_kind"] == (
        "refusal_text"
    )


def test_trace_behavior_evaluator_scores_response_patch_and_tool_history() -> None:
    examples = load_dataset_examples(Path("examples/trace-data/held-out.trace.jsonl"))
    responses = [
        {
            "patch": example.target.get("patch"),
            "final_response": example.target["final_response"],
            "tool_history": [
                {"tool_name": item["tool_call"]["tool_name"]}
                for item in example.input["tool_history"]
            ],
        }
        for example in examples
    ]
    model = ScriptedModelProvider([_response(response) for response in responses])

    result = asyncio.run(TraceBehaviorEvaluationSuite().evaluate_model(model, examples))

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["example_count"] == 8
    assert result.details["metrics"]["final_response_match_rate"] == 1.0
    assert result.details["metrics"]["patch_match_rate"] == 1.0
    assert result.details["metrics"]["tool_history_match_rate"] == 1.0
    assert "repo.read" in result.details["examples"][0]["tool_profile"]["available_tools"]
    category_metrics = result.details["category_metrics"]
    assert set(category_metrics) >= {
        "trace_failure_response",
        "trace_final_response",
        "trace_patch",
        "trace_patch_repair",
        "trace_search_read_patch",
        "trace_unsafe_path_refusal",
        "trace_unsafe_shell_refusal",
        "trace_verification_loop",
    }
    for category in {
        "trace_unsafe_path_refusal",
        "trace_unsafe_shell_refusal",
        "trace_verification_loop",
    }:
        assert category_metrics[category]["score"] == 1.0
