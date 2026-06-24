"""Tests for staged workspace reasoning evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.infrastructure.dataset_store import load_dataset_examples
from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
from micro_model_agent.infrastructure.workspace_staged_evaluation import (
    WorkspaceStagedEvaluationSuite,
    build_workspace_staged_review_records,
)


def _response(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def _passing_response(example: DatasetExample) -> dict[str, object]:
    stages = example.target["stages"]
    read_search = stages["read_search"]
    diagnosis = stages["diagnosis"]
    patch_proposal = stages["patch_proposal"]
    test_selection = stages["test_selection"]
    final_summary = stages["final_summary"]
    return {
        "read_search": {
            "files": read_search["required_files"],
            "queries": read_search["required_queries"],
        },
        "diagnosis": {
            "root_cause": " ".join(diagnosis["required_terms"]),
            "plan": ["Read first", "diagnose before patching", "keep it in infrastructure"],
        },
        "patch_proposal": {
            "changed_files": patch_proposal["required_changed_files"],
            "patch": " ".join(patch_proposal["patch_contains"]),
        },
        "test_selection": {
            "commands": test_selection["required_commands"],
            "rationale": "focused command only",
        },
        "final_summary": {
            "summary": "dry-run " + " ".join(final_summary["required_summary_terms"]),
            "tests": test_selection["required_commands"],
        },
    }


def _v2_example() -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.EVALUATION,
        source="test_workspace_staged_v2",
        tool_schema_version="v1",
        input={
            "goal": "Recover after a requested docs file is missing.",
            "available_tools": ["repo.search", "repo.read", "repo.write_patch", "test.run"],
            "workspace_files": {
                "docs/usage.md": "CLI usage lives here.",
                "docs/api.md": "stale decoy",
            },
        },
        target={
            "stage_rubric_v2": {
                "read_search": {
                    "required_files": ["docs/usage.md"],
                    "required_queries": ["CLI usage"],
                    "forbidden_files": ["docs/api.md"],
                },
                "diagnosis": {
                    "root_cause_codes": ["missing_requested_file"],
                    "expected_actions": ["recover"],
                    "required_evidence_files": ["docs/usage.md"],
                },
                "patch_proposal": {
                    "expected_actions": ["recover"],
                    "expected_changed_files": [],
                    "forbidden_changed_files": ["docs/api.md"],
                    "dry_run_required": True,
                    "concrete_patch_required": False,
                },
                "test_selection": {
                    "command_policies": ["no_command"],
                    "required_commands": ["No command until the correct docs file is confirmed."],
                },
                "final_summary": {
                    "expected_statuses": ["dry_run_proposal"],
                    "required_files": [],
                    "required_tests": ["No command until the correct docs file is confirmed."],
                    "dry_run_truthful": True,
                },
            }
        },
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
    )


def _passing_v2_response() -> dict[str, object]:
    return {
        "read_search": {
            "files": ["docs/usage.md"],
            "queries": ["CLI usage"],
            "rationale": "Use the nearby docs entrypoint instead of the stale path.",
        },
        "diagnosis": {
            "root_cause_code": "missing_requested_file",
            "expected_action": "recover",
            "evidence_files": ["docs/usage.md"],
            "root_cause": (
                "The requested file is absent, so recover by using the current usage doc."
            ),
        },
        "patch_proposal": {
            "action": "recover",
            "changed_files": [],
            "dry_run": True,
            "dry_run_patch": "No patch until the docs target is confirmed.",
        },
        "test_selection": {
            "command_policy": "no_command",
            "commands": ["No command until the correct docs file is confirmed."],
        },
        "final_summary": {
            "status": "dry_run_proposal",
            "changed_files": [],
            "tests": ["No command until the correct docs file is confirmed."],
            "summary": "dry-run recovery plan only; no files were modified.",
        },
    }


def test_workspace_staged_evaluator_scores_all_five_stages() -> None:
    examples = load_dataset_examples(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl")
    )
    model = ScriptedModelProvider([_response(_passing_response(example)) for example in examples])

    result = asyncio.run(WorkspaceStagedEvaluationSuite().evaluate_model(model, examples))

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["example_count"] == 2
    metrics = result.details["metrics"]
    assert metrics["parse_success_rate"] == 1.0
    assert metrics["read_search_score"] == 1.0
    assert metrics["diagnosis_score"] == 1.0
    assert metrics["patch_proposal_score"] == 1.0
    assert metrics["test_selection_score"] == 1.0
    assert metrics["final_summary_score"] == 1.0
    assert metrics["final_summary_pass_rate"] == 1.0
    assert result.details["examples"][0]["tool_profile"]["available_tools"] == examples[
        0
    ].input["available_tools"]


def test_workspace_staged_evaluator_v2_scores_structured_decisions() -> None:
    example = _v2_example()
    model = ScriptedModelProvider([_response(_passing_v2_response())])

    result = asyncio.run(
        WorkspaceStagedEvaluationSuite(rubric_version="v2").evaluate_model(model, [example])
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["rubric_version"] == "v2"
    assert result.details["metrics"]["read_search_score"] == 1.0


def test_workspace_staged_evaluator_v2_penalizes_decoy_file_selection() -> None:
    example = _v2_example()
    response = _passing_v2_response()
    response["read_search"] = {
        "files": ["docs/api.md"],
        "queries": ["CLI usage"],
        "rationale": "This says missing_requested_file recover docs/usage.md, but picks a decoy.",
    }
    model = ScriptedModelProvider([_response(response)])

    result = asyncio.run(
        WorkspaceStagedEvaluationSuite(pass_threshold=1.0, rubric_version="v2").evaluate_model(
            model,
            [example],
        )
    )

    detail = result.details["examples"][0]
    stage_scores = {stage["name"]: stage for stage in detail["stages"]}
    assert result.passed is False
    assert stage_scores["read_search"]["score"] < 1.0
    assert any("forbidden" in error for error in stage_scores["read_search"]["errors"])


def test_workspace_staged_evaluator_localizes_stage_failures() -> None:
    [example] = load_dataset_examples(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl")
    )[:1]
    response = _passing_response(example)
    response["patch_proposal"] = {
        "changed_files": ["src/micro_model_agent/domain/contracts.py"],
        "patch": "apply patch in domain/contracts.py",
    }
    model = ScriptedModelProvider([_response(response)])

    result = asyncio.run(
        WorkspaceStagedEvaluationSuite(pass_threshold=1.0).evaluate_model(model, [example])
    )

    assert result.passed is False
    detail = result.details["examples"][0]
    stage_scores = {stage["name"]: stage for stage in detail["stages"]}
    assert stage_scores["read_search"]["score"] == 1.0
    assert stage_scores["patch_proposal"]["score"] < 1.0
    assert any(
        "required_changed_files" in error
        for error in stage_scores["patch_proposal"]["errors"]
    )
    assert any("forbidden" in error for error in stage_scores["patch_proposal"]["errors"])


def test_workspace_staged_evaluator_records_parse_errors() -> None:
    examples = load_dataset_examples(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl")
    )
    model = ScriptedModelProvider(["not json"])

    result = asyncio.run(WorkspaceStagedEvaluationSuite().evaluate_model(model, examples[:1]))

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["metrics"]["parse_success_rate"] == 0.0
    assert result.details["examples"][0]["stages"] == []


def test_workspace_staged_prompt_includes_virtual_filesystem() -> None:
    [example] = load_dataset_examples(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl")
    )[:1]

    prompt = WorkspaceStagedEvaluationSuite()._prompt_for_example(example)
    user_payload = json.loads(prompt.split("<|user|>\n", 1)[1].split("\n<|assistant|>", 1)[0])

    assert "workspace_files" in user_payload
    assert "src/micro_model_agent/infrastructure/workspace_staged_evaluation.py" in user_payload[
        "workspace_files"
    ]
    assert "parse_success_rate" in user_payload["workspace_files"][
        "src/micro_model_agent/infrastructure/test_workspace_staged_evaluation.py"
    ]


def test_workspace_staged_review_records_auto_triage_simple_failures() -> None:
    examples = load_dataset_examples(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl")
    )
    report = EvaluationResult(
        passed=False,
        summary="weak",
        score=0.2,
        details={
            "evaluation_metadata": {"run_id": "base", "provider": "scripted"},
            "examples": [
                {
                    "example_id": str(examples[0].id),
                    "score": 0.2,
                    "parse_success": True,
                    "stages": [{"name": "read_search", "score": 0.2, "errors": ["missing"]}],
                    "raw_response": "{}",
                    "parsed_response": {},
                }
            ],
        },
    )

    records = build_workspace_staged_review_records(
        examples=examples[:1],
        reports=[(Path("report.json"), report)],
        simple_failure_threshold=0.4,
    )

    assert records[0]["auto_triage"]["decision"] == "auto_reject_simple_failure"
    assert records[0]["workspace_files"] == examples[0].input["workspace_files"]
    assert records[0]["model_results"][0]["run_id"] == "base"
