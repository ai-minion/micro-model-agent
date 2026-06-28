"""Tests for the local synthetic dataset and fake training pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.domain.training import TrainingConfig
from micro_model_agent.infrastructure.dataset_metadata import dataset_file_sha256
from micro_model_agent.infrastructure.dataset_store import (
    JsonlDatasetExampleStore,
    load_dataset_examples,
)
from micro_model_agent.infrastructure.dataset_validation import (
    LocalDatasetValidator,
    export_sft_jsonl,
)
from micro_model_agent.infrastructure.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.training_artifacts import (
    FakeTrainingRunner,
    SyntheticEvaluationSuite,
    load_artifact_from_training_run,
)


def test_synthetic_generator_reads_seed_templates() -> None:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(8))

    assert len(examples) == 8
    assert {example.kind for example in examples} >= {
        DatasetExampleKind.TOOL_USE,
        DatasetExampleKind.REPAIR,
    }
    assert len({example.id for example in examples}) == 8


def test_jsonl_dataset_store_round_trips_examples(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.jsonl"
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(3))
    store = JsonlDatasetExampleStore(output)

    asyncio.run(store.save_many(examples))
    loaded = asyncio.run(store.list())

    assert [example.target for example in loaded] == [example.target for example in examples]
    assert loaded[0].metadata["tool_profile"]["available_tools"] == examples[0].input.get(
        "available_tools",
        [],
    )


def test_dataset_validator_accepts_seed_examples() -> None:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(6))

    result = asyncio.run(LocalDatasetValidator().validate(examples))

    assert result.passed is True
    assert result.details["example_count"] == 6
    assert sum(result.details["category_counts"].values()) == 6
    assert sum(result.details["kind_counts"].values()) == 6
    assert sum(result.details["outcome_counts"].values()) == 6
    assert "repo.search" in result.details["tool_profile"]["available_tools"]


def test_dataset_validator_accepts_all_committed_seed_templates() -> None:
    examples: list[DatasetExample] = []
    for path in sorted(Path("examples/synthetic-data").glob("*.seed.jsonl")):
        examples.extend(load_dataset_examples(path))

    result = asyncio.run(LocalDatasetValidator().validate(examples))

    assert result.passed is True
    assert result.details["example_count"] == 40
    assert result.details["category_counts"]["diff_inspection"] == 1
    assert result.details["category_counts"]["patch_preview"] == 1
    assert result.details["category_counts"]["prose_patch_repair"] == 1
    assert result.details["category_counts"]["missing_tool_name_patch_repair"] == 1
    assert result.details["category_counts"]["missing_tool_name_search_repair"] == 1
    assert result.details["category_counts"]["search_limit_repair"] == 1
    assert result.details["category_counts"]["search_limit_250_repair"] == 1
    assert result.details["category_counts"]["git_diff_command_alias_repair"] == 1
    assert result.details["category_counts"]["invented_patch_tool_repair"] == 1
    assert result.details["category_counts"]["repo_read_directory_alias_repair"] == 1
    assert result.details["category_counts"]["search_sort_alias_repair"] == 1
    assert result.details["category_counts"]["test_command_alias_repair"] == 1
    assert result.details["category_counts"]["test_command_field_repair"] == 1
    assert result.details["category_counts"]["test_command_key_alias_repair"] == 1
    assert result.details["category_counts"]["test_command_key_repair"] == 1
    assert result.details["category_counts"]["test_argument_repair"] == 1
    assert result.details["category_counts"]["unsafe_final_response_repair"] == 1
    assert result.details["category_counts"]["variant_focus_argument_repair"] == 1
    assert result.details["category_counts"]["write_patch_max_bytes_repair"] == 1
    assert result.details["category_counts"]["repo_search_request_schema_contrast"] == 1
    assert result.details["category_counts"]["test_run_request_schema_contrast"] == 1
    assert result.details["category_counts"]["variant_focus_request_schema_contrast"] == 1
    assert result.details["category_counts"]["write_patch_request_schema_contrast"] == 1
    assert result.details["category_counts"]["trace_patch_training"] == 1
    assert result.details["category_counts"]["trace_verification_loop_training"] == 1
    assert result.details["kind_counts"]["repair"] == 21
    assert result.details["kind_counts"]["tool_use"] == 12
    assert result.details["kind_counts"]["evaluation"] == 7
    assert "git.diff" in result.details["tool_profile"]["tools_used"]
    assert "test.run" in result.details["tool_profile"]["tools_used"]


def test_synthetic_generator_balances_categories_and_uses_seeded_variants() -> None:
    first = asyncio.run(
        SyntheticTemplateGenerator("examples/synthetic-data").generate(11, seed=17)
    )
    second = asyncio.run(
        SyntheticTemplateGenerator("examples/synthetic-data").generate(11, seed=17)
    )

    assert [example.id for example in first] == [example.id for example in second]
    assert [example.input["goal"] for example in first] == [
        example.input["goal"] for example in second
    ]
    counts: dict[str, int] = {}
    for example in first:
        category = str(example.metadata["category"])
        counts[category] = counts.get(category, 0) + 1

    assert max(counts.values()) - min(counts.values()) <= 1
    assert all("variant_strategy" in example.metadata for example in first)
    assert any("variant_focus" in example.input for example in first)
    assert all("Scenario" not in str(example.input["goal"]) for example in first)


def test_synthetic_generator_filters_categories() -> None:
    included = asyncio.run(
        SyntheticTemplateGenerator("examples/synthetic-data").generate(
            4,
            include_categories=("trace_patch_training", "trace_final_response_training"),
            seed=31,
        )
    )
    excluded = asyncio.run(
        SyntheticTemplateGenerator("examples/synthetic-data").generate(
            6,
            exclude_categories=("trace_patch_training",),
            seed=31,
        )
    )

    assert {
        example.metadata["category"] for example in included
    } == {"trace_patch_training", "trace_final_response_training"}
    assert all(example.metadata["category"] != "trace_patch_training" for example in excluded)


def test_dataset_validator_accepts_held_out_behavior_examples() -> None:
    examples = load_dataset_examples(Path("examples/synthetic-data/held-out.behavior.jsonl"))

    result = asyncio.run(LocalDatasetValidator().validate(examples))

    assert result.passed is True
    assert result.details["example_count"] == 15
    assert result.details["category_counts"]["documentation_grounded_retrieval"] == 1
    assert result.details["category_counts"]["failed_patch_repair"] == 1
    assert result.details["category_counts"]["patch_repair"] == 1
    assert result.details["category_counts"]["schema_repair"] == 1
    assert result.details["category_counts"]["unsafe_request_refusal"] == 1
    assert result.details["category_counts"]["verification_loop"] == 1
    assert result.details["kind_counts"]["repair"] == 5
    assert result.details["outcome_counts"]["rejected"] == 3


def test_dataset_validator_accepts_held_out_trace_examples() -> None:
    examples = load_dataset_examples(Path("examples/trace-data/held-out.trace.jsonl"))

    result = asyncio.run(LocalDatasetValidator().validate(examples))

    assert result.passed is True
    assert result.details["example_count"] == 8
    assert result.details["category_counts"]["trace_unsafe_path_refusal"] == 1
    assert result.details["category_counts"]["trace_unsafe_shell_refusal"] == 1
    assert result.details["category_counts"]["trace_verification_loop"] == 1
    assert result.details["kind_counts"]["evaluation"] == 8
    assert result.details["outcome_counts"]["accepted"] == 5
    assert result.details["outcome_counts"]["rejected"] == 2


def test_dataset_validator_rejects_unknown_quality_label() -> None:
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "Find files"},
        target={"tool_name": "repo.search", "arguments": {"query": "files"}},
        label=DatasetLabel(outcome=OutcomeLabel.NEEDS_REVIEW, quality=QualityLabel.UNKNOWN),
    )

    result = asyncio.run(LocalDatasetValidator().validate([example]))

    assert result.passed is False
    assert "quality label must be known" in result.details["errors"][0]


def test_dataset_validator_accepts_repair_example_with_final_response() -> None:
    example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "Summarize the change"},
        target={"final_response": "Updated app.py."},
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
    )

    result = asyncio.run(LocalDatasetValidator().validate([example]))

    assert result.passed is True


def test_dataset_validator_rejects_refusal_and_tool_inconsistency() -> None:
    examples = [
        DatasetExample(
            kind=DatasetExampleKind.TOOL_USE,
            input={
                "goal": "Read a file",
                "available_tools": ["repo.search"],
            },
            target={
                "tool_name": "repo.read",
                "arguments": {"files": [{"path": "README.md"}]},
            },
            label=DatasetLabel(outcome=OutcomeLabel.REJECTED, quality=QualityLabel.GOOD),
        ),
        DatasetExample(
            kind=DatasetExampleKind.TOOL_USE,
            input={
                "goal": "Search files",
                "available_tools": ["repo.search"],
            },
            target={
                "tool_name": "repo.search",
                "arguments": {"kind": "text", "limit": 25},
                "refusal": "No.",
            },
            label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        ),
        DatasetExample(
            kind=DatasetExampleKind.TOOL_USE,
            input={
                "goal": "Read a private file",
                "available_tools": ["repo.read"],
            },
            target={
                "tool_name": "repo.read",
                "arguments": {"files": [{"path": "README.md"}]},
                "refusal": "No.",
            },
            label=DatasetLabel(outcome=OutcomeLabel.REJECTED, quality=QualityLabel.GOOD),
        ),
    ]

    result = asyncio.run(LocalDatasetValidator().validate(examples))

    assert result.passed is False
    errors = result.details["errors"]
    assert any("rejected examples must include target.refusal" in error for error in errors)
    assert any("is not in input.available_tools" in error for error in errors)
    assert any("only rejected examples may include target.refusal" in error for error in errors)
    assert any("refusal targets must not include tool arguments" in error for error in errors)
    assert any("invalid repo.search arguments" in error for error in errors)


def test_dataset_validator_accepts_refusal_only_tool_use_example() -> None:
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"goal": "Read ../secret.txt", "available_tools": ["repo.read"]},
        target={"refusal": "Reject outside-repository paths."},
        label=DatasetLabel(
            outcome=OutcomeLabel.REJECTED,
            quality=QualityLabel.GOOD,
        ),
    )

    result = asyncio.run(LocalDatasetValidator().validate([example]))

    assert result.passed is True


def test_export_sft_jsonl_writes_chat_records(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.sft.jsonl"
    examples = [
        DatasetExample(
            kind=DatasetExampleKind.TOOL_USE,
            input={
                "goal": "Find DatasetExample",
                "available_tools": ["repo.search", "repo.read"],
                "context": "Search before reading when the path is unknown.",
            },
            target={
                "tool_name": "repo.search",
                "arguments": {"query": "DatasetExample", "kind": "text", "limit": 25},
                "reason": "Search for the symbol before reading.",
            },
            label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
            tool_schema_version="v1",
        ),
        DatasetExample(
            kind=DatasetExampleKind.TOOL_USE,
            input={"goal": "Read ../secret.txt", "available_tools": ["repo.read"]},
            target={"refusal": "Reject outside-repository paths."},
            label=DatasetLabel(outcome=OutcomeLabel.REJECTED, quality=QualityLabel.GOOD),
            tool_schema_version="v1",
        ),
        DatasetExample(
            kind=DatasetExampleKind.REPAIR,
            input={"goal": "Repair an unsafe fake tool response"},
            target={"final_response": "Reject path traversal.", "ok": False},
            label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
            tool_schema_version="v1",
        ),
        DatasetExample(
            kind=DatasetExampleKind.EVALUATION,
            input={
                "goal": "Change value() in app.py to return 2.",
                "tool_history": [
                    {
                        "tool_call": {
                            "tool_name": "repo.read",
                            "arguments": {"files": [{"path": "app.py"}]},
                        }
                    }
                ],
            },
            target={
                "patch": (
                    "diff --git a/app.py b/app.py\n"
                    "--- a/app.py\n"
                    "+++ b/app.py\n"
                    "@@\n"
                    " def value():\n"
                    "-    return 1\n"
                    "+    return 2\n"
                ),
                "final_response": "Updated app.py so value() returns 2.",
                "changed_files": ["app.py"],
            },
            label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
            tool_schema_version="v1",
        ),
        DatasetExample(
            kind=DatasetExampleKind.EVALUATION,
            input={
                "goal": "Plan a dry-run docs update.",
                "available_tools": ["repo.search", "repo.read", "repo.write_patch", "test.run"],
                "workspace_files": {"docs/usage.md": "# Usage\n"},
                "candidate_files": ["docs/usage.md"],
            },
            target={
                "gold_response": {
                    "read_search": {"files": ["docs/usage.md"], "queries": ["Usage"]},
                    "diagnosis": {"root_cause": "Docs update only.", "plan": ["Read docs."]},
                    "patch_proposal": {
                        "changed_files": ["docs/usage.md"],
                        "dry_run_patch": "Add usage note.",
                    },
                    "test_selection": {
                        "commands": ["uv run pytest src/micro_model_agent/interfaces/test_cli.py"]
                    },
                    "final_summary": {"summary": "dry-run docs/usage.md update."},
                },
                "stages": {
                    "read_search": {
                        "required_files": ["docs/usage.md"],
                        "required_queries": ["Usage"],
                    }
                },
            },
            label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
            tool_schema_version="v1",
        ),
    ]

    export_sft_jsonl(output, examples)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 5
    assert records[0]["messages"][0]["role"] == "system"
    assert records[0]["messages"][2]["role"] == "assistant"
    assert json.loads(records[0]["messages"][1]["content"])["tool_schemas"]["repo.search"]
    assert json.loads(records[0]["messages"][2]["content"]) == {
        "arguments": {"kind": "text", "limit": 25, "query": "DatasetExample"},
        "reason": "Search for the symbol before reading.",
        "tool_name": "repo.search",
    }
    assert json.loads(records[1]["messages"][2]["content"]) == {
        "refusal": "Reject outside-repository paths.",
    }
    assert json.loads(records[2]["messages"][2]["content"]) == {
        "final_response": "Reject path traversal.",
        "ok": False,
    }
    assert json.loads(records[3]["messages"][2]["content"]) == {
        "changed_files": ["app.py"],
        "final_response": "Updated app.py so value() returns 2.",
        "patch": (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@\n"
            " def value():\n"
            "-    return 1\n"
            "+    return 2\n"
        ),
        "tool_history": [
            {
                "arguments": {"files": [{"path": "app.py"}]},
                "tool_name": "repo.read",
            }
        ],
    }
    trace_user_payload = json.loads(records[3]["messages"][1]["content"])
    assert "tool_schemas" not in trace_user_payload
    assert trace_user_payload["available_tools"] == [
        "repo.search",
        "repo.read",
        "repo.semantic_search",
        "repo.write_patch",
        "repo.write_files",
        "test.run",
        "git.diff",
    ]
    assert records[3]["messages"][0]["content"].startswith(
        "You are MicroModelAgent replaying a held-out workflow trace."
    )
    staged_user_payload = json.loads(records[4]["messages"][1]["content"])
    assert staged_user_payload["workspace_files"] == {"docs/usage.md": "# Usage\n"}
    assert records[4]["messages"][0]["content"].startswith(
        "You are MicroModelAgent evaluating a dry-run coding task."
    )
    assert json.loads(records[4]["messages"][2]["content"]) == examples[4].target[
        "gold_response"
    ]
    assert records[0]["metadata"]["tool_profile"]["tool_schema_version"] == "v1"
    assert records[0]["metadata"]["tool_profile"]["available_tools"] == examples[0].input.get(
        "available_tools",
        [],
    )


def test_export_sft_jsonl_sanitizes_repair_bad_outputs(tmp_path: Path) -> None:
    output = tmp_path / "repair.sft.jsonl"
    example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={
            "goal": "Repair a safe search request that returned a refusal.",
            "available_tools": ["repo.search", "repo.read"],
            "bad_output": {
                "refusal": "Use repo.search first, then repo.read.",
                "reason": "Previous response refused a safe lookup.",
            },
            "validation_error": "tool_name must be a string",
        },
        target={
            "tool_name": "repo.search",
            "arguments": {"query": "DatasetExample", "kind": "text", "limit": 25},
            "reason": "Safe lookup requests should use the available search tool.",
        },
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        tool_schema_version="v1",
    )

    export_sft_jsonl(output, [example])
    record = json.loads(output.read_text(encoding="utf-8"))
    user_payload = json.loads(record["messages"][1]["content"])
    assistant_payload = json.loads(record["messages"][2]["content"])

    assert user_payload["response_contract"] == {
        "type": "tool_call",
        "required_keys": ["tool_name", "arguments"],
        "forbidden_keys": ["refusal", "final_response", "ok"],
        "allowed_top_level_keys": ["tool_name", "arguments", "reason"],
        "forbidden_top_level_keys": [
            "argument_keys",
            "argument_values",
            "argument_changes",
            "argument_reconciliation",
            "selected_tool",
            "changed_fields",
        ],
    }
    assert "bad_output" not in user_payload["input"]
    assert user_payload["input"]["previous_invalid_response"] == {
        "invalid_response_kind": "refusal_text",
        "previous_reason": "Previous response refused a safe lookup.",
    }
    assert assistant_payload == {
        "arguments": {"kind": "text", "limit": 25, "query": "DatasetExample"},
        "reason": "Safe lookup requests should use the available search tool.",
        "tool_name": "repo.search",
    }


def test_export_sft_jsonl_sanitizes_invalid_argument_helpers(tmp_path: Path) -> None:
    output = tmp_path / "command-repair.sft.jsonl"
    example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={
            "goal": "Repair a test command alias.",
            "available_tools": ["test.run"],
            "bad_output": {
                "tool_name": "test.run",
                "arguments": {
                    "command": "pytest",
                    "extra_args": ["src/micro_model_agent/infrastructure"],
                },
            },
            "validation_error": "test.run requires command_name and forbids command",
        },
        target={
            "tool_name": "test.run",
            "arguments": {
                "command_name": "pytest",
                "extra_args": ["src/micro_model_agent/infrastructure"],
                "timeout_seconds": 120,
            },
            "reason": "Use command_name, not command.",
        },
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        tool_schema_version="v1",
    )

    export_sft_jsonl(output, [example])
    record = json.loads(output.read_text(encoding="utf-8"))
    user_payload = json.loads(record["messages"][1]["content"])
    previous = user_payload["input"]["previous_invalid_response"]

    assert previous["invalid_argument_field_names"] == ["command", "extra_args"]
    assert previous["invalid_argument_value_notes"]["invalid_shell_command"] == "pytest"
    assert previous["invalid_selected_tool"] == "test.run"
    assert "argument_keys" not in previous
    assert "argument_values" not in previous
    assert "selected_tool" not in previous


def test_export_sft_jsonl_omits_generation_variant_focus(tmp_path: Path) -> None:
    output = tmp_path / "variant-focus.sft.jsonl"
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={
            "goal": "Search for the tool catalog.",
            "available_tools": ["repo.search"],
            "context": "The path is not known.",
            "variant_focus": "current workspace",
        },
        target={
            "tool_name": "repo.search",
            "arguments": {"query": "BuiltinToolSpec", "kind": "text", "limit": 25},
            "reason": "Search first because the file path is unknown.",
        },
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        tool_schema_version="v1",
    )

    export_sft_jsonl(output, [example])
    record = json.loads(output.read_text(encoding="utf-8"))
    user_payload = json.loads(record["messages"][1]["content"])

    assert "variant_focus" not in user_payload["input"]


def test_fake_training_runner_writes_artifact_and_evaluates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic.jsonl"
    run_dir = tmp_path / "training" / "runs" / "latest"
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(4))
    asyncio.run(JsonlDatasetExampleStore(dataset_path).save_many(examples))
    dataset_sha256 = dataset_file_sha256(dataset_path)

    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        output_dir=str(run_dir),
        parameters={
            "dataset_path": str(dataset_path),
            "source_dataset_path": str(dataset_path),
            "source_dataset_sha256": dataset_sha256,
            "training_dataset_sha256": dataset_sha256,
            "example_count": len(examples),
            "dataset_tool_profile": {
                "available_tools": ["repo.read", "repo.search"],
                "tool_schema_versions": ["v1"],
            },
        },
    )
    run = asyncio.run(FakeTrainingRunner().run(config))
    artifact = load_artifact_from_training_run(run_dir)
    evaluation = asyncio.run(SyntheticEvaluationSuite().evaluate_artifact(artifact))

    assert run.status.value == "succeeded"
    assert run.dataset_version == dataset_sha256
    assert (run_dir / "run.json").exists()
    assert artifact.metrics["synthetic_example_count"] == 4.0
    assert artifact.metadata["source_dataset_path"] == str(dataset_path)
    assert artifact.metadata["source_dataset_sha256"] == dataset_sha256
    assert artifact.metadata["training_dataset_sha256"] == dataset_sha256
    assert artifact.metadata["dataset_tool_profile"]["tool_schema_versions"] == ["v1"]
    assert evaluation.passed is True
    assert load_dataset_examples(dataset_path)
