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

    assert [example.target["tool_name"] for example in loaded] == [
        example.target["tool_name"] for example in examples
    ]


def test_dataset_validator_accepts_seed_examples() -> None:
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(6))

    result = asyncio.run(LocalDatasetValidator().validate(examples))

    assert result.passed is True
    assert result.details["example_count"] == 6


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


def test_export_sft_jsonl_writes_chat_records(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.sft.jsonl"
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(2))

    export_sft_jsonl(output, examples)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 2
    assert records[0]["messages"][0]["role"] == "system"
    assert records[0]["messages"][2]["role"] == "assistant"


def test_fake_training_runner_writes_artifact_and_evaluates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic.jsonl"
    run_dir = tmp_path / "training" / "runs" / "latest"
    examples = asyncio.run(SyntheticTemplateGenerator("examples/synthetic-data").generate(4))
    asyncio.run(JsonlDatasetExampleStore(dataset_path).save_many(examples))

    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        output_dir=str(run_dir),
        parameters={"dataset_path": str(dataset_path), "example_count": len(examples)},
    )
    run = asyncio.run(FakeTrainingRunner().run(config))
    artifact = load_artifact_from_training_run(run_dir)
    evaluation = asyncio.run(SyntheticEvaluationSuite().evaluate_artifact(artifact))

    assert run.status.value == "succeeded"
    assert (run_dir / "run.json").exists()
    assert artifact.metrics["synthetic_example_count"] == 4.0
    assert evaluation.passed is True
    assert load_dataset_examples(dataset_path)
