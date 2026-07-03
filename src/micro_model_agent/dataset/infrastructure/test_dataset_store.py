"""Tests for training record serialization and dataset store round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.dataset_store import (
    JsonlDatasetExampleStore,
    LocalDatasetExampleReader,
    LocalDatasetExampleWriter,
    dataset_example_from_record,
    dataset_example_to_record,
    load_dataset_examples,
    write_dataset_examples,
)
from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from micro_model_agent.training.infrastructure.training_records import (
    model_artifact_from_record,
    model_artifact_to_record,
    training_run_to_record,
)

# ---------------------------------------------------------------------------
# DatasetExample serialization
# ---------------------------------------------------------------------------


def _example(
    source: str = "trace:abc",
    outcome: OutcomeLabel = OutcomeLabel.ACCEPTED,
    failure_modes: tuple[FailureMode, ...] = (),
) -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "fix"},
        target={"patch": "--- a/f.py\n+++ b/f.py"},
        label=DatasetLabel(
            outcome=outcome,
            quality=QualityLabel.GOOD,
            failure_modes=failure_modes,
            reviewer_notes="ok" if outcome == OutcomeLabel.ACCEPTED else None,
        ),
        source=source,
        metadata={"curated": True},
    )


def test_dataset_example_round_trip() -> None:
    original = _example()
    record = dataset_example_to_record(original)
    restored = dataset_example_from_record(record)

    assert restored.id == original.id
    assert restored.kind == original.kind
    assert restored.source == original.source
    assert restored.label.outcome == original.label.outcome
    assert restored.metadata["curated"] is True


def test_dataset_example_with_failure_modes_round_trips() -> None:
    original = _example(
        outcome=OutcomeLabel.REJECTED,
        failure_modes=(FailureMode.BAD_PATCH, FailureMode.TEST_FAILED),
    )
    record = dataset_example_to_record(original)
    restored = dataset_example_from_record(record)
    assert FailureMode.BAD_PATCH in restored.label.failure_modes
    assert FailureMode.TEST_FAILED in restored.label.failure_modes


def test_load_and_write_examples_round_trip(tmp_path: Path) -> None:
    examples = [_example(f"s{i}") for i in range(3)]
    path = tmp_path / "data.jsonl"
    write_dataset_examples(path, examples)

    loaded = load_dataset_examples(path)
    assert len(loaded) == 3
    assert {e.source for e in loaded} == {"s0", "s1", "s2"}


def test_load_empty_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert load_dataset_examples(path) == []


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_dataset_examples(tmp_path / "nonexistent.jsonl") == []


def test_record_is_json_serializable() -> None:
    record = dataset_example_to_record(_example())
    json.dumps(record)  # should not raise


# ---------------------------------------------------------------------------
# LocalDatasetExampleReader / Writer adapters
# ---------------------------------------------------------------------------


def test_reader_writer_round_trip(tmp_path: Path) -> None:
    import asyncio
    writer = LocalDatasetExampleWriter()
    reader = LocalDatasetExampleReader()
    examples = [_example("x"), _example("y")]
    path = tmp_path / "data.jsonl"

    asyncio.run(writer.save_dataset_examples(path, examples))
    loaded = reader.load_dataset_examples(path)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# JsonlDatasetExampleStore
# ---------------------------------------------------------------------------


def test_jsonl_store_save_and_list(tmp_path: Path) -> None:
    import asyncio
    store = JsonlDatasetExampleStore(tmp_path / "store.jsonl")
    ex1 = _example("s1")
    ex2 = _example("s2")
    asyncio.run(store.save(ex1))
    asyncio.run(store.save(ex2))

    all_examples = asyncio.run(store.list())
    assert len(all_examples) == 2


def test_jsonl_store_list_filtered_by_kind(tmp_path: Path) -> None:
    import asyncio
    store = JsonlDatasetExampleStore(tmp_path / "store.jsonl")
    repair = _example()
    tool_use = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"tool_name": "x"},
        target={"tool_name": "x"},
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
    )
    asyncio.run(store.save(repair))
    asyncio.run(store.save(tool_use))

    repair_list = asyncio.run(store.list(kind="repair"))
    assert len(repair_list) == 1


# ---------------------------------------------------------------------------
# Training record serialization
# ---------------------------------------------------------------------------


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        name="adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="/tmp/adapter",
        base_model="tinyllama",
        metrics={"loss": 0.42},
        metadata={"lora_r": 16},
    )


def test_model_artifact_round_trip() -> None:
    original = _artifact()
    record = model_artifact_to_record(original)
    restored = model_artifact_from_record(record)

    assert restored.id == original.id
    assert restored.name == original.name
    assert restored.kind == original.kind
    assert restored.path == original.path
    assert restored.base_model == original.base_model
    assert restored.metrics["loss"] == pytest.approx(0.42)


def test_model_artifact_record_is_json_serializable() -> None:
    record = model_artifact_to_record(_artifact())
    json.dumps(record)  # should not raise


def test_training_run_to_record_is_json_serializable() -> None:
    config = TrainingConfig(base_model="tinyllama", output_dir="/tmp/run")
    run = TrainingRun(
        kind=TrainingRunKind.SYNTHETIC,
        config=config,
        status=TrainingRunStatus.SUCCEEDED,
        artifacts=(_artifact(),),
        metrics={"loss": 0.42},
    )
    record = training_run_to_record(run)
    json.dumps(record)  # should not raise


def test_training_run_record_has_expected_keys() -> None:
    config = TrainingConfig(base_model="m", output_dir="/tmp")
    run = TrainingRun(kind=TrainingRunKind.SYNTHETIC, config=config)
    record = training_run_to_record(run)
    assert "id" in record
    assert "kind" in record
    assert "config" in record
    assert "status" in record
