"""Tests for dataset validation and SFT JSONL export."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.validation import (
    LocalDatasetValidator,
    SftJsonlDatasetExporter,
    export_sft_jsonl,
)


def _run(coro):  # type: ignore[return]
    return asyncio.run(coro)


def _label(
    outcome: OutcomeLabel = OutcomeLabel.ACCEPTED,
    quality: QualityLabel = QualityLabel.GOOD,
) -> DatasetLabel:
    return DatasetLabel(outcome=outcome, quality=quality)


def _tool_use_example(path: str = "main.py") -> DatasetExample:
    # Use a repair example so we don't need to supply pydantic-validated tool args
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "fix the bug", "steps": ["retrieve_context"]},
        target={"patch": f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-bug\n+fix"},
        label=_label(),
    )


def _repair_example() -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "fix the bug", "steps": ["retrieve_context"]},
        target={"patch": "--- a/hello.py\n+++ b/hello.py\n@@ -1 +1 @@\n-bug\n+fix"},
        label=_label(),
    )


# ---------------------------------------------------------------------------
# LocalDatasetValidator
# ---------------------------------------------------------------------------


def test_valid_tool_use_example_passes() -> None:
    validator = LocalDatasetValidator()
    result = _run(validator.validate([_tool_use_example()]))
    assert result.passed is True


def test_valid_repair_example_passes() -> None:
    validator = LocalDatasetValidator()
    result = _run(validator.validate([_repair_example()]))
    assert result.passed is True


def test_empty_dataset_fails() -> None:
    validator = LocalDatasetValidator()
    result = _run(validator.validate([]))
    assert result.passed is False
    assert "no examples" in result.details["errors"][0]


def test_unknown_quality_label_fails() -> None:
    validator = LocalDatasetValidator()
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"tool_name": "x", "arguments": {}},
        target={"tool_name": "x", "arguments": {}},
        label=_label(quality=QualityLabel.UNKNOWN),
    )
    result = _run(validator.validate([example]))
    assert result.passed is False
    assert any("quality" in e for e in result.details["errors"])


def test_empty_input_fails() -> None:
    validator = LocalDatasetValidator()
    example = DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={},
        target={"tool_name": "x", "arguments": {}},
        label=_label(),
    )
    result = _run(validator.validate([example]))
    assert result.passed is False
    assert any("input" in e for e in result.details["errors"])


def test_repair_without_patch_or_response_fails() -> None:
    validator = LocalDatasetValidator()
    example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "fix"},
        target={},  # no patch or final_response
        label=_label(),
    )
    result = _run(validator.validate([example]))
    assert result.passed is False


def test_multiple_valid_examples_pass() -> None:
    validator = LocalDatasetValidator()
    examples = [_tool_use_example(f"file_{i}.py") for i in range(5)]
    result = _run(validator.validate(examples))
    assert result.passed is True
    assert result.details["example_count"] == 5


def test_validation_result_has_summary() -> None:
    validator = LocalDatasetValidator()
    result = _run(validator.validate([_tool_use_example()]))
    assert "validated" in result.summary


# ---------------------------------------------------------------------------
# SftJsonlDatasetExporter / export_sft_jsonl
# ---------------------------------------------------------------------------


def test_export_writes_jsonl_file(tmp_path: Path) -> None:
    exporter = SftJsonlDatasetExporter()
    examples = [_tool_use_example(), _repair_example()]
    out = tmp_path / "train.sft.jsonl"
    exporter.export_dataset_examples(out, examples)
    assert out.exists()


def test_export_writes_one_line_per_example(tmp_path: Path) -> None:
    examples = [_tool_use_example(f"f{i}.py") for i in range(3)]
    out = tmp_path / "train.sft.jsonl"
    export_sft_jsonl(out, examples)

    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


def test_export_each_line_is_valid_json(tmp_path: Path) -> None:
    examples = [_tool_use_example()]
    out = tmp_path / "train.sft.jsonl"
    export_sft_jsonl(out, examples)

    for line in out.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            assert "messages" in record  # SFT format has messages key


def test_export_repair_example_produces_messages(tmp_path: Path) -> None:
    out = tmp_path / "repair.jsonl"
    export_sft_jsonl(out, [_repair_example()])

    record = json.loads(out.read_text().strip())
    messages = record["messages"]
    assert any(m["role"] == "system" for m in messages)
    assert any(m["role"] == "user" for m in messages)
    assert any(m["role"] == "assistant" for m in messages)
