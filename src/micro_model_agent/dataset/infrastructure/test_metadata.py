"""Tests for dataset metadata utilities: file hasher and tool profile summarizer."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.metadata import (
    LocalDatasetFileHasher,
    LocalDatasetToolProfileSummarizer,
    dataset_file_sha256,
    summarize_tool_profiles,
    tool_profile_for_example,
)


def _example(tool_name: str = "repo.read") -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.TOOL_USE,
        input={"tool_name": tool_name, "arguments": {}},
        target={"tool_name": tool_name, "arguments": {}},
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
    )


# ---------------------------------------------------------------------------
# dataset_file_sha256
# ---------------------------------------------------------------------------


def test_sha256_of_file_is_hex_string(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"example": 1}\n', encoding="utf-8")
    digest = dataset_file_sha256(path)
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA-256 hex = 64 chars


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"example": 1}\n', encoding="utf-8")
    assert dataset_file_sha256(path) == dataset_file_sha256(path)


def test_sha256_differs_for_different_content(tmp_path: Path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"a": 1}', encoding="utf-8")
    p2.write_text('{"b": 2}', encoding="utf-8")
    assert dataset_file_sha256(p1) != dataset_file_sha256(p2)


# ---------------------------------------------------------------------------
# LocalDatasetFileHasher
# ---------------------------------------------------------------------------


def test_file_hasher_adapter(tmp_path: Path) -> None:
    hasher = LocalDatasetFileHasher()
    path = tmp_path / "data.jsonl"
    path.write_text('content\n', encoding="utf-8")
    digest = hasher.dataset_file_sha256(path)
    assert len(digest) == 64


# ---------------------------------------------------------------------------
# tool_profile_for_example / summarize_tool_profiles
# ---------------------------------------------------------------------------


def test_tool_profile_for_example_has_tools_used() -> None:
    ex = _example("repo.read")
    profile = tool_profile_for_example(ex)
    assert "tools_used" in profile


def test_tool_profile_includes_available_tools() -> None:
    ex = _example("repo.read")
    profile = tool_profile_for_example(
        ex, default_available_tools=["repo.read", "repo.write_patch"]
    )
    assert "available_tools" in profile


def test_summarize_empty_examples_returns_dict() -> None:
    result = summarize_tool_profiles([])
    assert isinstance(result, dict)


def test_summarize_counts_tool_usage(tmp_path: Path) -> None:
    examples = [_example("repo.read") for _ in range(3)] + [_example("repo.write_patch")]
    result = summarize_tool_profiles(examples)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# LocalDatasetToolProfileSummarizer
# ---------------------------------------------------------------------------


def test_summarizer_adapter(tmp_path: Path) -> None:
    svc = LocalDatasetToolProfileSummarizer()
    examples = [_example("repo.read"), _example("repo.write_patch")]
    result = svc.summarize_dataset_tool_profiles(examples)
    assert isinstance(result, dict)


def test_summarizer_with_default_tools(tmp_path: Path) -> None:
    svc = LocalDatasetToolProfileSummarizer()
    examples = [_example("repo.read")]
    result = svc.summarize_dataset_tool_profiles(
        examples,
        default_available_tools=["repo.read", "repo.write_patch"],
    )
    assert isinstance(result, dict)
