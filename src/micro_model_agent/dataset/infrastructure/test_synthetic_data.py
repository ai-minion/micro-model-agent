"""Tests for SyntheticTemplateGenerator."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.dataset_store import (
    write_dataset_examples,
)
from micro_model_agent.dataset.infrastructure.synthetic_data import (
    SyntheticTemplateGenerator,
)


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _write_seed(path: Path, examples: list[DatasetExample]) -> None:
    """Write a *.seed.jsonl template file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_examples(path, examples)


def _seed_example(
    source: str = "template:repair",
    category: str = "tool_use_basic",
) -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "fix the null pointer", "steps": ["retrieve_context"]},
        target={"patch": "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-bug\n+fix"},
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        source=source,
        metadata={"category": category},
    )


def _setup_templates(tmp_path: Path, count: int = 5) -> Path:
    """Create a templates directory with one seed file."""
    templates_dir = tmp_path / "templates"
    examples = [_seed_example(f"template:{i}", f"cat_{i % 2}") for i in range(count)]
    _write_seed(templates_dir / "test.seed.jsonl", examples)
    return templates_dir


# ---------------------------------------------------------------------------
# SyntheticTemplateGenerator
# ---------------------------------------------------------------------------


def test_generate_returns_requested_count(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    examples = _run(gen.generate(3))
    assert len(examples) == 3


def test_generate_returns_dataset_examples(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    examples = _run(gen.generate(2))
    assert all(isinstance(e, DatasetExample) for e in examples)


def test_generate_with_seed_is_deterministic(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    run1 = _run(gen.generate(3, seed=42))
    run2 = _run(gen.generate(3, seed=42))
    assert [e.id for e in run1] == [e.id for e in run2]


def test_generate_different_seeds_differ(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    run1 = _run(gen.generate(5, seed=1))
    run2 = _run(gen.generate(5, seed=2))
    # IDs should differ for different seeds
    assert [e.id for e in run1] != [e.id for e in run2]


def test_generate_sets_variant_metadata(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    examples = _run(gen.generate(2, seed=99))
    for e in examples:
        assert "variant_index" in e.metadata
        assert "template_index" in e.metadata
        assert "generation_seed" in e.metadata
        assert e.metadata["generation_seed"] == 99


def test_generate_zero_count_raises(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    with pytest.raises(ValueError, match="greater than zero"):
        _run(gen.generate(0))


def test_generate_no_templates_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    gen = SyntheticTemplateGenerator(empty_dir)
    with pytest.raises(ValueError, match="no synthetic template"):
        _run(gen.generate(1))


def test_generate_include_categories_filters(tmp_path: Path) -> None:
    templates_dir = _setup_templates(tmp_path, count=6)
    gen = SyntheticTemplateGenerator(templates_dir)
    # Categories are cat_0 and cat_1 alternating
    examples = _run(gen.generate(3, include_categories=("cat_0",)))
    assert all(e.metadata.get("category") == "cat_0" for e in examples)


def test_generate_exclude_categories_filters(tmp_path: Path) -> None:
    templates_dir = _setup_templates(tmp_path, count=6)
    gen = SyntheticTemplateGenerator(templates_dir)
    examples = _run(gen.generate(3, exclude_categories=("cat_1",)))
    assert all(e.metadata.get("category") != "cat_1" for e in examples)


def test_generate_vary_scenarios_adds_metadata(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    examples = _run(gen.generate(2, vary_scenarios=True))
    assert all(e.metadata.get("variant_strategy") == "scenario_text" for e in examples)


def test_generate_no_vary_uses_template_copy_strategy(tmp_path: Path) -> None:
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path))
    examples = _run(gen.generate(2, vary_scenarios=False))
    assert all(e.metadata.get("variant_strategy") == "template_copy" for e in examples)


def test_generate_more_than_templates_repeats(tmp_path: Path) -> None:
    # 5 templates, requesting 10 examples — should repeat
    gen = SyntheticTemplateGenerator(_setup_templates(tmp_path, count=5))
    examples = _run(gen.generate(10))
    assert len(examples) == 10
