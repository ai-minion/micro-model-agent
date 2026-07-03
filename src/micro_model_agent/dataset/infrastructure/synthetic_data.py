"""Synthetic dataset generation from committed JSONL templates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from micro_model_agent.dataset.domain.value_objects import DatasetExample
from micro_model_agent.dataset.infrastructure.dataset_store import load_dataset_examples


class SyntheticTemplateGenerator:
    """Generate synthetic examples from curated seed templates."""

    def __init__(self, templates_dir: str | Path) -> None:
        self.templates_dir = Path(templates_dir)

    async def generate(
        self,
        count: int,
        *,
        seed: int | None = None,
        balance_categories: bool = True,
        vary_scenarios: bool = True,
        include_categories: tuple[str, ...] = (),
        exclude_categories: tuple[str, ...] = (),
    ) -> list[DatasetExample]:
        if count < 1:
            raise ValueError("count must be greater than zero")

        templates = self._filter_templates(
            self._load_templates(),
            include_categories=include_categories,
            exclude_categories=exclude_categories,
        )
        if not templates:
            raise ValueError(f"no synthetic template records found in {self.templates_dir}")

        selected_templates = self._select_templates(
            templates,
            count,
            seed=seed,
            balance_categories=balance_categories,
        )
        random = Random(seed)
        examples: list[DatasetExample] = []
        for index, template in enumerate(selected_templates):
            variant_index = index // max(1, len(templates))
            metadata = {
                **template.metadata,
                "template_index": templates.index(template),
                "variant_index": variant_index,
                "variant_strategy": "scenario_text" if vary_scenarios else "template_copy",
            }
            if seed is not None:
                metadata["generation_seed"] = seed
            input_payload = (
                self._variant_input(template, index, random)
                if vary_scenarios
                else dict(template.input)
            )
            examples.append(
                replace(
                    template,
                    id=self._example_id(seed, index),
                    input=input_payload,
                    metadata=metadata,
                )
            )
        return examples

    def _load_templates(self) -> list[DatasetExample]:
        """Load every committed *.seed.jsonl file from the template directory."""

        examples: list[DatasetExample] = []
        for path in sorted(self.templates_dir.glob("*.seed.jsonl")):
            examples.extend(load_dataset_examples(path))
        return examples

    def _filter_templates(
        self,
        templates: list[DatasetExample],
        *,
        include_categories: tuple[str, ...],
        exclude_categories: tuple[str, ...],
    ) -> list[DatasetExample]:
        """Filter templates by metadata.category before selection."""

        includes = set(include_categories)
        excludes = set(exclude_categories)
        if not includes and not excludes:
            return templates

        selected: list[DatasetExample] = []
        for template in templates:
            category = template.metadata.get("category")
            if not isinstance(category, str):
                category = "uncategorized"
            if includes and category not in includes:
                continue
            if category in excludes:
                continue
            selected.append(template)
        return selected

    def _select_templates(
        self,
        templates: list[DatasetExample],
        count: int,
        *,
        seed: int | None,
        balance_categories: bool,
    ) -> list[DatasetExample]:
        if not balance_categories:
            return [templates[index % len(templates)] for index in range(count)]

        grouped = self._templates_by_category(templates)
        categories = sorted(grouped)
        selected: list[DatasetExample] = []
        random = Random(seed)
        category_offsets = {category: 0 for category in categories}

        for index in range(count):
            category = categories[index % len(categories)]
            category_templates = grouped[category]
            offset = category_offsets[category]
            if seed is not None and offset == 0:
                random.shuffle(category_templates)
            selected.append(category_templates[offset % len(category_templates)])
            category_offsets[category] = offset + 1
        return selected

    def _templates_by_category(
        self,
        templates: list[DatasetExample],
    ) -> dict[str, list[DatasetExample]]:
        grouped: dict[str, list[DatasetExample]] = {}
        for template in templates:
            category = template.metadata.get("category")
            key = category if isinstance(category, str) and category else "uncategorized"
            grouped.setdefault(key, []).append(template)
        return grouped

    def _variant_input(
        self,
        template: DatasetExample,
        index: int,
        random: Random,
    ) -> dict[str, Any]:
        input_payload = dict(template.input)
        focus = random.choice(
            [
                "before making changes",
                "while keeping repository safety constraints",
                "using the most direct built-in tool",
                "without guessing missing repository context",
            ]
        )
        if isinstance(input_payload.get("goal"), str):
            input_payload["variant_focus"] = focus
        if isinstance(input_payload.get("context"), str):
            input_payload["variant_focus"] = focus
        return input_payload

    def _example_id(self, seed: int | None, index: int) -> UUID:
        if seed is None:
            return uuid4()
        return uuid5(NAMESPACE_URL, f"{self.templates_dir.resolve()}:{seed}:{index}")
