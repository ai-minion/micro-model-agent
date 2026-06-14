"""Synthetic dataset generation from committed JSONL templates.

The generator does not invent new content yet. It repeats curated seed records
and gives each output example a fresh ID so downstream code can treat them as
separate examples.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from micro_model_agent.domain.datasets import DatasetExample
from micro_model_agent.infrastructure.dataset_store import load_dataset_examples


class SyntheticTemplateGenerator:
    """Generate synthetic examples by cycling committed template records."""

    def __init__(self, templates_dir: str | Path) -> None:
        self.templates_dir = Path(templates_dir)

    async def generate(self, count: int) -> list[DatasetExample]:
        if count < 1:
            raise ValueError("count must be greater than zero")

        templates = self._load_templates()
        if not templates:
            raise ValueError(f"no synthetic template records found in {self.templates_dir}")

        examples: list[DatasetExample] = []
        for index in range(count):
            # The modulo operator cycles back to the first template when count
            # is larger than the number of seed records.
            template = templates[index % len(templates)]
            examples.append(
                replace(
                    template,
                    id=uuid4(),
                    metadata={**template.metadata, "template_index": index % len(templates)},
                )
            )
        return examples

    def _load_templates(self) -> list[DatasetExample]:
        """Load every committed *.seed.jsonl file from the template directory."""

        examples: list[DatasetExample] = []
        for path in sorted(self.templates_dir.glob("*.seed.jsonl")):
            examples.extend(load_dataset_examples(path))
        return examples
