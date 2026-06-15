"""Dataset validation and export helpers.

Validation catches bad or incomplete examples before they are handed to a
training backend, where mistakes are usually slower and harder to diagnose.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS


class LocalDatasetValidator:
    """Validate dataset examples against labels and tool contracts."""

    async def validate(self, examples: list[DatasetExample]) -> EvaluationResult:
        errors: list[str] = []

        if not examples:
            errors.append("dataset contains no examples")

        # Accumulate every error instead of failing fast so users can fix a batch
        # of dataset issues in one pass.
        for index, example in enumerate(examples):
            errors.extend(self._validate_example(example, index))

        passed = not errors
        return EvaluationResult(
            passed=passed,
            summary=f"validated {len(examples)} examples with {len(errors)} error(s)",
            score=1.0 if passed else 0.0,
            details={
                "errors": errors,
                "example_count": len(examples),
                "category_counts": self._category_counts(examples),
                "kind_counts": self._kind_counts(examples),
                "outcome_counts": self._outcome_counts(examples),
            },
        )

    def _validate_example(self, example: DatasetExample, index: int) -> list[str]:
        errors: list[str] = []
        prefix = f"example[{index}]"

        # All example kinds need input, target, and a known quality label.
        if not example.input:
            errors.append(f"{prefix}: input is required")
        if not example.target:
            errors.append(f"{prefix}: target is required")
        if example.label.quality is QualityLabel.UNKNOWN:
            errors.append(f"{prefix}: quality label must be known before training")
        errors.extend(self._validate_refusal_consistency(example, prefix))

        if example.kind in {DatasetExampleKind.TOOL_USE, DatasetExampleKind.REPAIR}:
            # Tool-use examples must also match the tool argument schemas.
            errors.extend(self._validate_tool_target(example, prefix))

        return errors

    def _validate_tool_target(self, example: DatasetExample, prefix: str) -> list[str]:
        errors: list[str] = []
        target = example.target
        tool_name = target.get("tool_name")
        if not isinstance(tool_name, str):
            return [f"{prefix}: target.tool_name is required"]

        available_tools = example.input.get("available_tools")
        if isinstance(available_tools, list) and all(
            isinstance(tool, str) for tool in available_tools
        ) and tool_name not in available_tools:
            errors.append(
                f"{prefix}: target.tool_name {tool_name!r} is not in input.available_tools"
            )

        contract = TOOL_ARGUMENT_CONTRACTS.get(tool_name)
        if contract is None:
            errors.append(f"{prefix}: unknown tool_name {tool_name!r}")
            return errors

        arguments = target.get("arguments")
        if arguments is None:
            if "refusal" in target:
                return errors
            errors.append(f"{prefix}: target.arguments is required when no refusal is present")
            return errors
        if not isinstance(arguments, dict):
            errors.append(f"{prefix}: target.arguments must be an object")
            return errors

        try:
            # Pydantic performs detailed type and range validation from the
            # contract model for the selected tool.
            contract.model_validate(arguments)
        except ValidationError as exc:
            errors.append(f"{prefix}: invalid {tool_name} arguments: {exc.errors()}")
        return errors

    def _validate_refusal_consistency(
        self,
        example: DatasetExample,
        prefix: str,
    ) -> list[str]:
        errors: list[str] = []
        refusal = example.target.get("refusal")
        has_refusal = isinstance(refusal, str) and bool(refusal.strip())

        if example.label.outcome is OutcomeLabel.REJECTED and not has_refusal:
            errors.append(f"{prefix}: rejected examples must include target.refusal")
        if example.label.outcome is not OutcomeLabel.REJECTED and "refusal" in example.target:
            errors.append(f"{prefix}: only rejected examples may include target.refusal")
        return errors

    def _category_counts(self, examples: list[DatasetExample]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in examples:
            category = example.metadata.get("category")
            key = category if isinstance(category, str) and category else "uncategorized"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _kind_counts(self, examples: list[DatasetExample]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in examples:
            key = example.kind.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _outcome_counts(self, examples: list[DatasetExample]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in examples:
            key = example.label.outcome.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


def export_sft_jsonl(path: Path, examples: list[DatasetExample]) -> None:
    """Export examples in a simple supervised fine-tuning chat JSONL shape."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            # The training backend expects chat-style records: system, user,
            # assistant. We serialize the flexible input/target dictionaries as
            # JSON strings inside those messages.
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are MicroModelAgent's workflow executor. "
                            "Choose safe typed tool calls and follow retrieved context."
                        ),
                    },
                    {"role": "user", "content": json.dumps(example.input, sort_keys=True)},
                    {"role": "assistant", "content": json.dumps(example.target, sort_keys=True)},
                ],
                "metadata": {
                    "id": str(example.id),
                    "kind": example.kind.value,
                    "source": example.source,
                    "tool_schema_version": example.tool_schema_version,
                    "label": {
                        "outcome": example.label.outcome.value,
                        "quality": example.label.quality.value,
                        "failure_modes": [
                            failure_mode.value for failure_mode in example.label.failure_modes
                        ],
                    },
                    **example.metadata,
                },
            }
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")
