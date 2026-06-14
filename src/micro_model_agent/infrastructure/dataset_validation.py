"""Dataset validation and export helpers.

Validation catches bad or incomplete examples before they are handed to a
training backend, where mistakes are usually slower and harder to diagnose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import DatasetExample, DatasetExampleKind, QualityLabel
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
            details={"errors": errors, "example_count": len(examples)},
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

        if example.kind in {DatasetExampleKind.TOOL_USE, DatasetExampleKind.REPAIR}:
            # Tool-use examples must also match the tool argument schemas.
            errors.extend(self._validate_tool_target(example.target, prefix))

        return errors

    def _validate_tool_target(self, target: dict[str, Any], prefix: str) -> list[str]:
        errors: list[str] = []
        tool_name = target.get("tool_name")
        if not isinstance(tool_name, str):
            return [f"{prefix}: target.tool_name is required"]

        contract = TOOL_ARGUMENT_CONTRACTS.get(tool_name)
        if contract is None:
            return [f"{prefix}: unknown tool_name {tool_name!r}"]

        arguments = target.get("arguments")
        if arguments is None:
            if "refusal" in target:
                return []
            return [f"{prefix}: target.arguments is required when no refusal is present"]
        if not isinstance(arguments, dict):
            return [f"{prefix}: target.arguments must be an object"]

        try:
            # Pydantic performs detailed type and range validation from the
            # contract model for the selected tool.
            contract.model_validate(arguments)
        except ValidationError as exc:
            errors.append(f"{prefix}: invalid {tool_name} arguments: {exc.errors()}")
        return errors


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
