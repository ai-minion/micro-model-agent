"""Dataset validation and export helpers.

Validation catches bad or incomplete examples before they are handed to a
training backend, where mistakes are usually slower and harder to diagnose.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.metadata import (
    metadata_with_tool_profile,
    summarize_tool_profiles,
    tool_profile_for_example,
)
from micro_model_agent.dataset.infrastructure.prompting import synthetic_prompt_payload
from micro_model_agent.evaluation.infrastructure.workspace_staged import (
    WORKSPACE_STAGED_SYSTEM_PROMPT,
    is_workspace_staged_example,
    workspace_staged_prompt_payload,
)
from micro_model_agent.repository_ops.infrastructure.catalog import (
    TOOL_ARGUMENT_CONTRACTS,
)
from micro_model_agent.shared.domain.value_objects import EvaluationResult


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
                "tool_profile": summarize_tool_profiles(examples),
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

        if example.kind is DatasetExampleKind.TOOL_USE:
            # Tool-use examples must also match the tool argument schemas.
            if self._has_refusal(example.target) and "tool_name" not in example.target:
                return errors
            errors.extend(self._validate_tool_target(example, prefix))
        if example.kind is DatasetExampleKind.REPAIR:
            errors.extend(self._validate_repair_target(example, prefix))

        return errors

    def _validate_repair_target(self, example: DatasetExample, prefix: str) -> list[str]:
        target = example.target
        if isinstance(target.get("tool_name"), str):
            return self._validate_tool_target(example, prefix)

        patch = target.get("patch")
        final_response = target.get("final_response") or target.get("summary")
        if isinstance(patch, str) and patch.strip():
            return []
        if isinstance(final_response, str) and final_response.strip():
            return []
        return [f"{prefix}: repair target requires patch or final_response"]

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
        has_refusal = self._has_refusal(example.target)

        if example.label.outcome is OutcomeLabel.REJECTED and not has_refusal:
            errors.append(f"{prefix}: rejected examples must include target.refusal")
        if example.label.outcome is not OutcomeLabel.REJECTED and "refusal" in example.target:
            errors.append(f"{prefix}: only rejected examples may include target.refusal")
        if has_refusal and "arguments" in example.target:
            errors.append(f"{prefix}: refusal targets must not include tool arguments")
        return errors

    def _has_refusal(self, target: dict[str, object]) -> bool:
        refusal = target.get("refusal")
        return isinstance(refusal, str) and bool(refusal.strip())

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


class SftJsonlDatasetExporter:
    """Filesystem adapter for supervised fine-tuning JSONL exports."""

    def export_dataset_examples(self, path: Path, examples: list[DatasetExample]) -> None:
        """Export examples in SFT JSONL format."""

        export_sft_jsonl(path, examples)


def export_sft_jsonl(path: Path, examples: list[DatasetExample]) -> None:
    """Export examples in a simple supervised fine-tuning chat JSONL shape."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            # The training backend expects chat-style records: system, user,
            # assistant. The payload mirrors the runtime loop prompt closely so
            # SFT teaches one strict JSON decision instead of a loose dataset
            # dictionary shape.
            user_payload = _sft_user_payload(example)
            assistant_payload = _sft_assistant_payload(example)
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": _sft_system_prompt(example),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            user_payload,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            assistant_payload,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
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
                    **metadata_with_tool_profile(example),
                },
            }
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


_SFT_SYSTEM_PROMPT = (
    "You are MicroModelAgent's workflow executor. "
    "Respond with exactly one JSON object and no markdown. "
    "Use only these tool_name values when making a tool call: repo.search, repo.read, "
    "repo.semantic_search, repo.write_patch, repo.write_files, test.run, git.diff. "
    "Never invent tool names such as shell.command, none, or null. "
    "Tool arguments must use exactly the schema keys shown in tool_schemas; do not rename "
    "fields such as command_name to command. "
    "When response_contract.type is tool_call, you must return tool_name and arguments; "
    "do not return refusal, final_response, or ok. "
    "For tool_call responses, do not return helper or analysis keys such as "
    "argument_keys, argument_values, argument_changes, argument_reconciliation, "
    "selected_tool, or changed_fields. "
    "When response_contract.type is refusal, return refusal only and no tool call. "
    "When response_contract.type is final_response, return final_response and ok only. "
    "For safe unfinished work, choose one available typed tool call using "
    '{"tool_name":"repo.read","arguments":{"files":[{"path":"README.md"}]},'
    '"reason":"..."}. '
    "When the request is unsafe or impossible, return "
    '{"refusal":"Concise reason the request cannot be performed safely."}. '
    "When the task is complete and no tool call is needed, return "
    '{"final_response":"Concise answer to the user.","ok":true}.'
)

_TRACE_SFT_SYSTEM_PROMPT = (
    "You are MicroModelAgent replaying a held-out workflow trace. "
    "Respond with exactly one JSON object and no markdown. "
    "Include final_response when the task is complete, "
    "patch when a code change is required, "
    "and tool_history when tool calls were part of the workflow."
)


def _sft_system_prompt(example: DatasetExample) -> str:
    """Return the system prompt that matches the example's evaluation surface."""

    if example.kind is DatasetExampleKind.EVALUATION:
        if is_workspace_staged_example(example):
            return WORKSPACE_STAGED_SYSTEM_PROMPT
        return _TRACE_SFT_SYSTEM_PROMPT
    return _SFT_SYSTEM_PROMPT


def _sft_user_payload(example: DatasetExample) -> dict[str, object]:
    """Build the model-facing training prompt payload for one example."""

    if example.kind is DatasetExampleKind.EVALUATION:
        if is_workspace_staged_example(example):
            return workspace_staged_prompt_payload(example)
        return _sft_trace_user_payload(example)

    return synthetic_prompt_payload(example, include_tool_schemas=True)


def _sft_trace_user_payload(example: DatasetExample) -> dict[str, object]:
    """Build the same trace payload shape used by held-out trace evaluation."""

    return {
        "goal": example.input.get("goal", ""),
        "available_tools": tool_profile_for_example(
            example,
            default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
        )["available_tools"],
        "retrieved_context": example.input.get("retrieved_context", {}),
        "tool_history": example.input.get("tool_history", []),
        "steps": example.input.get("steps", []),
    }


def _sft_assistant_payload(example: DatasetExample) -> dict[str, object]:
    """Canonicalize dataset targets to a strict runtime JSON decision."""

    if example.kind is DatasetExampleKind.EVALUATION:
        if is_workspace_staged_example(example):
            return _sft_workspace_staged_assistant_payload(example)
        return _sft_trace_assistant_payload(example)

    target = example.target
    refusal = target.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return {"refusal": refusal}

    final_response = target.get("final_response")
    if isinstance(final_response, str) and final_response.strip():
        ok = target.get("ok", True)
        return {"final_response": final_response, "ok": ok if isinstance(ok, bool) else True}

    tool_name = target.get("tool_name")
    arguments = target.get("arguments")
    if isinstance(tool_name, str) and isinstance(arguments, dict):
        payload: dict[str, object] = {
            "tool_name": tool_name,
            "arguments": arguments,
        }
        reason = target.get("reason")
        if isinstance(reason, str) and reason.strip():
            payload["reason"] = reason
        return payload

    return dict(target)


def _sft_trace_assistant_payload(example: DatasetExample) -> dict[str, object]:
    """Build the trace-evaluation response shape expected by trace scoring."""

    target = example.target
    payload: dict[str, object] = {}

    final_response = target.get("final_response") or target.get("summary")
    if isinstance(final_response, str) and final_response.strip():
        payload["final_response"] = final_response

    patch = target.get("patch")
    if isinstance(patch, str) and patch.strip():
        payload["patch"] = patch

    changed_files = target.get("changed_files")
    if isinstance(changed_files, list) and all(isinstance(path, str) for path in changed_files):
        payload["changed_files"] = changed_files

    tool_history = _sft_trace_tool_history(example)
    if tool_history:
        payload["tool_history"] = tool_history

    refusal = target.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        payload["refusal"] = refusal

    return payload or dict(target)


def _sft_workspace_staged_assistant_payload(example: DatasetExample) -> dict[str, object]:
    """Return the reviewed gold staged workspace answer for SFT."""

    gold_response = example.target.get("gold_response")
    if isinstance(gold_response, dict):
        return gold_response
    return dict(example.target)


def _sft_trace_tool_history(example: DatasetExample) -> list[dict[str, object]]:
    """Return compact tool calls from trace input history for SFT targets."""

    history = example.input.get("tool_history")
    if not isinstance(history, list):
        return []

    tool_calls: list[dict[str, object]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        tool_call = item.get("tool_call")
        if not isinstance(tool_call, dict):
            continue
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments")
        if isinstance(tool_name, str):
            entry: dict[str, object] = {"tool_name": tool_name}
            if isinstance(arguments, dict):
                entry["arguments"] = arguments
            tool_calls.append(entry)
    return tool_calls
