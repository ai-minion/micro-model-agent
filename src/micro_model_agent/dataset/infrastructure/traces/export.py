"""Export stored workflow traces into reviewable dataset examples."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.traces.review import TraceReview
from micro_model_agent.execution.application.workflows import TraceDatasetBuilder
from micro_model_agent.execution.domain.value_objects import (
    WorkflowStatus,
    WorkflowTrace,
)

SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|auth|credential|password|secret|token)", re.IGNORECASE
)
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(sk-[a-z0-9_-]{16,}|ghp_[a-z0-9_]{16,}|xox[baprs]-[a-z0-9-]{16,})"
)


class TraceSafetyError(ValueError):
    """Raised when a trace is unsafe to export without review."""


class TraceDatasetExporter:
    """Build redacted dataset examples from stored workflow traces."""

    def __init__(self, builder: TraceDatasetBuilder | None = None) -> None:
        self.builder = builder or TraceDatasetBuilder()

    def export(
        self,
        traces: list[WorkflowTrace],
        *,
        kind: DatasetExampleKind = DatasetExampleKind.REPAIR,
        label_mode: str = "review",
        reviews_by_trace_id: dict[str, TraceReview] | None = None,
        outcome: OutcomeLabel | None = None,
        quality: QualityLabel | None = None,
        workflow_status: WorkflowStatus | None = None,
        require_tool_call: bool = False,
        max_examples: int | None = None,
    ) -> list[DatasetExample]:
        """Convert traces to redacted dataset examples after filtering."""

        examples: list[DatasetExample] = []
        for trace in traces:
            if workflow_status is not None and trace.status is not workflow_status:
                continue
            if require_tool_call and not _has_tool_call(trace):
                continue

            review = (reviews_by_trace_id or {}).get(str(trace.id))
            if label_mode == "reviewed" and review is None:
                continue
            label = self._label_for_trace(trace, label_mode, review)
            if outcome is not None and label.outcome is not outcome:
                continue
            if quality is not None and label.quality is not quality:
                continue

            example = self.builder.build_example(
                trace,
                label,
                kind=kind,
                metadata={
                    "export_label_mode": label_mode,
                    "review_required": label.quality is QualityLabel.UNKNOWN,
                    "review_id": str(review.id) if review else None,
                    "has_corrected_target": bool(review and review.corrected_target),
                },
            )
            if review is not None and review.corrected_target is not None:
                example = replace(example, target=review.corrected_target)
            examples.append(redact_dataset_example(example))
            if max_examples is not None and len(examples) >= max_examples:
                break
        return examples

    def _label_for_trace(
        self,
        trace: WorkflowTrace,
        label_mode: str,
        review: TraceReview | None,
    ) -> DatasetLabel:
        if label_mode == "review":
            return DatasetLabel(
                outcome=OutcomeLabel.NEEDS_REVIEW,
                quality=QualityLabel.UNKNOWN,
                reviewer_notes="Exported from stored trace; requires human review.",
            )
        if label_mode == "reviewed":
            if review is None:
                raise ValueError("reviewed label mode requires a trace review")
            return review.label
        if label_mode == "evaluation":
            return _label_from_trace_evaluation(trace)
        raise ValueError(f"unsupported trace export label mode: {label_mode}")


class LocalTraceDatasetExporter:
    """Adapter exposing trace export through the application port shape."""

    def __init__(self, exporter: TraceDatasetExporter | None = None) -> None:
        self.exporter = exporter or TraceDatasetExporter()

    def export_trace_dataset_examples(
        self,
        traces: list[WorkflowTrace],
        *,
        kind: DatasetExampleKind = DatasetExampleKind.REPAIR,
        label_mode: str = "review",
        reviews_by_trace_id: Mapping[str, object],
        outcome: OutcomeLabel | None = None,
        quality: QualityLabel | None = None,
        workflow_status: WorkflowStatus | None = None,
        require_tool_call: bool = False,
        max_examples: int | None = None,
    ) -> list[DatasetExample]:
        """Export trace-derived dataset examples."""

        return self.exporter.export(
            traces,
            kind=kind,
            label_mode=label_mode,
            reviews_by_trace_id=cast(dict[str, TraceReview], dict(reviews_by_trace_id)),
            outcome=outcome,
            quality=quality,
            workflow_status=workflow_status,
            require_tool_call=require_tool_call,
            max_examples=max_examples,
        )


class LocalTraceDatasetExportValidator:
    """Adapter exposing trace export validation through an application port."""

    def validate_trace_dataset_examples(self, examples: list[DatasetExample]) -> list[str]:
        """Return trace export validation errors."""

        return validate_trace_export_examples(examples)


def validate_trace_export_examples(examples: list[DatasetExample]) -> list[str]:
    """Validate that trace exports are well-formed review records."""

    errors: list[str] = []
    if not examples:
        errors.append("trace export contains no examples")

    for index, example in enumerate(examples):
        prefix = f"example[{index}]"
        if not example.input:
            errors.append(f"{prefix}: input is required")
        if not example.target:
            errors.append(f"{prefix}: target is required")
        if not example.source.startswith("trace:"):
            errors.append(f"{prefix}: source must reference a trace")
        trace_id = example.metadata.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            errors.append(f"{prefix}: metadata.trace_id is required")
    return errors


def _has_tool_call(trace: WorkflowTrace) -> bool:
    return any(step.tool_call is not None for step in trace.steps)


def _label_from_trace_evaluation(trace: WorkflowTrace) -> DatasetLabel:
    evaluation = trace.final_output.get("evaluation")
    if not isinstance(evaluation, dict):
        return DatasetLabel(
            outcome=OutcomeLabel.NEEDS_REVIEW,
            quality=QualityLabel.UNKNOWN,
            reviewer_notes="Trace has no evaluation metadata; requires human review.",
        )

    passed = bool(evaluation.get("passed"))
    if passed:
        return DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD)

    failure_modes: tuple[FailureMode, ...] = (FailureMode.BAD_PATCH,)
    details = evaluation.get("details")
    if isinstance(details, dict) and details.get("verification_passed") is False:
        failure_modes = (FailureMode.TEST_FAILED,)
    return DatasetLabel(
        outcome=OutcomeLabel.REJECTED,
        quality=QualityLabel.BAD,
        failure_modes=failure_modes,
    )


def redact_dataset_example(example: DatasetExample) -> DatasetExample:
    """Return a copy of an example with common secrets redacted."""

    input_payload, input_redactions = _redact_value(example.input)
    target_payload, target_redactions = _redact_value(example.target)
    metadata, metadata_redactions = _redact_value(example.metadata)
    total_redactions = input_redactions + target_redactions + metadata_redactions
    if not isinstance(input_payload, dict):
        raise TraceSafetyError("redacted dataset input is not a JSON object")
    if not isinstance(target_payload, dict):
        raise TraceSafetyError("redacted dataset target is not a JSON object")
    if not isinstance(metadata, dict):
        raise TraceSafetyError("redacted dataset metadata is not a JSON object")

    if total_redactions:
        metadata = {
            **metadata,
            "redacted": True,
            "redaction_count": total_redactions,
        }
    return replace(example, input=input_payload, target=target_payload, metadata=metadata)


def _redact_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        redactions = 0
        for key, item in value.items():
            key_text = str(key)
            if SECRET_KEY_PATTERN.search(key_text):
                redacted[key_text] = "[REDACTED]"
                redactions += 1
                continue
            redacted_item, item_redactions = _redact_value(item)
            redacted[key_text] = redacted_item
            redactions += item_redactions
        return redacted, redactions

    if isinstance(value, list):
        redacted_items: list[Any] = []
        redactions = 0
        for item in value:
            redacted_item, item_redactions = _redact_value(item)
            redacted_items.append(redacted_item)
            redactions += item_redactions
        return redacted_items, redactions

    if isinstance(value, str):
        redacted_text, count = SECRET_VALUE_PATTERN.subn("[REDACTED]", value)
        return redacted_text, count

    return value, 0
