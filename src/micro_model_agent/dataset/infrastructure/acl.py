"""Anti-Corruption Layer: execution → dataset context translation.

This module translates execution-context objects (WorkflowTrace) into
dataset-context objects (DatasetExample) without the dataset domain importing
the execution domain directly.  Any future cross-context event (WorkflowCompleted)
would be translated here before being consumed by the dataset application layer.
"""

from __future__ import annotations

from typing import Any

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
)
from micro_model_agent.execution.domain.value_objects import WorkflowTrace


class ExecutionToDatasetTranslator:
    """Translate an execution WorkflowTrace into a DatasetExample.

    The translator is the canonical home for the shape-mapping logic that was
    previously owned by ``TraceDatasetBuilder`` in the execution application
    layer.  Placing it in ``dataset/infrastructure/acl.py`` makes the
    cross-context dependency explicit and directional: dataset infrastructure
    depends on execution domain types, but the execution context never imports
    dataset types.
    """

    def translate(
        self,
        trace: WorkflowTrace,
        label: DatasetLabel,
        *,
        kind: DatasetExampleKind = DatasetExampleKind.REPAIR,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetExample:
        """Convert a workflow trace into a fine-tuning dataset example."""

        patch = self._generated_patch(trace)
        input_payload: dict[str, Any] = {
            "goal": trace.goal,
            "retrieved_context": self._retrieval_output(trace),
            "steps": [step.name for step in trace.steps],
            "tool_history": self._tool_history(trace),
        }
        target_payload: dict[str, Any] = {
            "patch": patch,
            "summary": (
                trace.final_output.get("summary") or trace.final_output.get("response")
            ),
            "final_response": trace.final_output.get("response"),
            "changed_files": list(trace.final_output.get("changed_files", [])),
        }
        return DatasetExample(
            kind=kind,
            input=input_payload,
            target=target_payload,
            label=label,
            source=f"trace:{trace.id}",
            metadata={
                "trace_id": str(trace.id),
                "workflow_status": trace.status.value,
                **(metadata or {}),
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generated_patch(self, trace: WorkflowTrace) -> str | None:
        for step in trace.steps:
            if step.name == "generate_patch":
                patch = step.output.get("patch")
                return str(patch) if patch is not None else None
        return None

    def _retrieval_output(self, trace: WorkflowTrace) -> dict[str, Any]:
        for step in trace.steps:
            if step.name == "retrieve_context" and step.tool_result:
                return dict(step.tool_result.output)
        return {}

    def _tool_history(self, trace: WorkflowTrace) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for step in trace.steps:
            if not step.tool_call and not step.tool_result:
                continue
            item: dict[str, Any] = {"step": step.name, "status": step.status.value}
            if step.tool_call:
                item["tool_call"] = {
                    "tool_name": step.tool_call.tool_name,
                    "arguments": step.tool_call.arguments,
                }
            if step.tool_result:
                item["tool_result"] = {
                    "tool_name": step.tool_result.tool_name,
                    "ok": step.tool_result.ok,
                    "output": step.tool_result.output,
                    "error": step.tool_result.error,
                }
            history.append(item)
        return history
