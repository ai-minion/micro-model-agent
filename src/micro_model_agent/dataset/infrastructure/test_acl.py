"""Tests for ExecutionToDatasetTranslator ACL."""

from __future__ import annotations

from uuid import uuid4

from micro_model_agent.dataset.infrastructure.acl import ExecutionToDatasetTranslator
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.execution.domain.value_objects import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)


def _label(
    outcome: OutcomeLabel = OutcomeLabel.ACCEPTED,
    quality: QualityLabel = QualityLabel.GOOD,
) -> DatasetLabel:
    return DatasetLabel(outcome=outcome, quality=quality)


def _trace(
    goal: str = "fix the bug",
    steps: list[WorkflowStep] | None = None,
    final_output: dict | None = None,
) -> WorkflowTrace:
    return WorkflowTrace(
        goal=goal,
        status=WorkflowStatus.SUCCEEDED,
        steps=steps or [],
        final_output=final_output or {"ok": True},
    )


def test_basic_translation_creates_dataset_example() -> None:
    translator = ExecutionToDatasetTranslator()
    trace = _trace()
    example = translator.translate(trace, _label())

    assert isinstance(example, DatasetExample)
    assert example.kind == DatasetExampleKind.REPAIR
    assert example.input["goal"] == "fix the bug"
    assert example.label.outcome == OutcomeLabel.ACCEPTED


def test_source_is_trace_id() -> None:
    translator = ExecutionToDatasetTranslator()
    trace = _trace()
    example = translator.translate(trace, _label())

    assert example.source == f"trace:{trace.id}"


def test_metadata_contains_trace_id_and_status() -> None:
    translator = ExecutionToDatasetTranslator()
    trace = _trace()
    example = translator.translate(trace, _label())

    assert example.metadata["trace_id"] == str(trace.id)
    assert example.metadata["workflow_status"] == "succeeded"


def test_custom_kind() -> None:
    translator = ExecutionToDatasetTranslator()
    trace = _trace()
    example = translator.translate(trace, _label(), kind=DatasetExampleKind.TOOL_USE)

    assert example.kind == DatasetExampleKind.TOOL_USE


def test_extra_metadata_merged() -> None:
    translator = ExecutionToDatasetTranslator()
    trace = _trace()
    example = translator.translate(trace, _label(), metadata={"source_version": "v2"})

    assert example.metadata["source_version"] == "v2"
    assert "trace_id" in example.metadata  # original metadata preserved


def test_patch_extracted_from_generate_patch_step() -> None:
    translator = ExecutionToDatasetTranslator()
    step = WorkflowStep(
        name="generate_patch",
        output={"patch": "--- a/hello.py\n+++ b/hello.py"},
    )
    trace = _trace(steps=[step])
    example = translator.translate(trace, _label())

    assert "--- a/hello.py" in example.target["patch"]


def test_no_patch_step_gives_none_patch() -> None:
    translator = ExecutionToDatasetTranslator()
    trace = _trace(steps=[WorkflowStep(name="retrieve_context")])
    example = translator.translate(trace, _label())

    assert example.target["patch"] is None


def test_retrieval_context_extracted() -> None:
    translator = ExecutionToDatasetTranslator()
    call = ToolCall(tool_name="repo_read", arguments={"path": "auth.py"})
    result = ToolResult(
        tool_call_id=call.id,
        tool_name="repo_read",
        ok=True,
        output={"content": "def authenticate(): ..."},
    )
    step = WorkflowStep(name="retrieve_context", tool_call=call, tool_result=result)
    trace = _trace(steps=[step])
    example = translator.translate(trace, _label())

    assert example.input["retrieved_context"]["content"] == "def authenticate(): ..."


def test_tool_history_included_for_tool_call_steps() -> None:
    translator = ExecutionToDatasetTranslator()
    call = ToolCall(tool_name="git_diff", arguments={})
    result = ToolResult(tool_call_id=call.id, tool_name="git_diff", ok=True)
    step = WorkflowStep(name="check_diff", tool_call=call, tool_result=result)
    trace = _trace(steps=[step])
    example = translator.translate(trace, _label())

    history = example.input["tool_history"]
    assert len(history) == 1
    assert history[0]["tool_call"]["tool_name"] == "git_diff"


def test_steps_list_in_input() -> None:
    translator = ExecutionToDatasetTranslator()
    steps = [WorkflowStep(name="step_a"), WorkflowStep(name="step_b")]
    trace = _trace(steps=steps)
    example = translator.translate(trace, _label())

    assert example.input["steps"] == ["step_a", "step_b"]


def test_final_response_from_final_output() -> None:
    translator = ExecutionToDatasetTranslator()
    trace = _trace(final_output={"ok": True, "response": "Done!", "changed_files": ["hello.py"]})
    example = translator.translate(trace, _label())

    assert example.target["final_response"] == "Done!"
    assert "hello.py" in example.target["changed_files"]
