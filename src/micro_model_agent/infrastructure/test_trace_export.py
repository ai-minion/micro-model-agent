"""Tests for exporting stored traces into dataset examples."""

from __future__ import annotations

from micro_model_agent.domain.contracts import (
    ToolCall,
    ToolResult,
    WorkflowStatus,
    WorkflowStep,
    WorkflowTrace,
)
from micro_model_agent.domain.datasets import OutcomeLabel, QualityLabel
from micro_model_agent.infrastructure.trace_export import (
    TraceDatasetExporter,
    validate_trace_export_examples,
)


def test_trace_export_defaults_to_review_label_and_redacts_secrets() -> None:
    tool_call = ToolCall(
        tool_name="repo.search",
        arguments={"query": "sk-testtoken1234567890", "api_key": "secret-value"},
    )
    trace = WorkflowTrace(
        goal="Find token use",
        status=WorkflowStatus.SUCCEEDED,
        steps=[
            WorkflowStep(
                name="tool_call_1",
                status=WorkflowStatus.SUCCEEDED,
                tool_call=tool_call,
                tool_result=ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    ok=True,
                    output={"matches": [{"path": "app.py", "token": "abc123"}]},
                ),
            )
        ],
        final_output={"summary": "Found sk-testtoken1234567890", "changed_files": []},
    )

    examples = TraceDatasetExporter().export([trace])

    assert validate_trace_export_examples(examples) == []
    assert examples[0].label.outcome is OutcomeLabel.NEEDS_REVIEW
    assert examples[0].label.quality is QualityLabel.UNKNOWN
    assert examples[0].metadata["redacted"] is True
    assert examples[0].target["summary"] == "Found [REDACTED]"


def test_trace_export_can_filter_evaluation_labels() -> None:
    accepted = WorkflowTrace(
        goal="Good run",
        status=WorkflowStatus.SUCCEEDED,
        final_output={
            "summary": "Done",
            "changed_files": ["app.py"],
            "evaluation": {"passed": True, "details": {"verification_passed": True}},
        },
    )
    rejected = WorkflowTrace(
        goal="Bad run",
        status=WorkflowStatus.FAILED,
        final_output={
            "summary": "Failed",
            "changed_files": [],
            "evaluation": {"passed": False, "details": {"verification_passed": False}},
        },
    )

    examples = TraceDatasetExporter().export(
        [accepted, rejected],
        label_mode="evaluation",
        outcome=OutcomeLabel.ACCEPTED,
        quality=QualityLabel.GOOD,
    )

    assert len(examples) == 1
    assert examples[0].metadata["trace_id"] == str(accepted.id)
    assert examples[0].label.outcome is OutcomeLabel.ACCEPTED
