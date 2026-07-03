"""Tests for execution domain events and exceptions."""

from __future__ import annotations

from uuid import uuid4

import pytest

from micro_model_agent.execution.domain.events import (
    StepAdded,
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowStarted,
)
from micro_model_agent.execution.domain.exceptions import InvalidTransitionError
from micro_model_agent.shared.domain.domain_event import DomainEvent

# ---------------------------------------------------------------------------
# Domain events
# ---------------------------------------------------------------------------


def test_workflow_started_fields() -> None:
    execution_id = uuid4()
    event = WorkflowStarted(execution_id=execution_id, goal="fix the bug")
    assert event.execution_id == execution_id
    assert event.goal == "fix the bug"
    assert event.event_id is not None
    assert event.occurred_at is not None


def test_step_added_fields() -> None:
    execution_id = uuid4()
    step_id = uuid4()
    event = StepAdded(execution_id=execution_id, step_id=step_id, step_name="retrieve_context")
    assert event.step_id == step_id
    assert event.step_name == "retrieve_context"


def test_workflow_completed_carries_output() -> None:
    output = {"ok": True, "patch_applied": True}
    event = WorkflowCompleted(execution_id=uuid4(), output=output)
    assert event.output["ok"] is True


def test_workflow_failed_carries_reason() -> None:
    event = WorkflowFailed(execution_id=uuid4(), reason="model timeout")
    assert event.reason == "model timeout"


def test_all_execution_events_are_domain_events() -> None:
    ex_id = uuid4()
    events = [
        WorkflowStarted(execution_id=ex_id, goal="x"),
        StepAdded(execution_id=ex_id, step_id=uuid4(), step_name="s"),
        WorkflowCompleted(execution_id=ex_id, output={}),
        WorkflowFailed(execution_id=ex_id, reason="err"),
    ]
    for event in events:
        assert isinstance(event, DomainEvent)


def test_events_are_frozen() -> None:
    event = WorkflowStarted(execution_id=uuid4(), goal="test")
    try:
        event.goal = "other"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


def test_each_event_has_unique_id() -> None:
    ex_id = uuid4()
    e1 = WorkflowStarted(execution_id=ex_id, goal="x")
    e2 = WorkflowStarted(execution_id=ex_id, goal="x")
    assert e1.event_id != e2.event_id


# ---------------------------------------------------------------------------
# InvalidTransitionError
# ---------------------------------------------------------------------------


def test_invalid_transition_error_message() -> None:
    exc = InvalidTransitionError("Cannot start a succeeded workflow")
    assert "Cannot start" in str(exc)


def test_invalid_transition_error_is_domain_exception() -> None:
    from micro_model_agent.shared.domain.exceptions import DomainException
    with pytest.raises(DomainException):
        raise InvalidTransitionError("test")
