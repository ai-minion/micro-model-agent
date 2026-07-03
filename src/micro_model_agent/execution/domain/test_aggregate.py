"""Domain invariant and event tests for the execution bounded context."""

from __future__ import annotations

from uuid import uuid4

import pytest

from micro_model_agent.execution.domain.aggregate import WorkflowExecution, WorkflowStepEntity
from micro_model_agent.execution.domain.events import (
    StepAdded,
    WorkflowCompleted,
    WorkflowFailed,
    WorkflowStarted,
)
from micro_model_agent.execution.domain.exceptions import InvalidTransitionError
from micro_model_agent.execution.domain.services import WorkflowEvaluationService
from micro_model_agent.execution.domain.value_objects import WorkflowStatus, WorkflowTrace

# ---------------------------------------------------------------------------
# WorkflowExecution aggregate
# ---------------------------------------------------------------------------


def test_new_execution_is_pending() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    assert ex.status == WorkflowStatus.PENDING


def test_start_transitions_to_running_and_emits_event() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    assert ex.status == WorkflowStatus.RUNNING
    events = ex.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], WorkflowStarted)
    assert events[0].goal == "fix the bug"
    assert events[0].execution_id == ex.id


def test_start_twice_raises() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    with pytest.raises(InvalidTransitionError):
        ex.start()


def test_add_step_emits_event() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    ex.pull_events()  # drain WorkflowStarted

    step = WorkflowStepEntity(name="retrieve_context", id=uuid4())
    ex.add_step(step)

    events = ex.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], StepAdded)
    assert events[0].step_name == "retrieve_context"


def test_add_step_to_pending_raises() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    step = WorkflowStepEntity(name="retrieve_context", id=uuid4())
    with pytest.raises(InvalidTransitionError):
        ex.add_step(step)


def test_complete_transitions_to_succeeded_and_emits_event() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    ex.pull_events()

    output = {"ok": True, "patch_applied": True}
    ex.complete(output)

    assert ex.status == WorkflowStatus.SUCCEEDED
    assert ex.final_output == output
    events = ex.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], WorkflowCompleted)
    assert events[0].output == output


def test_complete_pending_raises() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    with pytest.raises(InvalidTransitionError):
        ex.complete({})


def test_fail_transitions_to_failed_and_emits_event() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    ex.pull_events()

    ex.fail("model timed out")

    assert ex.status == WorkflowStatus.FAILED
    events = ex.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], WorkflowFailed)
    assert events[0].reason == "model timed out"


def test_fail_succeeded_raises() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    ex.complete({"ok": True})
    with pytest.raises(InvalidTransitionError):
        ex.fail("too late")


def test_pull_events_clears_list() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    first = ex.pull_events()
    second = ex.pull_events()
    assert len(first) == 1
    assert len(second) == 0


def test_to_snapshot_returns_workflow_trace() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    step = WorkflowStepEntity(name="retrieve_context", id=uuid4())
    ex.add_step(step)
    ex.complete({"ok": True})

    trace = ex.to_snapshot()
    assert isinstance(trace, WorkflowTrace)
    assert trace.id == ex.id
    assert trace.goal == "fix the bug"
    assert trace.status == WorkflowStatus.SUCCEEDED
    assert len(trace.steps) == 1
    assert trace.steps[0].name == "retrieve_context"


def test_from_snapshot_round_trips() -> None:
    ex = WorkflowExecution(goal="fix the bug")
    ex.start()
    ex.complete({"ok": True})
    ex.pull_events()  # drain

    trace = ex.to_snapshot()
    restored = WorkflowExecution.from_snapshot(trace)

    assert restored.id == ex.id
    assert restored.status == WorkflowStatus.SUCCEEDED
    assert restored.goal == "fix the bug"
    assert len(restored.pull_events()) == 0  # no pending events on restore


# ---------------------------------------------------------------------------
# WorkflowEvaluationService
# ---------------------------------------------------------------------------


def _make_trace(**kwargs: object) -> WorkflowTrace:
    ex = WorkflowExecution(goal="goal")
    ex.start()
    ex.complete({"ok": True, "patch_applied": True, "verification_passed": True, **kwargs})
    return ex.to_snapshot()


def test_evaluation_passes_when_ok() -> None:
    svc = WorkflowEvaluationService()
    trace = _make_trace()
    result = svc.evaluate(trace)
    assert result.passed is True
    assert result.score == 1.0


def test_evaluation_fails_when_not_ok() -> None:
    svc = WorkflowEvaluationService()
    ex = WorkflowExecution(goal="goal")
    ex.start()
    ex.fail("error")
    trace = ex.to_snapshot()
    result = svc.evaluate(trace)
    assert result.passed is False
    assert result.score == 0.0


def test_evaluation_score_penalised_without_verification() -> None:
    svc = WorkflowEvaluationService()
    ex = WorkflowExecution(goal="goal")
    ex.start()
    ex.complete({"ok": True, "patch_applied": True})  # no verification_passed key
    trace = ex.to_snapshot()
    result = svc.evaluate(trace)
    assert result.score == 0.8


def test_evaluation_score_penalised_without_patch() -> None:
    svc = WorkflowEvaluationService()
    ex = WorkflowExecution(goal="goal")
    ex.start()
    ex.complete({"ok": True, "patch_applied": False, "verification_passed": True})
    trace = ex.to_snapshot()
    result = svc.evaluate(trace)
    assert result.score == 0.7
