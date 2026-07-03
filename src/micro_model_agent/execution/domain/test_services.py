"""Tests for the WorkflowEvaluationService domain service."""

from __future__ import annotations

from micro_model_agent.execution.domain.aggregate import WorkflowExecution
from micro_model_agent.execution.domain.services import WorkflowEvaluationService
from micro_model_agent.shared.domain.value_objects import EvaluationResult


def _completed_trace(final_output: dict):
    ex = WorkflowExecution(goal="fix")
    ex.start()
    ex.complete(final_output)
    return ex.to_snapshot()


def _failed_trace():
    ex = WorkflowExecution(goal="fix")
    ex.start()
    ex.fail("error")
    return ex.to_snapshot()


class TestWorkflowEvaluationService:
    def setup_method(self) -> None:
        self.svc = WorkflowEvaluationService()

    def test_returns_evaluation_result(self) -> None:
        trace = _completed_trace({"ok": True, "patch_applied": True, "verification_passed": True})
        result = self.svc.evaluate(trace)
        assert isinstance(result, EvaluationResult)

    def test_perfect_run_score_one(self) -> None:
        trace = _completed_trace({"ok": True, "patch_applied": True, "verification_passed": True})
        result = self.svc.evaluate(trace)
        assert result.passed is True
        assert result.score == 1.0

    def test_no_verification_penalises_to_08(self) -> None:
        trace = _completed_trace({"ok": True, "patch_applied": True})
        result = self.svc.evaluate(trace)
        assert result.passed is True
        assert result.score == 0.8

    def test_no_patch_applied_penalises_to_07(self) -> None:
        trace = _completed_trace({"ok": True, "patch_applied": False, "verification_passed": True})
        result = self.svc.evaluate(trace)
        assert result.score == 0.7

    def test_failed_workflow_score_zero(self) -> None:
        trace = _failed_trace()
        result = self.svc.evaluate(trace)
        assert result.passed is False
        assert result.score == 0.0

    def test_details_include_trace_id(self) -> None:
        trace = _completed_trace({"ok": True})
        result = self.svc.evaluate(trace)
        assert "trace_id" in result.details
        assert result.details["trace_id"] == str(trace.id)

    def test_details_include_changed_files(self) -> None:
        trace = _completed_trace({"ok": True, "changed_files": ["auth.py", "tests/test_auth.py"]})
        result = self.svc.evaluate(trace)
        assert result.details["changed_files"] == ["auth.py", "tests/test_auth.py"]

    def test_summary_matches_outcome(self) -> None:
        ok_trace = _completed_trace({"ok": True})
        fail_trace = _failed_trace()
        assert "passed" in self.svc.evaluate(ok_trace).summary
        assert "failed" in self.svc.evaluate(fail_trace).summary
