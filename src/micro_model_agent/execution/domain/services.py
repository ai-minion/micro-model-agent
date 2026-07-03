"""Execution context domain services.

Domain services contain stateless business logic that operates on domain objects
from the execution context. They have zero infrastructure dependencies and can be
unit-tested without mocks.
"""

from __future__ import annotations

from micro_model_agent.execution.domain.value_objects import WorkflowTrace
from micro_model_agent.shared.domain.value_objects import EvaluationResult


class WorkflowEvaluationService:
    """Evaluate a completed coding workflow from its captured trace.

    This is the canonical home for workflow-outcome scoring logic.  The service
    is stateless: call ``evaluate`` with a trace, receive an ``EvaluationResult``.
    """

    def evaluate(self, trace: WorkflowTrace) -> EvaluationResult:
        """Score a finished workflow trace and return a structured result."""

        final = trace.final_output
        ok = bool(final.get("ok"))
        verification_passed = final.get("verification_passed")
        patch_applied = bool(final.get("patch_applied"))

        score = 1.0 if ok else 0.0
        if ok and verification_passed is None:
            score = 0.8
        if ok and not patch_applied:
            score = min(score, 0.7)

        return EvaluationResult(
            passed=ok,
            summary="workflow passed" if ok else "workflow failed",
            score=score,
            details={
                "trace_id": str(trace.id),
                "status": trace.status.value,
                "patch_applied": patch_applied,
                "verification_passed": verification_passed,
                "changed_files": list(final.get("changed_files", [])),
            },
        )
