"""Domain exceptions for the execution bounded context."""

from __future__ import annotations

from micro_model_agent.shared.domain.exceptions import DomainException


class InvalidTransitionError(DomainException):
    """Raised when a WorkflowExecution state transition is not allowed."""
