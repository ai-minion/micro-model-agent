"""Domain exceptions for the dataset bounded context."""

from __future__ import annotations

from uuid import UUID

from micro_model_agent.shared.domain.exceptions import DomainException


class DuplicateExampleError(DomainException):
    """Raised when an example with the same ID already exists in the Dataset."""

    def __init__(self, example_id: UUID) -> None:
        super().__init__(f"Example {example_id} already exists in the dataset")
        self.example_id = example_id


class ExampleNotFoundError(DomainException):
    """Raised when a requested DatasetExample does not exist in the Dataset."""

    def __init__(self, example_id: UUID) -> None:
        super().__init__(f"Example {example_id} not found in the dataset")
        self.example_id = example_id


class InvalidLabelError(DomainException):
    """Raised when a supplied DatasetLabel violates invariants."""
