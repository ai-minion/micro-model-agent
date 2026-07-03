"""Base class for domain entities."""
from __future__ import annotations

from uuid import UUID, uuid4

from micro_model_agent.shared.domain.domain_event import DomainEvent


class Entity:
    """Equality by identity; owns a pending-events list."""

    def __init__(self, id: UUID | None = None) -> None:
        self.id: UUID = id or uuid4()
        self._events: list[DomainEvent] = []

    def pull_events(self) -> list[DomainEvent]:
        """Return and clear the accumulated domain events."""
        events, self._events = self._events, []
        return events

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
