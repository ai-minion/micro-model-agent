"""EventBus protocol for publishing domain events."""
from __future__ import annotations

from typing import Protocol

from micro_model_agent.shared.domain.domain_event import DomainEvent


class EventBus(Protocol):
    """Anything that can receive and route domain events."""

    async def publish(self, event: DomainEvent) -> None:
        """Publish a single domain event."""

    async def publish_all(self, events: list[DomainEvent]) -> None:
        """Publish a list of domain events."""
