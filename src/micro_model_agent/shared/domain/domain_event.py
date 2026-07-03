"""Base class for all domain events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Base for all domain events. Immutable; carry only what listeners need."""

    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
