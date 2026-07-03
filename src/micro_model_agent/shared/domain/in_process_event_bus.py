"""In-process synchronous event bus implementation.

This bus is suitable for single-process deployments and tests.  Handlers are
registered per event type and called synchronously in registration order.
For production workloads that need durability or cross-process delivery, swap
this out for a Kafka/RabbitMQ/Redis adapter behind the same ``EventBus`` Protocol.
"""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from micro_model_agent.shared.domain.domain_event import DomainEvent

_log = logging.getLogger(__name__)

# A handler is any async callable that accepts a single DomainEvent argument.
Handler = Callable[[Any], Coroutine[Any, Any, None]]

T = TypeVar("T", bound=DomainEvent)


class InProcessEventBus:
    """Simple in-process event bus.

    Usage::

        bus = InProcessEventBus()
        bus.subscribe(WorkflowCompleted, my_handler)
        await bus.publish(WorkflowCompleted(execution_id=..., output={}))
    """

    def __init__(self) -> None:
        self._handlers: dict[type[DomainEvent], list[Handler]] = defaultdict(list)

    def subscribe(
        self, event_type: type[T], handler: Callable[[T], Coroutine[Any, Any, None]]
    ) -> None:
        """Register *handler* to be called whenever *event_type* is published."""

        self._handlers[event_type].append(handler)

    def unsubscribe(
        self, event_type: type[T], handler: Callable[[T], Coroutine[Any, Any, None]]
    ) -> None:
        """Remove a previously registered handler (no-op if not registered)."""

        with contextlib.suppress(ValueError):
            self._handlers[event_type].remove(handler)

    async def publish(self, event: DomainEvent) -> None:
        """Dispatch *event* to all registered handlers for its exact type.

        Errors in handlers are logged but do not prevent other handlers from
        running or propagate to the caller.
        """

        for handler in list(self._handlers.get(type(event), [])):
            try:
                await handler(event)
            except Exception:
                _log.exception(
                    "Event handler %r raised while processing %r",
                    handler,
                    event,
                )

    async def publish_all(self, events: list[DomainEvent]) -> None:
        """Publish each event in *events* sequentially."""

        for event in events:
            await self.publish(event)
