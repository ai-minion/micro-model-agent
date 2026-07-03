"""Tests for InProcessEventBus."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any
from uuid import uuid4

from micro_model_agent.execution.domain.events import WorkflowCompleted, WorkflowStarted
from micro_model_agent.shared.domain.domain_event import DomainEvent
from micro_model_agent.shared.domain.in_process_event_bus import InProcessEventBus


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def test_published_event_reaches_handler() -> None:
    bus = InProcessEventBus()
    received: list[WorkflowStarted] = []

    async def handler(event: WorkflowStarted) -> None:
        received.append(event)

    bus.subscribe(WorkflowStarted, handler)
    event = WorkflowStarted(execution_id=uuid4(), goal="fix bug")
    _run(bus.publish(event))

    assert len(received) == 1
    assert received[0] is event


def test_handler_only_called_for_subscribed_type() -> None:
    bus = InProcessEventBus()
    called: list[bool] = []

    async def handler(event: WorkflowStarted) -> None:
        called.append(True)

    bus.subscribe(WorkflowStarted, handler)
    _run(bus.publish(WorkflowCompleted(execution_id=uuid4(), output={})))

    assert called == []


def test_multiple_handlers_all_called() -> None:
    bus = InProcessEventBus()
    log: list[int] = []

    async def h1(event: WorkflowStarted) -> None:
        log.append(1)

    async def h2(event: WorkflowStarted) -> None:
        log.append(2)

    bus.subscribe(WorkflowStarted, h1)
    bus.subscribe(WorkflowStarted, h2)
    _run(bus.publish(WorkflowStarted(execution_id=uuid4(), goal="x")))

    assert sorted(log) == [1, 2]


def test_publish_all_dispatches_each_event() -> None:
    bus = InProcessEventBus()
    ids: list[str] = []

    async def handler(event: WorkflowStarted) -> None:
        ids.append(event.goal)

    bus.subscribe(WorkflowStarted, handler)
    events: list[DomainEvent] = [
        WorkflowStarted(execution_id=uuid4(), goal="a"),
        WorkflowStarted(execution_id=uuid4(), goal="b"),
    ]
    _run(bus.publish_all(events))

    assert ids == ["a", "b"]


def test_handler_error_does_not_prevent_other_handlers() -> None:
    bus = InProcessEventBus()
    reached: list[bool] = []

    async def bad_handler(event: WorkflowStarted) -> None:
        raise RuntimeError("boom")

    async def good_handler(event: WorkflowStarted) -> None:
        reached.append(True)

    bus.subscribe(WorkflowStarted, bad_handler)
    bus.subscribe(WorkflowStarted, good_handler)
    _run(bus.publish(WorkflowStarted(execution_id=uuid4(), goal="test")))

    assert reached == [True]


def test_unsubscribe_stops_calls() -> None:
    bus = InProcessEventBus()
    called: list[bool] = []

    async def handler(event: WorkflowStarted) -> None:
        called.append(True)

    bus.subscribe(WorkflowStarted, handler)
    bus.unsubscribe(WorkflowStarted, handler)
    _run(bus.publish(WorkflowStarted(execution_id=uuid4(), goal="test")))

    assert called == []
