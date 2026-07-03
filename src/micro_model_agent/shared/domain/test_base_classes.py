"""Tests for shared kernel base classes: DomainEvent, Entity, DomainException."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from micro_model_agent.shared.domain.domain_event import DomainEvent
from micro_model_agent.shared.domain.entity import Entity
from micro_model_agent.shared.domain.exceptions import DomainException

# ---------------------------------------------------------------------------
# DomainEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _PingEvent(DomainEvent):
    message: str


def test_domain_event_auto_id() -> None:
    e1 = _PingEvent(message="hello")
    e2 = _PingEvent(message="hello")
    assert e1.event_id != e2.event_id  # each instance gets its own UUID


def test_domain_event_auto_timestamp() -> None:
    before = datetime.now(UTC)
    event = _PingEvent(message="ts")
    after = datetime.now(UTC)
    assert before <= event.occurred_at <= after


def test_domain_event_is_frozen() -> None:
    event = _PingEvent(message="x")
    try:
        event.message = "y"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass


def test_domain_event_explicit_id_preserved() -> None:
    fixed_id = uuid4()
    event = _PingEvent(event_id=fixed_id, message="x")
    assert event.event_id == fixed_id


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class _ConcreteEntity(Entity):
    def __init__(self, value: str, id: UUID | None = None) -> None:
        super().__init__(id)
        self.value = value


def test_entity_auto_id() -> None:
    e1 = _ConcreteEntity("a")
    e2 = _ConcreteEntity("b")
    assert isinstance(e1.id, UUID)
    assert e1.id != e2.id


def test_entity_explicit_id_preserved() -> None:
    fixed = uuid4()
    entity = _ConcreteEntity("x", id=fixed)
    assert entity.id == fixed


def test_entity_equality_by_identity() -> None:
    fixed = uuid4()
    e1 = _ConcreteEntity("a", id=fixed)
    e2 = _ConcreteEntity("b", id=fixed)  # same id, different value
    assert e1 == e2


def test_entity_inequality_different_ids() -> None:
    e1 = _ConcreteEntity("a")
    e2 = _ConcreteEntity("a")  # same value, different id
    assert e1 != e2


def test_entity_hash_by_id() -> None:
    fixed = uuid4()
    e1 = _ConcreteEntity("a", id=fixed)
    e2 = _ConcreteEntity("b", id=fixed)
    assert hash(e1) == hash(e2)
    assert {e1, e2} == {e1}  # set deduplicates by id


def test_entity_starts_with_empty_events() -> None:
    entity = _ConcreteEntity("x")
    assert entity.pull_events() == []


def test_entity_pull_events_clears_list() -> None:
    entity = _ConcreteEntity("x")
    entity._events.append(_PingEvent(message="test"))
    first = entity.pull_events()
    assert len(first) == 1
    assert entity.pull_events() == []


def test_entity_accumulates_multiple_events() -> None:
    entity = _ConcreteEntity("x")
    for i in range(3):
        entity._events.append(_PingEvent(message=str(i)))
    events = entity.pull_events()
    assert len(events) == 3
    assert [e.message for e in events] == ["0", "1", "2"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# DomainException
# ---------------------------------------------------------------------------


def test_domain_exception_is_exception() -> None:
    exc = DomainException("something went wrong")
    assert isinstance(exc, Exception)
    assert str(exc) == "something went wrong"


def test_domain_exception_subclassing() -> None:
    class _MyError(DomainException):
        pass

    with pytest.raises(DomainException):
        raise _MyError("custom error")

    with pytest.raises(_MyError):
        raise _MyError("custom error")
