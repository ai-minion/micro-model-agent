"""Comparison trace sessions for shadowing local model runs against a human agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class ComparisonTraceEvent:
    """One event recorded during a comparison trace session."""

    event_type: str
    actor: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class ComparisonTraceSession:
    """A trace session comparing MCP local-model output with consumer work."""

    goal: str
    repository_root: str
    status: str = "running"
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[ComparisonTraceEvent] = field(default_factory=list)
    local_trace_ids: list[str] = field(default_factory=list)
    actual_result: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def comparison_session_to_record(session: ComparisonTraceSession) -> dict[str, Any]:
    """Convert a comparison trace session to JSON-friendly primitives."""

    return {
        "id": str(session.id),
        "goal": session.goal,
        "repository_root": session.repository_root,
        "status": session.status,
        "context": session.context,
        "metadata": session.metadata,
        "events": [_comparison_event_to_record(event) for event in session.events],
        "local_trace_ids": session.local_trace_ids,
        "actual_result": session.actual_result,
        "review": session.review,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


def comparison_session_from_record(record: dict[str, Any]) -> ComparisonTraceSession:
    """Rebuild a comparison trace session from a JSON record."""

    return ComparisonTraceSession(
        id=UUID(record["id"]),
        goal=str(record["goal"]),
        repository_root=str(record["repository_root"]),
        status=str(record.get("status", "running")),
        context=str(record.get("context", "")),
        metadata=dict(record.get("metadata", {})),
        events=[
            _comparison_event_from_record(event)
            for event in record.get("events", [])
            if isinstance(event, dict)
        ],
        local_trace_ids=[
            str(trace_id) for trace_id in record.get("local_trace_ids", [])
        ],
        actual_result=dict(record.get("actual_result", {})),
        review=dict(record.get("review", {})),
        created_at=datetime.fromisoformat(record["created_at"]),
        updated_at=datetime.fromisoformat(record["updated_at"]),
    )


def add_comparison_event(
    session: ComparisonTraceSession,
    *,
    event_type: str,
    actor: str,
    payload: dict[str, Any] | None = None,
) -> ComparisonTraceSession:
    """Return a session with one appended event."""

    event = ComparisonTraceEvent(
        event_type=event_type,
        actor=actor,
        payload=payload or {},
    )
    local_trace_ids = list(session.local_trace_ids)
    trace_id = event.payload.get("trace_id")
    if event_type == "local_model_run" and isinstance(trace_id, str):
        local_trace_ids.append(trace_id)
    return replace(
        session,
        events=[*session.events, event],
        local_trace_ids=list(dict.fromkeys(local_trace_ids)),
        updated_at=datetime.now(UTC),
    )


def stop_comparison_session(
    session: ComparisonTraceSession,
    *,
    actual_result: dict[str, Any],
) -> ComparisonTraceSession:
    """Return a stopped session with the consumer's actual result attached."""

    return replace(
        session,
        status="stopped",
        actual_result=actual_result,
        updated_at=datetime.now(UTC),
    )


def review_comparison_session(
    session: ComparisonTraceSession,
    *,
    review: dict[str, Any],
) -> ComparisonTraceSession:
    """Return a session with human or consumer review details attached."""

    return replace(
        session,
        review=review,
        updated_at=datetime.now(UTC),
    )


class JsonlComparisonTraceStore:
    """Comparison trace store with per-session folder and JSONL index.

    Layout::

        <root>/
          comparison_sessions.jsonl     # append-only index (id, goal, status)
          <session-id>/
            metadata.json               # full session record (canonical)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._root = self.path.parent

    def _session_dir(self, session_id: str) -> Path:
        return self._root / session_id

    async def save(self, session: ComparisonTraceSession) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        session_id = str(session.id)
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        record = comparison_session_to_record(session)

        # Canonical full record in the per-session folder.
        (session_dir / "metadata.json").write_text(
            json.dumps(record, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        # Lightweight index entry: id, goal, status, timestamps only.
        index_entry = {
            "id": record["id"],
            "goal": record["goal"],
            "status": record["status"],
            "repository_root": record["repository_root"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(index_entry, sort_keys=True))
            file.write("\n")

    async def get(self, session_id: str) -> ComparisonTraceSession | None:
        meta_file = self._session_dir(session_id) / "metadata.json"
        if meta_file.exists():
            return comparison_session_from_record(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
        return None

    async def list(self) -> list[ComparisonTraceSession]:
        """List sessions in first-seen order, loading each from its folder."""

        if not self.path.exists():
            return []

        seen: dict[str, bool] = {}
        order: list[str] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}:{line_number}: invalid JSONL record") from exc
            session_id = str(entry["id"])
            if session_id not in seen:
                seen[session_id] = True
                order.append(session_id)

        sessions: list[ComparisonTraceSession] = []
        for session_id in order:
            session = await self.get(session_id)
            if session is not None:
                sessions.append(session)
        return sessions


def _comparison_event_to_record(event: ComparisonTraceEvent) -> dict[str, Any]:
    return {
        "id": str(event.id),
        "event_type": event.event_type,
        "actor": event.actor,
        "payload": event.payload,
        "created_at": event.created_at.isoformat(),
    }


def _comparison_event_from_record(record: dict[str, Any]) -> ComparisonTraceEvent:
    return ComparisonTraceEvent(
        id=UUID(record["id"]),
        event_type=str(record["event_type"]),
        actor=str(record["actor"]),
        payload=dict(record.get("payload", {})),
        created_at=datetime.fromisoformat(record["created_at"]),
    )
