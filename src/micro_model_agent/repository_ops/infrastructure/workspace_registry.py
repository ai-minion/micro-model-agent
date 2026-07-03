"""Workspace registry for globally configured MCP servers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class WorkspaceRecord:
    """One named workspace that MCP tools can target by ID."""

    path: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def workspace_record_to_dict(record: WorkspaceRecord) -> dict[str, Any]:
    """Convert a workspace record to JSON-friendly primitives."""

    return {
        "id": str(record.id),
        "path": record.path,
        "name": record.name,
        "metadata": record.metadata,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


def workspace_record_from_dict(raw: dict[str, Any]) -> WorkspaceRecord:
    """Rebuild a workspace record from a JSON object."""

    return WorkspaceRecord(
        id=UUID(str(raw["id"])),
        path=str(raw["path"]),
        name=str(raw["name"]) if raw.get("name") is not None else None,
        metadata=dict(raw.get("metadata", {})),
        created_at=datetime.fromisoformat(str(raw["created_at"])),
        updated_at=datetime.fromisoformat(str(raw["updated_at"])),
    )


class JsonlWorkspaceRegistry:
    """Append-only workspace registry."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def save(self, record: WorkspaceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(workspace_record_to_dict(record), sort_keys=True))
            file.write("\n")

    async def get(self, workspace_id: str) -> WorkspaceRecord | None:
        records = await self.list()
        for record in reversed(records):
            if str(record.id) == workspace_id:
                return record
        return None

    async def list(self) -> list[WorkspaceRecord]:
        """List newest saved versions of each workspace in first-seen order."""

        if not self.path.exists():
            return []

        records_by_id: dict[str, WorkspaceRecord] = {}
        order: list[str] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}:{line_number}: invalid JSONL record") from exc
            record = workspace_record_from_dict(raw)
            record_id = str(record.id)
            if record_id not in records_by_id:
                order.append(record_id)
            records_by_id[record_id] = record
        return [records_by_id[record_id] for record_id in order]
