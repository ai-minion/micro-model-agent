"""JsonlModelRegistryRepository — promotion context DDD repository.

The ModelRegistry is a singleton aggregate — there is typically one registry
per deployment.  It is stored as a single JSON file at ``<root>/registry.json``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

from micro_model_agent.promotion.domain.aggregate import ModelRegistry, PromotedModel


class JsonlModelRegistryRepository:
    """DDD-style ModelRegistryRepository backed by a single JSON file."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._registry_path = self._root / "registry.json"

    def _write(self, registry: ModelRegistry) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        record = {
            "id": str(registry.id),
            "models": [
                {
                    "id": str(m.id),
                    "artifact_id": str(m.artifact_id),
                    "artifact_name": m.artifact_name,
                    "artifact_path": m.artifact_path,
                    "base_model": m.base_model,
                    "promoted_at": m.promoted_at.isoformat(),
                }
                for m in registry.list_models()
            ],
        }
        self._registry_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _read(self) -> ModelRegistry | None:
        if not self._registry_path.exists():
            return None
        record = json.loads(self._registry_path.read_text(encoding="utf-8"))
        registry = ModelRegistry(id=UUID(record["id"]))
        for m in record.get("models", []):
            model = PromotedModel(
                id=UUID(m["id"]),
                artifact_id=UUID(m["artifact_id"]),
                artifact_name=m["artifact_name"],
                artifact_path=m["artifact_path"],
                base_model=m["base_model"],
                promoted_at=datetime.fromisoformat(m["promoted_at"]),
            )
            registry._models[model.id] = model  # noqa: SLF001 — state restoration
        registry._events.clear()  # noqa: SLF001
        return registry

    async def get_or_create(self) -> ModelRegistry:
        registry = self._read()
        if registry is None:
            registry = ModelRegistry()
            self._write(registry)
        return registry

    async def save(self, registry: ModelRegistry) -> None:
        self._write(registry)

    async def get(self, id: UUID) -> ModelRegistry | None:
        registry = self._read()
        if registry is not None and registry.id == id:
            return registry
        return None
