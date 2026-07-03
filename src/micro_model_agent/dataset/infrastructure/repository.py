"""JsonlDatasetRepository — dataset context DDD repository.

Implements ``DatasetRepository`` by wrapping the existing JSONL dataset store
functions.  A Dataset aggregate is serialised as a JSONL file of its examples;
the aggregate identity and name are stored in a companion metadata file.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from micro_model_agent.dataset.domain.aggregate import Dataset
from micro_model_agent.dataset.infrastructure.dataset_store import (
    dataset_example_to_record,
    load_dataset_examples,
)


class JsonlDatasetRepository:
    """DDD-style DatasetRepository backed by JSONL files.

    Layout on disk::

        <root>/
          <name>/
            dataset.jsonl     # examples, one per line
            meta.json         # {"id": "<uuid>", "name": "<name>"}
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dir(self, name: str) -> Path:
        return self._root / name

    def _meta_path(self, name: str) -> Path:
        return self._dir(name) / "meta.json"

    def _data_path(self, name: str) -> Path:
        return self._dir(name) / "dataset.jsonl"

    def _write(self, dataset: Dataset) -> None:
        d = self._dir(dataset.name)
        d.mkdir(parents=True, exist_ok=True)
        self._meta_path(dataset.name).write_text(
            json.dumps({"id": str(dataset.id), "name": dataset.name}, indent=2),
            encoding="utf-8",
        )
        with self._data_path(dataset.name).open("w", encoding="utf-8") as fh:
            for example in dataset.examples:
                fh.write(json.dumps(dataset_example_to_record(example)) + "\n")

    def _read(self, name: str) -> Dataset | None:
        meta_path = self._meta_path(name)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ds = Dataset(name=meta["name"], id=UUID(meta["id"]))
        for example in load_dataset_examples(self._data_path(name)):
            ds._examples[example.id] = example  # noqa: SLF001 — direct state restoration
        return ds

    # ------------------------------------------------------------------
    # Repository protocol
    # ------------------------------------------------------------------

    async def add(self, dataset: Dataset) -> None:
        """Persist a newly created Dataset."""
        self._write(dataset)

    async def save(self, dataset: Dataset) -> None:
        """Persist an updated Dataset snapshot."""
        self._write(dataset)

    async def get(self, id: UUID) -> Dataset | None:
        """Load a Dataset by identity (scans all subdirectories)."""
        if not self._root.exists():
            return None
        for subdir in self._root.iterdir():
            if not subdir.is_dir():
                continue
            meta_path = subdir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("id") == str(id):
                return self._read(meta["name"])
        return None

    async def find_by_name(self, name: str) -> Dataset | None:
        """Return the Dataset with the given name, or None."""
        return self._read(name)
