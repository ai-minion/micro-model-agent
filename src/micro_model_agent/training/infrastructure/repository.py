"""JsonlTrainingJobRepository — training context DDD repository.

Stores each TrainingJob as a JSON file containing config, run state, and
artifact references.  Uses the same directory layout as existing training
infrastructure so existing artifact stores stay compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from micro_model_agent.training.domain.aggregate import TrainingJob
from micro_model_agent.training.domain.value_objects import (
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from micro_model_agent.training.infrastructure.training_records import (
    _to_jsonable,
    model_artifact_from_record,
    model_artifact_to_record,
    training_run_to_record,
)


def _training_config_to_record(config: TrainingConfig) -> dict[str, Any]:
    from dataclasses import asdict
    return _to_jsonable(asdict(config))  # type: ignore[no-any-return]


def _training_config_from_record(record: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        base_model=record["base_model"],
        output_dir=record["output_dir"],
        max_steps=record.get("max_steps"),
        learning_rate=record.get("learning_rate"),
        batch_size=record.get("batch_size"),
        gradient_accumulation_steps=record.get("gradient_accumulation_steps"),
        seed=record.get("seed", 42),
        dry_run=record.get("dry_run", True),
        parameters=dict(record.get("parameters", {})),
    )


def _training_run_from_record(record: dict[str, Any]) -> TrainingRun:
    from datetime import datetime
    from uuid import uuid4
    started_at = record.get("started_at")
    finished_at = record.get("finished_at")
    return TrainingRun(
        id=UUID(record["id"]) if record.get("id") else uuid4(),
        kind=TrainingRunKind(record.get("kind", "synthetic")),
        config=_training_config_from_record(record["config"]),
        status=TrainingRunStatus(record.get("status", "pending")),
        dataset_version=record.get("dataset_version"),
        artifacts=tuple(
            model_artifact_from_record(a) for a in record.get("artifacts", [])
        ),
        metrics=dict(record.get("metrics", {})),
        error=record.get("error"),
        started_at=datetime.fromisoformat(started_at) if started_at else None,
        finished_at=datetime.fromisoformat(finished_at) if finished_at else None,
    )


class JsonlTrainingJobRepository:
    """DDD-style TrainingJobRepository backed by JSON files.

    Each job is stored as ``<root>/jobs/<job_id>.json``.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _job_path(self, job_id: UUID) -> Path:
        return self._root / "jobs" / f"{job_id}.json"

    def _write(self, job: TrainingJob) -> None:
        path = self._job_path(job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id": str(job.id),
            "config": _training_config_to_record(job.config),
            "run": training_run_to_record(job.run) if job.run else None,
            "artifacts": [model_artifact_to_record(a) for a in job.artifacts],
        }
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _read(self, path: Path) -> TrainingJob:
        record = json.loads(path.read_text(encoding="utf-8"))
        config = _training_config_from_record(record["config"])
        job = TrainingJob(config=config, id=UUID(record["id"]))
        job._events.clear()  # noqa: SLF001 — suppress creation event during reconstruction
        if record.get("run"):
            job._run = _training_run_from_record(record["run"])  # noqa: SLF001
        for artifact_record in record.get("artifacts", []):
            job._artifacts.append(model_artifact_from_record(artifact_record))  # noqa: SLF001
        return job

    async def add(self, job: TrainingJob) -> None:
        self._write(job)

    async def save(self, job: TrainingJob) -> None:
        self._write(job)

    async def get(self, id: UUID) -> TrainingJob | None:
        path = self._job_path(id)
        if not path.exists():
            return None
        return self._read(path)

    async def find_by_status(self, status: TrainingRunStatus) -> list[TrainingJob]:
        jobs_dir = self._root / "jobs"
        if not jobs_dir.exists():
            return []
        result = []
        for path in jobs_dir.glob("*.json"):
            job = self._read(path)
            if job.status == status:
                result.append(job)
        return result
