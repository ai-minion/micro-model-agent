"""Human review labels for stored workflow traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from micro_model_agent.application.ports import TraceReviewRecord
from micro_model_agent.domain.datasets import (
    DatasetLabel,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)


@dataclass(frozen=True, slots=True)
class TraceReview:
    """One human review decision for a workflow trace."""

    trace_id: str
    label: DatasetLabel
    corrected_target: dict[str, Any] | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def trace_review_to_record(review: TraceReview) -> dict[str, Any]:
    """Convert a trace review to JSON-friendly primitives."""

    return {
        "id": str(review.id),
        "trace_id": review.trace_id,
        "label": {
            "outcome": review.label.outcome.value,
            "quality": review.label.quality.value,
            "failure_modes": [mode.value for mode in review.label.failure_modes],
            "reviewer_notes": review.label.reviewer_notes,
        },
        "corrected_target": review.corrected_target,
        "created_at": review.created_at.isoformat(),
    }


def trace_review_from_record(record: dict[str, Any]) -> TraceReview:
    """Rebuild a trace review from a JSON record."""

    label_record = record["label"]
    created_at = record.get("created_at")
    corrected_target = record.get("corrected_target")
    return TraceReview(
        id=UUID(record["id"]) if record.get("id") else uuid4(),
        trace_id=str(record["trace_id"]),
        label=DatasetLabel(
            outcome=OutcomeLabel(label_record["outcome"]),
            quality=QualityLabel(label_record["quality"]),
            failure_modes=tuple(
                FailureMode(mode) for mode in label_record.get("failure_modes", [])
            ),
            reviewer_notes=label_record.get("reviewer_notes"),
        ),
        corrected_target=(
            dict(corrected_target) if isinstance(corrected_target, dict) else None
        ),
        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
    )


class JsonlTraceReviewStore:
    """Append-only JSONL store for human trace review decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def save(self, review: TraceReview) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(trace_review_to_record(review), sort_keys=True))
            file.write("\n")

    async def list(self) -> list[TraceReview]:
        if not self.path.exists():
            return []

        reviews: list[TraceReview] = []
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
            reviews.append(trace_review_from_record(raw))
        return reviews

    async def latest_by_trace_id(self) -> dict[str, TraceReview]:
        """Return the newest review for each trace id."""

        reviews_by_trace_id: dict[str, TraceReview] = {}
        for review in await self.list():
            reviews_by_trace_id[review.trace_id] = review
        return reviews_by_trace_id


class LocalTraceReviewReader:
    """Filesystem adapter for loading latest trace reviews."""

    def __init__(self, path: str | Path) -> None:
        self.store = JsonlTraceReviewStore(path)

    async def latest_trace_reviews_by_trace_id(self) -> dict[str, TraceReview]:
        """Return the newest review for each trace id."""

        return await self.store.latest_by_trace_id()


class LocalTraceReviewWriter:
    """Filesystem adapter for saving trace review records."""

    async def save_trace_review(self, path: Path, review: TraceReviewRecord) -> None:
        """Save a trace review to a JSONL review store."""

        await JsonlTraceReviewStore(path).save(
            TraceReview(
                trace_id=review.trace_id,
                label=review.label,
                corrected_target=review.corrected_target,
                id=review.id,
                created_at=review.created_at,
            )
        )
