"""Tests for human trace review storage."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.dataset.domain.value_objects import DatasetLabel, OutcomeLabel, QualityLabel
from micro_model_agent.dataset.infrastructure.traces.review import JsonlTraceReviewStore, TraceReview


def test_jsonl_trace_review_store_returns_latest_review_by_trace_id(tmp_path: Path) -> None:
    store = JsonlTraceReviewStore(tmp_path / "reviews.jsonl")
    first = TraceReview(
        trace_id="trace-1",
        label=DatasetLabel(outcome=OutcomeLabel.NEEDS_REVIEW, quality=QualityLabel.UNKNOWN),
    )
    updated = TraceReview(
        trace_id="trace-1",
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        corrected_target={"summary": "corrected"},
    )

    asyncio.run(store.save(first))
    asyncio.run(store.save(updated))

    reviews = asyncio.run(store.latest_by_trace_id())

    assert reviews["trace-1"].label.outcome is OutcomeLabel.ACCEPTED
    assert reviews["trace-1"].corrected_target == {"summary": "corrected"}
