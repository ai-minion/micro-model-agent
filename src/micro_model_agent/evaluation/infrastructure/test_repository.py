"""Integration tests for JsonlEvaluationReportRepository."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from micro_model_agent.evaluation.domain.aggregate import (
    EvaluationReport,
    EvaluationResult,
    EvaluationThreshold,
)
from micro_model_agent.evaluation.infrastructure.repository import (
    JsonlEvaluationReportRepository,
)


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _report(run_id: str = "run-1", minimum: float = 0.8) -> EvaluationReport:
    return EvaluationReport(
        run_id=run_id,
        threshold=EvaluationThreshold(minimum_score=minimum),
    )


def test_add_and_get_round_trip(tmp_path: Path) -> None:
    repo = JsonlEvaluationReportRepository(tmp_path)
    report = _report()
    report.finalize(0.9)
    _run(repo.add(report))

    loaded = _run(repo.get(report.id))
    assert loaded is not None
    assert loaded.id == report.id
    assert loaded.run_id == "run-1"
    assert loaded.summary_score == 0.9
    assert loaded.passed is True


def test_get_missing_returns_none(tmp_path: Path) -> None:
    from uuid import uuid4
    repo = JsonlEvaluationReportRepository(tmp_path)
    assert _run(repo.get(uuid4())) is None


def test_save_updates_finalised_state(tmp_path: Path) -> None:
    repo = JsonlEvaluationReportRepository(tmp_path)
    report = _report()
    _run(repo.add(report))  # not yet finalised

    report.finalize(0.85)
    _run(repo.save(report))

    loaded = _run(repo.get(report.id))
    assert loaded is not None
    assert loaded.summary_score == 0.85


def test_find_by_run_id(tmp_path: Path) -> None:
    repo = JsonlEvaluationReportRepository(tmp_path)
    r1 = _report(run_id="job-A")
    r1.finalize(0.9)
    r2 = _report(run_id="job-B")
    r2.finalize(0.6)
    _run(repo.add(r1))
    _run(repo.add(r2))

    found = _run(repo.find_by_run_id("job-A"))
    assert len(found) == 1
    assert found[0].run_id == "job-A"

    not_found = _run(repo.find_by_run_id("job-C"))
    assert not_found == []


def test_report_with_results_round_trips(tmp_path: Path) -> None:
    repo = JsonlEvaluationReportRepository(tmp_path)
    report = _report()
    report.add_result(EvaluationResult(category="tools", passed=True, score=0.9, summary="ok"))
    report.add_result(EvaluationResult(category="repair", passed=True, score=0.8, summary="ok"))
    report.finalize(0.85)
    _run(repo.add(report))

    loaded = _run(repo.get(report.id))
    assert loaded is not None
    assert len(loaded.results) == 2
    assert loaded.results[0].category == "tools"
