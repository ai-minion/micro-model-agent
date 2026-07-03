"""Tests for persisted evaluation report storage."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.evaluation.infrastructure.reports import (
    LocalEvaluationResultReader,
    LocalEvaluationResultWriter,
)
from micro_model_agent.shared.domain.value_objects import EvaluationResult


def test_local_evaluation_result_reader_writer_round_trips_report(tmp_path: Path) -> None:
    writer = LocalEvaluationResultWriter()
    reader = LocalEvaluationResultReader()
    run_dir = tmp_path / "runs" / "latest"
    output = tmp_path / "reports" / "synthetic-evaluation.json"

    path = writer.write_evaluation_result(
        run_dir,
        EvaluationResult(
            passed=True,
            summary="ok",
            score=0.9,
            details={"metrics": {"score": 0.9}},
        ),
        output,
    )
    loaded = reader.load_evaluation_result(path)

    assert path == output
    assert loaded.passed is True
    assert loaded.summary == "ok"
    assert loaded.score == 0.9
    assert loaded.details == {"metrics": {"score": 0.9}}
