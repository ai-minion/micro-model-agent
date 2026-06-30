"""Tests for evaluation application workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from micro_model_agent.application.evaluation import (
    RunEvaluationComparisonRequest,
    RunEvaluationComparisonWorkflow,
)
from micro_model_agent.domain.contracts import EvaluationResult


class FakeEvaluationResultReader:
    """In-memory evaluation report reader for application tests."""

    def __init__(self, reports: dict[Path, EvaluationResult]) -> None:
        self.reports = reports
        self.loaded_paths: list[Path] = []

    def load_evaluation_result(self, path: Path) -> EvaluationResult:
        self.loaded_paths.append(path)
        return self.reports[path]


class FakeEvaluationComparisonReportWriter:
    """In-memory comparison report writer for application tests."""

    def __init__(self) -> None:
        self.written: tuple[Path, dict[str, object]] | None = None

    def write_evaluation_comparison_report(
        self,
        path: Path,
        record: dict[str, object],
    ) -> None:
        self.written = (path, record)


def test_evaluation_comparison_workflow_loads_compares_and_writes_report() -> None:
    baseline_report = Path("base-evaluation.json")
    adapter_report = Path("adapter-evaluation.json")
    reader = FakeEvaluationResultReader(
        {
            baseline_report: EvaluationResult(
                passed=False,
                summary="base",
                score=0.70,
                details={"metrics": {"correct_tool_rate": 0.60}},
            ),
            adapter_report: EvaluationResult(
                passed=True,
                summary="adapter",
                score=0.84,
                details={"metrics": {"correct_tool_rate": 0.74}},
            ),
        }
    )
    writer = FakeEvaluationComparisonReportWriter()
    workflow = RunEvaluationComparisonWorkflow(
        evaluation_reader=reader,
        comparison_writer=writer,
    )

    result = workflow.run(
        RunEvaluationComparisonRequest(
            baseline_report_path=baseline_report,
            adapter_report_path=adapter_report,
            output_path=Path("comparison.json"),
            minimum_score_delta=0.10,
            minimum_metric_deltas={"correct_tool_rate": 0.10},
        )
    )

    assert reader.loaded_paths == [baseline_report, adapter_report]
    assert result.output_path == Path("comparison.json")
    assert result.comparison.passed is True
    assert result.comparison.score_delta == pytest.approx(0.14)
    assert writer.written is not None
    path, record = writer.written
    assert path == Path("comparison.json")
    assert record["passed"] is True
    assert record["score_delta"] == pytest.approx(0.14)
    metric_deltas = {
        delta["name"]: delta
        for delta in record["metric_deltas"]
        if isinstance(delta, dict)
    }
    assert metric_deltas["correct_tool_rate"]["passed"] is True


def test_evaluation_comparison_workflow_records_threshold_failures() -> None:
    baseline_report = Path("base-evaluation.json")
    adapter_report = Path("adapter-evaluation.json")
    reader = FakeEvaluationResultReader(
        {
            baseline_report: EvaluationResult(
                passed=True,
                summary="base",
                score=0.80,
                details={"metrics": {"tool_history_match_rate": 0.80}},
            ),
            adapter_report: EvaluationResult(
                passed=True,
                summary="adapter",
                score=0.82,
                details={"metrics": {"tool_history_match_rate": 0.84}},
            ),
        }
    )
    writer = FakeEvaluationComparisonReportWriter()
    workflow = RunEvaluationComparisonWorkflow(
        evaluation_reader=reader,
        comparison_writer=writer,
    )

    result = workflow.run(
        RunEvaluationComparisonRequest(
            baseline_report_path=baseline_report,
            adapter_report_path=adapter_report,
            output_path=Path("comparison.json"),
            minimum_score_delta=0.05,
            minimum_metric_deltas={"tool_history_match_rate": 0.10},
        )
    )

    assert result.comparison.passed is False
    assert result.comparison.errors == (
        "score delta 0.0200 is below minimum 0.0500",
        "metric 'tool_history_match_rate' delta 0.0400 is below minimum 0.1000",
    )
    assert writer.written is not None
    _, record = writer.written
    assert record["passed"] is False
    assert record["errors"] == list(result.comparison.errors)
