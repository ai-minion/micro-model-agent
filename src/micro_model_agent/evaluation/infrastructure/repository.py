"""JsonlEvaluationReportRepository — evaluation context DDD repository.

Each EvaluationReport is stored as a JSON file under ``<root>/reports/<id>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from micro_model_agent.evaluation.domain.aggregate import (
    EvaluationReport,
    EvaluationResult,
    EvaluationThreshold,
)


class JsonlEvaluationReportRepository:
    """DDD-style EvaluationReportRepository backed by JSON files."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _report_path(self, report_id: UUID) -> Path:
        return self._root / "reports" / f"{report_id}.json"

    def _write(self, report: EvaluationReport) -> None:
        path = self._report_path(report.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id": str(report.id),
            "run_id": report.run_id,
            "threshold": report.threshold.minimum_score,
            "summary_score": report.summary_score,
            "finalised": report._finalised,  # noqa: SLF001
            "results": [
                {
                    "category": r.category,
                    "passed": r.passed,
                    "score": r.score,
                    "summary": r.summary,
                }
                for r in report.results
            ],
        }
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _read(self, path: Path) -> EvaluationReport:
        record = json.loads(path.read_text(encoding="utf-8"))
        threshold = EvaluationThreshold(minimum_score=record["threshold"])
        report = EvaluationReport(
            run_id=record["run_id"],
            threshold=threshold,
            id=UUID(record["id"]),
        )
        for r in record.get("results", []):
            report._results.append(  # noqa: SLF001
                EvaluationResult(
                    category=r["category"],
                    passed=r["passed"],
                    score=r["score"],
                    summary=r["summary"],
                )
            )
        if record.get("finalised") and record.get("summary_score") is not None:
            report.summary_score = record["summary_score"]
            report._finalised = True  # noqa: SLF001
        return report

    async def add(self, report: EvaluationReport) -> None:
        self._write(report)

    async def save(self, report: EvaluationReport) -> None:
        self._write(report)

    async def get(self, id: UUID) -> EvaluationReport | None:
        path = self._report_path(id)
        if not path.exists():
            return None
        return self._read(path)

    async def find_by_run_id(self, run_id: str) -> list[EvaluationReport]:
        reports_dir = self._root / "reports"
        if not reports_dir.exists():
            return []
        result = []
        for path in reports_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("run_id") == run_id:
                result.append(self._read(path))
        return result
