"""Filesystem storage for persisted evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

from micro_model_agent.shared.domain.value_objects import EvaluationResult


class LocalEvaluationResultReader:
    """Filesystem adapter for loading persisted evaluation reports."""

    def load_evaluation_result(self, path: Path) -> EvaluationResult:
        """Load one evaluation report."""

        return load_evaluation_result(path)


class LocalEvaluationResultWriter:
    """Filesystem adapter for writing persisted evaluation reports."""

    def write_evaluation_result(
        self,
        run_dir: Path,
        result: EvaluationResult,
        output_path: Path | None = None,
    ) -> Path:
        """Write one evaluation report and return its path."""

        return write_evaluation_result(run_dir, result, output_path)


def write_evaluation_result(
    run_dir: Path,
    result: EvaluationResult,
    output: Path | None = None,
) -> Path:
    """Write an evaluation report next to the training run metadata."""

    path = output or run_dir / "evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "passed": result.passed,
                "summary": result.summary,
                "score": result.score,
                "details": result.details,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def load_evaluation_result(path: Path) -> EvaluationResult:
    """Load an evaluation report written by the CLI."""

    if not path.exists():
        raise FileNotFoundError(f"evaluation report not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationResult(
        passed=bool(record["passed"]),
        summary=str(record["summary"]),
        score=float(record["score"]) if record.get("score") is not None else None,
        details=dict(record.get("details", {})),
    )
