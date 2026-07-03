"""Review queue helpers for staged workspace evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from micro_model_agent.dataset.domain.value_objects import DatasetExample
from micro_model_agent.shared.domain.value_objects import EvaluationResult


def build_workspace_staged_review_records(
    *,
    examples: list[DatasetExample],
    reports: list[tuple[Path, EvaluationResult]],
    simple_failure_threshold: float = 0.4,
    auto_accept_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Build JSON-ready review records from staged eval reports."""

    report_examples = [_examples_by_id(report) for _, report in reports]
    records: list[dict[str, Any]] = []
    for example in examples:
        result_records = []
        for (report_path, report), examples_by_id in zip(
            reports,
            report_examples,
            strict=True,
        ):
            result = examples_by_id.get(str(example.id))
            result_records.append(
                {
                    "report_path": str(report_path),
                    "run_id": _metadata_value(report, "run_id"),
                    "provider": _metadata_value(report, "provider"),
                    "model": _metadata_value(report, "model"),
                    "base_model": _metadata_value(report, "base_model"),
                    "adapter_path": _metadata_value(report, "adapter_path"),
                    "score": result.get("score") if result else None,
                    "parse_success": result.get("parse_success") if result else False,
                    "stages": result.get("stages") if result else [],
                    "raw_response": result.get("raw_response") if result else "",
                    "parsed_response": result.get("parsed_response") if result else None,
                    "missing_from_report": result is None,
                }
            )

        auto_triage = _auto_triage_record(
            result_records,
            simple_failure_threshold=simple_failure_threshold,
            auto_accept_threshold=auto_accept_threshold,
        )
        records.append(
            {
                "example_id": str(example.id),
                "category": _category_for_review(example),
                "goal": example.input.get("goal", ""),
                "workspace_files": example.input.get("workspace_files", {}),
                "candidate_files": example.input.get("candidate_files", []),
                "observations": example.input.get("observations", []),
                "constraints": example.input.get("constraints", []),
                "expected_stages": example.target.get("stages", {}),
                "gold_response": example.target.get("gold_response"),
                "model_results": result_records,
                "auto_triage": auto_triage,
                "review": {
                    "decision": "unreviewed",
                    "notes": None,
                },
            }
        )
    return records


class LocalWorkspaceStagedReviewBuilder:
    """Adapter for building staged workspace review records."""

    def build_workspace_staged_review_records(
        self,
        *,
        examples: list[DatasetExample],
        reports: list[tuple[Path, EvaluationResult]],
        simple_failure_threshold: float = 0.4,
        auto_accept_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Build JSON-ready staged workspace review records."""

        return build_workspace_staged_review_records(
            examples=examples,
            reports=reports,
            simple_failure_threshold=simple_failure_threshold,
            auto_accept_threshold=auto_accept_threshold,
        )


class LocalWorkspaceStagedReviewQueueWriter:
    """Filesystem adapter for staged workspace review queues."""

    def write_workspace_staged_review_records(
        self,
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        """Write staged workspace review records as JSONL."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True))
                file.write("\n")


def _examples_by_id(report: EvaluationResult) -> dict[str, dict[str, Any]]:
    examples = report.details.get("examples")
    if not isinstance(examples, list):
        return {}
    examples_by_id: dict[str, dict[str, Any]] = {}
    for example in examples:
        if not isinstance(example, dict):
            continue
        example_id = example.get("example_id")
        if isinstance(example_id, str):
            examples_by_id[example_id] = example
    return examples_by_id


def _metadata_value(report: EvaluationResult, key: str) -> object:
    metadata = report.details.get("evaluation_metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata.get(key)


def _auto_triage_record(
    result_records: list[dict[str, Any]],
    *,
    simple_failure_threshold: float,
    auto_accept_threshold: float,
) -> dict[str, Any]:
    scores = [
        score
        for result in result_records
        if isinstance((score := result.get("score")), int | float)
    ]
    if not scores:
        return {
            "decision": "needs_human_review",
            "reason": "no scorable model results were found",
            "best_score": None,
        }

    best_score = max(float(score) for score in scores)
    if best_score >= auto_accept_threshold:
        return {
            "decision": "auto_accept_candidate",
            "reason": f"best score {best_score:.2f} reached auto-accept threshold",
            "best_score": best_score,
        }
    if best_score <= simple_failure_threshold:
        return {
            "decision": "auto_reject_simple_failure",
            "reason": f"best score {best_score:.2f} was at or below simple-failure threshold",
            "best_score": best_score,
        }
    return {
        "decision": "needs_human_review",
        "reason": f"best score {best_score:.2f} needs judgment",
        "best_score": best_score,
    }


def _category_for_review(example: DatasetExample) -> str | None:
    category = example.metadata.get("category")
    return category if isinstance(category, str) else None
