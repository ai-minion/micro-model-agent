"""Filesystem-backed promotion gate storage and policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from micro_model_agent.application.ports.contracts import PromotedArtifactRecord
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.training import ModelArtifact
from micro_model_agent.infrastructure.evaluation.reports import load_evaluation_result
from micro_model_agent.infrastructure.persistence.training_records import (
    _to_jsonable,
    load_artifact_from_training_run,
)


@dataclass(frozen=True, slots=True)
class PromotionRegistryEntry:
    """One manually recorded approved model artifact."""

    artifact_id: UUID
    artifact_name: str
    artifact_path: str
    artifact_kind: str
    base_model: str
    run_dir: str
    promotion_report_path: str
    evaluation_report_paths: tuple[str, ...]
    minimum_score: float
    reviewer_notes: str | None = None
    approved_by: str | None = None
    id: UUID = dataclass_field(default_factory=uuid4)
    created_at: datetime = dataclass_field(default_factory=lambda: datetime.now(UTC))


class LocalPromotionGateStore:
    """Filesystem adapter for promotion gate artifact, evaluation, and result files."""

    def load_artifact_from_training_run(self, run_dir: Path) -> ModelArtifact:
        """Load artifact metadata for one training run."""

        return load_artifact_from_training_run(run_dir)

    def load_evaluation_result(self, path: Path) -> EvaluationResult:
        """Load one evaluation report."""

        return load_evaluation_result(path)

    def write_promotion_gate_result(
        self,
        run_dir: Path,
        *,
        promoted: bool,
        minimum_score: float,
        evaluation_reports: list[tuple[Path, EvaluationResult, bool]],
    ) -> Path:
        """Write one promotion gate decision."""

        return write_promotion_gate_result(
            run_dir,
            promoted=promoted,
            minimum_score=minimum_score,
            evaluation_reports=evaluation_reports,
        )

    def record_promoted_artifact(
        self,
        registry_path: Path,
        *,
        artifact: ModelArtifact,
        run_dir: Path,
        promotion_report_path: Path,
        reviewer_notes: str | None = None,
        approved_by: str | None = None,
    ) -> PromotedArtifactRecord:
        """Append one promoted artifact record to a local registry."""

        entry = record_promoted_artifact(
            registry_path,
            artifact=artifact,
            run_dir=run_dir,
            promotion_report_path=promotion_report_path,
            reviewer_notes=reviewer_notes,
            approved_by=approved_by,
        )
        return _promoted_artifact_record_from_entry(entry)

    def load_promotion_registry(self, registry_path: Path) -> list[PromotedArtifactRecord]:
        """Load promoted artifact records from a local registry."""

        return [
            _promoted_artifact_record_from_entry(entry)
            for entry in load_promotion_registry(registry_path)
        ]


def write_promotion_gate_result(
    run_dir: Path,
    *,
    promoted: bool,
    minimum_score: float,
    evaluation_reports: list[tuple[Path, EvaluationResult, bool]],
) -> Path:
    """Write the promotion gate decision next to the training run metadata."""

    path = run_dir / "promotion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "promoted": promoted,
                "minimum_score": minimum_score,
                "evaluations": [
                    {
                        "path": str(report_path),
                        "passed": evaluation.passed,
                        "score": evaluation.score,
                        "summary": evaluation.summary,
                        "can_promote": can_promote,
                    }
                    for report_path, evaluation, can_promote in evaluation_reports
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def promotion_registry_entry_to_record(entry: PromotionRegistryEntry) -> dict[str, Any]:
    """Convert a promotion registry entry into a JSON-ready dictionary."""

    return cast(dict[str, Any], _to_jsonable(asdict(entry)))


def promotion_registry_entry_from_record(record: dict[str, Any]) -> PromotionRegistryEntry:
    """Rebuild a promotion registry entry from a JSON dictionary."""

    created_at = record.get("created_at")
    return PromotionRegistryEntry(
        id=UUID(record["id"]) if record.get("id") else uuid4(),
        artifact_id=UUID(record["artifact_id"]),
        artifact_name=record["artifact_name"],
        artifact_path=record["artifact_path"],
        artifact_kind=record["artifact_kind"],
        base_model=record["base_model"],
        run_dir=record["run_dir"],
        promotion_report_path=record["promotion_report_path"],
        evaluation_report_paths=tuple(record.get("evaluation_report_paths", [])),
        minimum_score=float(record["minimum_score"]),
        reviewer_notes=record.get("reviewer_notes"),
        approved_by=record.get("approved_by"),
        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
    )


def load_promotion_registry(path: Path) -> list[PromotionRegistryEntry]:
    """Load local promotion registry entries from JSONL."""

    if not path.exists():
        return []
    entries: list[PromotionRegistryEntry] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid promotion registry record") from exc
        entries.append(promotion_registry_entry_from_record(record))
    return entries


def record_promoted_artifact(
    registry_path: Path,
    *,
    artifact: ModelArtifact,
    run_dir: Path,
    promotion_report_path: Path,
    reviewer_notes: str | None = None,
    approved_by: str | None = None,
) -> PromotionRegistryEntry:
    """Append a passing promotion decision to the local registry."""

    if not promotion_report_path.exists():
        raise FileNotFoundError(f"promotion report not found: {promotion_report_path}")
    promotion_report = json.loads(promotion_report_path.read_text(encoding="utf-8"))
    if promotion_report.get("promoted") is not True:
        raise ValueError("promotion report has not passed")

    evaluations = promotion_report.get("evaluations", [])
    evaluation_report_paths = tuple(
        str(evaluation.get("path"))
        for evaluation in evaluations
        if isinstance(evaluation, dict) and evaluation.get("path")
    )
    entry = PromotionRegistryEntry(
        artifact_id=artifact.id,
        artifact_name=artifact.name,
        artifact_path=artifact.path,
        artifact_kind=artifact.kind.value,
        base_model=artifact.base_model,
        run_dir=str(run_dir),
        promotion_report_path=str(promotion_report_path),
        evaluation_report_paths=evaluation_report_paths,
        minimum_score=float(promotion_report.get("minimum_score", 0.0)),
        reviewer_notes=reviewer_notes,
        approved_by=approved_by,
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(promotion_registry_entry_to_record(entry), sort_keys=True))
        file.write("\n")
    return entry


class MinimumScorePromotionPolicy:
    """Promote artifacts only when evaluation passes and meets a score threshold."""

    def __init__(self, minimum_score: float = 0.8) -> None:
        self.minimum_score = minimum_score

    async def can_promote(self, artifact: ModelArtifact, evaluation: EvaluationResult) -> bool:
        _ = artifact
        return (
            evaluation.passed
            and evaluation.score is not None
            and evaluation.score >= self.minimum_score
        )


def _promoted_artifact_record_from_entry(
    entry: PromotionRegistryEntry,
) -> PromotedArtifactRecord:
    """Convert an infrastructure registry entry into the application view."""

    return PromotedArtifactRecord(
        artifact_id=entry.artifact_id,
        artifact_name=entry.artifact_name,
        artifact_path=entry.artifact_path,
        artifact_kind=entry.artifact_kind,
        base_model=entry.base_model,
        run_dir=entry.run_dir,
        promotion_report_path=entry.promotion_report_path,
        evaluation_report_paths=entry.evaluation_report_paths,
        minimum_score=entry.minimum_score,
        reviewer_notes=entry.reviewer_notes,
        approved_by=entry.approved_by,
        id=entry.id,
        created_at=entry.created_at,
    )
