"""Application workflows for model promotion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from micro_model_agent.application.ports.contracts import (
    EvaluationResultReader,
    ModelPromotionPolicy,
    OllamaPackageRecord,
    PromotedAdapterPackager,
    PromotedArtifactRecord,
    PromotionGateResultWriter,
    PromotionRegistryReader,
    PromotionRegistryWriter,
    RepositoryModelConfigurationWriter,
    TrainingRunArtifactReader,
)
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.training import ModelArtifact


@dataclass(frozen=True, slots=True)
class PromotionGateEvaluation:
    """One evaluation report considered by the promotion gate."""

    report_path: Path
    evaluation: EvaluationResult
    can_promote: bool


@dataclass(frozen=True, slots=True)
class RunPromotionGateRequest:
    """Request for gating a trained artifact on persisted evaluation reports."""

    run_id: str
    evaluation_report_paths: tuple[Path, ...] = ()
    minimum_score: float = 0.8
    training_runs_root: Path = Path(".micro_model_agent/training/runs")


@dataclass(frozen=True, slots=True)
class RunPromotionGateResult:
    """Result returned after evaluating a promotion gate."""

    artifact: ModelArtifact
    run_dir: Path
    promoted: bool
    output_path: Path
    minimum_score: float
    evaluations: tuple[PromotionGateEvaluation, ...]


@dataclass(frozen=True, slots=True)
class RunPromotionRecordRequest:
    """Request for recording a gate-passing artifact in a promotion registry."""

    run_id: str
    registry_path: Path
    promotion_report_path: Path | None = None
    reviewer_notes: str | None = None
    approved_by: str | None = None
    training_runs_root: Path = Path(".micro_model_agent/training/runs")


@dataclass(frozen=True, slots=True)
class RunPromotionRecordResult:
    """Result returned after recording a promoted artifact."""

    artifact: ModelArtifact
    run_dir: Path
    promotion_report_path: Path
    registry_path: Path
    record: PromotedArtifactRecord


@dataclass(frozen=True, slots=True)
class RunPromotionListRequest:
    """Request for listing promoted artifacts from a registry."""

    registry_path: Path


@dataclass(frozen=True, slots=True)
class RunPromotionListResult:
    """Result returned after loading promoted artifact records."""

    registry_path: Path
    records: tuple[PromotedArtifactRecord, ...]


@dataclass(frozen=True, slots=True)
class RunPromotionSelectRequest:
    """Request for selecting a promoted artifact as the repository default."""

    artifact_id: str
    repository_root: Path
    registry_path: Path


@dataclass(frozen=True, slots=True)
class RunPromotionSelectResult:
    """Result returned after selecting a promoted artifact."""

    record: PromotedArtifactRecord
    config_path: str


@dataclass(frozen=True, slots=True)
class RunPromotionPackageOllamaRequest:
    """Request for packaging a promoted artifact for Ollama."""

    artifact_id: str
    model_name: str
    registry_path: Path
    output_dir: Path | None = None
    ollama_base_model: str | None = None
    create: bool = False


@dataclass(frozen=True, slots=True)
class RunPromotionPackageOllamaResult:
    """Result returned after packaging a promoted artifact for Ollama."""

    record: PromotedArtifactRecord
    package: OllamaPackageRecord


class RunPromotionGateWorkflow:
    """Gate a trained artifact using persisted evaluation reports."""

    def __init__(
        self,
        *,
        artifact_reader: TrainingRunArtifactReader,
        evaluation_reader: EvaluationResultReader,
        result_writer: PromotionGateResultWriter,
        policy_factory: Callable[[float], ModelPromotionPolicy],
    ) -> None:
        self.artifact_reader = artifact_reader
        self.evaluation_reader = evaluation_reader
        self.result_writer = result_writer
        self.policy_factory = policy_factory

    async def run(self, request: RunPromotionGateRequest) -> RunPromotionGateResult:
        """Run the promotion gate and persist its decision."""

        run_dir = _resolve_run_dir(request.run_id, request.training_runs_root)
        artifact = self.artifact_reader.load_artifact_from_training_run(run_dir)
        report_paths = request.evaluation_report_paths or (run_dir / "evaluation.json",)
        if not report_paths:
            raise ValueError("at least one evaluation report is required")

        policy = self.policy_factory(request.minimum_score)
        evaluations = []
        for report_path in report_paths:
            evaluation = self.evaluation_reader.load_evaluation_result(report_path)
            can_promote = await policy.can_promote(artifact, evaluation)
            evaluations.append(
                PromotionGateEvaluation(
                    report_path=report_path,
                    evaluation=evaluation,
                    can_promote=can_promote,
                )
            )

        promoted = all(evaluation.can_promote for evaluation in evaluations)
        output_path = self.result_writer.write_promotion_gate_result(
            run_dir,
            promoted=promoted,
            minimum_score=request.minimum_score,
            evaluation_reports=[
                (evaluation.report_path, evaluation.evaluation, evaluation.can_promote)
                for evaluation in evaluations
            ],
        )
        return RunPromotionGateResult(
            artifact=artifact,
            run_dir=run_dir,
            promoted=promoted,
            output_path=output_path,
            minimum_score=request.minimum_score,
            evaluations=tuple(evaluations),
        )


class RunPromotionRecordWorkflow:
    """Record a gate-passing artifact in a promotion registry."""

    def __init__(
        self,
        *,
        artifact_reader: TrainingRunArtifactReader,
        registry_writer: PromotionRegistryWriter,
    ) -> None:
        self.artifact_reader = artifact_reader
        self.registry_writer = registry_writer

    async def run(self, request: RunPromotionRecordRequest) -> RunPromotionRecordResult:
        """Record a promoted artifact and return the stored registry record."""

        run_dir = _resolve_run_dir(request.run_id, request.training_runs_root)
        artifact = self.artifact_reader.load_artifact_from_training_run(run_dir)
        promotion_report_path = request.promotion_report_path or run_dir / "promotion.json"
        record = self.registry_writer.record_promoted_artifact(
            request.registry_path,
            artifact=artifact,
            run_dir=run_dir,
            promotion_report_path=promotion_report_path,
            reviewer_notes=request.reviewer_notes,
            approved_by=request.approved_by,
        )
        return RunPromotionRecordResult(
            artifact=artifact,
            run_dir=run_dir,
            promotion_report_path=promotion_report_path,
            registry_path=request.registry_path,
            record=record,
        )


class RunPromotionListWorkflow:
    """List locally recorded promoted artifacts."""

    def __init__(self, *, registry_reader: PromotionRegistryReader) -> None:
        self.registry_reader = registry_reader

    async def run(self, request: RunPromotionListRequest) -> RunPromotionListResult:
        """Load promoted artifact records from a registry."""

        records = self.registry_reader.load_promotion_registry(request.registry_path)
        return RunPromotionListResult(
            registry_path=request.registry_path,
            records=tuple(records),
        )


class RunPromotionSelectWorkflow:
    """Select a recorded promoted artifact as repository-local model defaults."""

    def __init__(
        self,
        *,
        registry_reader: PromotionRegistryReader,
        configuration_writer: RepositoryModelConfigurationWriter,
    ) -> None:
        self.registry_reader = registry_reader
        self.configuration_writer = configuration_writer

    async def run(self, request: RunPromotionSelectRequest) -> RunPromotionSelectResult:
        """Select a promoted artifact and persist repository model defaults."""

        entries = self.registry_reader.load_promotion_registry(request.registry_path)
        record = next(
            (entry for entry in entries if str(entry.artifact_id) == request.artifact_id),
            None,
        )
        if record is None:
            raise ValueError(
                f"promoted artifact id not found in {request.registry_path}: "
                f"{request.artifact_id}"
            )

        update = self.configuration_writer.update_model_configuration(
            request.repository_root,
            base_model=record.base_model,
            adapter_path=record.artifact_path,
            selected_promotion=_selected_promotion_metadata(record, request.registry_path),
        )
        if not update.ok:
            raise ValueError(update.error or "failed to update MicroModelAgent config")

        return RunPromotionSelectResult(record=record, config_path=update.config_path)


class RunPromotionPackageOllamaWorkflow:
    """Package a recorded promoted artifact for Ollama."""

    def __init__(
        self,
        *,
        registry_reader: PromotionRegistryReader,
        packager: PromotedAdapterPackager,
    ) -> None:
        self.registry_reader = registry_reader
        self.packager = packager

    async def run(
        self,
        request: RunPromotionPackageOllamaRequest,
    ) -> RunPromotionPackageOllamaResult:
        """Package a promoted artifact and return package metadata."""

        entries = self.registry_reader.load_promotion_registry(request.registry_path)
        record = next(
            (entry for entry in entries if str(entry.artifact_id) == request.artifact_id),
            None,
        )
        if record is None:
            raise ValueError(
                f"promoted artifact id not found in {request.registry_path}: "
                f"{request.artifact_id}"
            )

        output_dir = request.output_dir or (
            Path(".micro_model_agent/training/ollama") / _path_safe_name(request.model_name)
        )
        package = self.packager.package_promoted_adapter_for_ollama(
            record=record,
            model_name=request.model_name,
            output_dir=output_dir,
            ollama_base_model=request.ollama_base_model,
            create=request.create,
        )
        return RunPromotionPackageOllamaResult(record=record, package=package)


def _resolve_run_dir(run_id: str, training_runs_root: Path) -> Path:
    run_dir = Path(run_id)
    if run_dir.exists():
        return run_dir
    return training_runs_root / run_dir


def _selected_promotion_metadata(
    record: PromotedArtifactRecord,
    registry_path: Path,
) -> dict[str, object]:
    return {
        "artifact_id": str(record.artifact_id),
        "artifact_name": record.artifact_name,
        "promotion_report_path": record.promotion_report_path,
        "registry_path": str(registry_path),
        "minimum_score": record.minimum_score,
        "approved_by": record.approved_by,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


def _path_safe_name(value: str) -> str:
    """Return a conservative directory name for a model tag."""

    safe = "".join(character if character.isalnum() else "-" for character in value.lower())
    return "-".join(part for part in safe.split("-") if part) or "ollama-model"
