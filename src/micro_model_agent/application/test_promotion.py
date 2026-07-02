"""Tests for promotion application workflows."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from micro_model_agent.application.ports import (
    ModelConfigurationUpdate,
    OllamaPackageRecord,
    PromotedArtifactRecord,
)
from micro_model_agent.application.promotion.workflows import (
    RunPromotionGateRequest,
    RunPromotionGateWorkflow,
    RunPromotionListRequest,
    RunPromotionListWorkflow,
    RunPromotionPackageOllamaRequest,
    RunPromotionPackageOllamaWorkflow,
    RunPromotionRecordRequest,
    RunPromotionRecordWorkflow,
    RunPromotionSelectRequest,
    RunPromotionSelectWorkflow,
)
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.training import ModelArtifact, ModelArtifactKind


class FakePromotionPolicy:
    """Promotion policy used by tests."""

    def __init__(self, minimum_score: float) -> None:
        self.minimum_score = minimum_score

    async def can_promote(self, artifact: ModelArtifact, evaluation: EvaluationResult) -> bool:
        _ = artifact
        return (
            evaluation.passed
            and evaluation.score is not None
            and evaluation.score >= self.minimum_score
        )


class FakePromotionStore:
    """In-memory promotion persistence boundary."""

    def __init__(self, evaluations: dict[Path, EvaluationResult]) -> None:
        self.artifact = ModelArtifact(
            id=UUID("00000000-0000-4000-8000-000000000001"),
            name="trained-adapter",
            kind=ModelArtifactKind.ADAPTER,
            path="runs/latest/adapter",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            metrics={},
            metadata={},
        )
        self.evaluations = evaluations
        self.loaded_run_dir: Path | None = None
        self.written: tuple[
            Path,
            bool,
            float,
            list[tuple[Path, EvaluationResult, bool]],
        ] | None = None
        self.recorded: tuple[
            Path,
            ModelArtifact,
            Path,
            Path,
            str | None,
            str | None,
        ] | None = None
        self.registry_entries: list[PromotedArtifactRecord] = []
        self.updated_configuration: tuple[
            Path,
            str,
            str,
            dict[str, object],
        ] | None = None
        self.update_error: str | None = None
        self.packaged: tuple[
            PromotedArtifactRecord,
            str,
            Path,
            str | None,
            bool,
        ] | None = None

    def load_artifact_from_training_run(self, run_dir: Path) -> ModelArtifact:
        self.loaded_run_dir = run_dir
        return self.artifact

    def load_evaluation_result(self, path: Path) -> EvaluationResult:
        return self.evaluations[path]

    def write_promotion_gate_result(
        self,
        run_dir: Path,
        *,
        promoted: bool,
        minimum_score: float,
        evaluation_reports: list[tuple[Path, EvaluationResult, bool]],
    ) -> Path:
        self.written = (run_dir, promoted, minimum_score, evaluation_reports)
        return run_dir / "promotion.json"

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
        self.recorded = (
            registry_path,
            artifact,
            run_dir,
            promotion_report_path,
            reviewer_notes,
            approved_by,
        )
        return PromotedArtifactRecord(
            artifact_id=artifact.id,
            artifact_name=artifact.name,
            artifact_path=artifact.path,
            artifact_kind=artifact.kind.value,
            base_model=artifact.base_model,
            run_dir=str(run_dir),
            promotion_report_path=str(promotion_report_path),
            evaluation_report_paths=("synthetic-evaluation.json",),
            minimum_score=0.9,
            reviewer_notes=reviewer_notes,
            approved_by=approved_by,
        )

    def load_promotion_registry(self, registry_path: Path) -> list[PromotedArtifactRecord]:
        _ = registry_path
        return self.registry_entries

    def update_model_configuration(
        self,
        repository_root: Path,
        *,
        base_model: str,
        adapter_path: str,
        selected_promotion: dict[str, object],
    ) -> ModelConfigurationUpdate:
        self.updated_configuration = (
            repository_root,
            base_model,
            adapter_path,
            selected_promotion,
        )
        return ModelConfigurationUpdate(
            ok=self.update_error is None,
            config_path=str(repository_root / ".micro_model_agent" / "config.json"),
            error=self.update_error,
        )

    def package_promoted_adapter_for_ollama(
        self,
        *,
        record: PromotedArtifactRecord,
        model_name: str,
        output_dir: Path,
        ollama_base_model: str | None = None,
        create: bool = False,
    ) -> OllamaPackageRecord:
        self.packaged = (
            record,
            model_name,
            output_dir,
            ollama_base_model,
            create,
        )
        return OllamaPackageRecord(
            model_name=model_name,
            base_model=ollama_base_model or record.base_model,
            adapter_path=record.artifact_path,
            modelfile_path=str(output_dir / "Modelfile"),
            manifest_path=str(output_dir / "ollama-package.json"),
            command=("ollama", "create", model_name, "-f", str(output_dir / "Modelfile")),
            created=create,
            warnings=("verify base model",),
        )


def test_promotion_gate_promotes_when_all_reports_pass() -> None:
    reports = {
        Path("synthetic-evaluation.json"): EvaluationResult(
            passed=True,
            summary="synthetic ok",
            score=0.95,
        ),
        Path("trace-evaluation.json"): EvaluationResult(
            passed=True,
            summary="trace ok",
            score=0.91,
        ),
    }
    store = FakePromotionStore(reports)
    workflow = RunPromotionGateWorkflow(
        artifact_reader=store,
        evaluation_reader=store,
        result_writer=store,
        policy_factory=FakePromotionPolicy,
    )

    result = _run(
        workflow.run(
            RunPromotionGateRequest(
                run_id="latest",
                evaluation_report_paths=tuple(reports),
                minimum_score=0.9,
            )
        )
    )

    assert result.promoted is True
    assert result.output_path == Path(".micro_model_agent/training/runs/latest/promotion.json")
    assert [evaluation.can_promote for evaluation in result.evaluations] == [True, True]
    assert store.loaded_run_dir == Path(".micro_model_agent/training/runs/latest")
    assert store.written is not None
    _, promoted, minimum_score, written_reports = store.written
    assert promoted is True
    assert minimum_score == 0.9
    assert [can_promote for _, _, can_promote in written_reports] == [True, True]


def test_promotion_gate_blocks_when_any_report_misses_policy() -> None:
    reports = {
        Path("synthetic-evaluation.json"): EvaluationResult(
            passed=True,
            summary="synthetic ok",
            score=0.95,
        ),
        Path("trace-evaluation.json"): EvaluationResult(
            passed=True,
            summary="trace weak",
            score=0.7,
        ),
    }
    store = FakePromotionStore(reports)
    workflow = RunPromotionGateWorkflow(
        artifact_reader=store,
        evaluation_reader=store,
        result_writer=store,
        policy_factory=FakePromotionPolicy,
    )

    result = _run(
        workflow.run(
            RunPromotionGateRequest(
                run_id="latest",
                evaluation_report_paths=tuple(reports),
                minimum_score=0.8,
            )
        )
    )

    assert result.promoted is False
    assert [evaluation.can_promote for evaluation in result.evaluations] == [True, False]
    assert store.written is not None
    _, promoted, _, written_reports = store.written
    assert promoted is False
    assert [report_path for report_path, _, _ in written_reports] == list(reports)


def test_promotion_gate_defaults_to_run_evaluation_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "training" / "runs" / "latest"
    run_dir.mkdir(parents=True)
    report_path = run_dir / "evaluation.json"
    store = FakePromotionStore(
        {
            report_path: EvaluationResult(
                passed=True,
                summary="default ok",
                score=1.0,
            )
        }
    )
    workflow = RunPromotionGateWorkflow(
        artifact_reader=store,
        evaluation_reader=store,
        result_writer=store,
        policy_factory=FakePromotionPolicy,
    )

    result = _run(
        workflow.run(
            RunPromotionGateRequest(
                run_id=str(run_dir),
                minimum_score=0.8,
            )
        )
    )

    assert result.run_dir == run_dir
    assert [evaluation.report_path for evaluation in result.evaluations] == [report_path]
    assert result.promoted is True


def test_promotion_record_records_default_promotion_report() -> None:
    store = FakePromotionStore({})
    workflow = RunPromotionRecordWorkflow(
        artifact_reader=store,
        registry_writer=store,
    )

    result = _run(
        workflow.run(
            RunPromotionRecordRequest(
                run_id="latest",
                registry_path=Path("promoted_models.jsonl"),
                reviewer_notes="Reviewed.",
                approved_by="tests",
            )
        )
    )

    assert result.record.artifact_name == "trained-adapter"
    assert result.record.artifact_id == store.artifact.id
    assert result.promotion_report_path == Path(
        ".micro_model_agent/training/runs/latest/promotion.json"
    )
    assert store.recorded is not None
    registry_path, artifact, run_dir, report_path, reviewer_notes, approved_by = store.recorded
    assert registry_path == Path("promoted_models.jsonl")
    assert artifact == store.artifact
    assert run_dir == Path(".micro_model_agent/training/runs/latest")
    assert report_path == Path(".micro_model_agent/training/runs/latest/promotion.json")
    assert reviewer_notes == "Reviewed."
    assert approved_by == "tests"


def test_promotion_record_uses_explicit_run_and_report_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = tmp_path / "promotion-report.json"
    store = FakePromotionStore({})
    workflow = RunPromotionRecordWorkflow(
        artifact_reader=store,
        registry_writer=store,
    )

    result = _run(
        workflow.run(
            RunPromotionRecordRequest(
                run_id=str(run_dir),
                registry_path=tmp_path / "registry.jsonl",
                promotion_report_path=report_path,
            )
        )
    )

    assert result.run_dir == run_dir
    assert result.promotion_report_path == report_path
    assert store.recorded is not None
    _, _, recorded_run_dir, recorded_report_path, _, _ = store.recorded
    assert recorded_run_dir == run_dir
    assert recorded_report_path == report_path


def test_promotion_list_returns_registry_records(tmp_path: Path) -> None:
    artifact_id = UUID("00000000-0000-4000-8000-000000000001")
    store = FakePromotionStore({})
    store.registry_entries = [
        PromotedArtifactRecord(
            artifact_id=artifact_id,
            artifact_name="trained-adapter",
            artifact_path="runs/latest/adapter",
            artifact_kind="adapter",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            run_dir="runs/latest",
            promotion_report_path="runs/latest/promotion.json",
            evaluation_report_paths=(),
            minimum_score=0.9,
        )
    ]
    workflow = RunPromotionListWorkflow(registry_reader=store)

    result = _run(
        workflow.run(
            RunPromotionListRequest(
                registry_path=tmp_path / "promoted_models.jsonl",
            )
        )
    )

    assert result.registry_path == tmp_path / "promoted_models.jsonl"
    assert result.records == tuple(store.registry_entries)


def test_promotion_list_allows_empty_registry(tmp_path: Path) -> None:
    store = FakePromotionStore({})
    workflow = RunPromotionListWorkflow(registry_reader=store)

    result = _run(
        workflow.run(
            RunPromotionListRequest(
                registry_path=tmp_path / "promoted_models.jsonl",
            )
        )
    )

    assert result.records == ()


def test_promotion_select_updates_repository_model_configuration(tmp_path: Path) -> None:
    artifact_id = UUID("00000000-0000-4000-8000-000000000001")
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    store = FakePromotionStore({})
    store.registry_entries = [
        PromotedArtifactRecord(
            artifact_id=artifact_id,
            artifact_name="trained-adapter",
            artifact_path="runs/latest/adapter",
            artifact_kind="adapter",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            run_dir="runs/latest",
            promotion_report_path="runs/latest/promotion.json",
            evaluation_report_paths=("runs/latest/evaluation.json",),
            minimum_score=0.9,
            approved_by="tests",
            created_at=created_at,
        )
    ]
    workflow = RunPromotionSelectWorkflow(
        registry_reader=store,
        configuration_writer=store,
    )

    result = _run(
        workflow.run(
            RunPromotionSelectRequest(
                artifact_id=str(artifact_id),
                repository_root=tmp_path,
                registry_path=tmp_path / "promoted_models.jsonl",
            )
        )
    )

    assert result.record.artifact_id == artifact_id
    assert result.config_path == str(tmp_path / ".micro_model_agent" / "config.json")
    assert store.updated_configuration is not None
    repository_root, base_model, adapter_path, selected_promotion = (
        store.updated_configuration
    )
    assert repository_root == tmp_path
    assert base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert adapter_path == "runs/latest/adapter"
    assert selected_promotion == {
        "artifact_id": str(artifact_id),
        "artifact_name": "trained-adapter",
        "promotion_report_path": "runs/latest/promotion.json",
        "registry_path": str(tmp_path / "promoted_models.jsonl"),
        "minimum_score": 0.9,
        "approved_by": "tests",
        "created_at": created_at.isoformat(),
    }


def test_promotion_select_rejects_unknown_artifact_id(tmp_path: Path) -> None:
    store = FakePromotionStore({})
    workflow = RunPromotionSelectWorkflow(
        registry_reader=store,
        configuration_writer=store,
    )

    try:
        _run(
            workflow.run(
                RunPromotionSelectRequest(
                    artifact_id="missing",
                    repository_root=tmp_path,
                    registry_path=tmp_path / "promoted_models.jsonl",
                )
            )
        )
    except ValueError as exc:
        assert "promoted artifact id not found" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_promotion_select_reports_configuration_update_failure(tmp_path: Path) -> None:
    artifact_id = UUID("00000000-0000-4000-8000-000000000001")
    store = FakePromotionStore({})
    store.registry_entries = [
        PromotedArtifactRecord(
            artifact_id=artifact_id,
            artifact_name="trained-adapter",
            artifact_path="runs/latest/adapter",
            artifact_kind="adapter",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            run_dir="runs/latest",
            promotion_report_path="runs/latest/promotion.json",
            evaluation_report_paths=(),
            minimum_score=0.9,
        )
    ]
    store.update_error = "invalid config"
    workflow = RunPromotionSelectWorkflow(
        registry_reader=store,
        configuration_writer=store,
    )

    try:
        _run(
            workflow.run(
                RunPromotionSelectRequest(
                    artifact_id=str(artifact_id),
                    repository_root=tmp_path,
                    registry_path=tmp_path / "promoted_models.jsonl",
                )
            )
        )
    except ValueError as exc:
        assert str(exc) == "invalid config"
    else:
        raise AssertionError("expected ValueError")


def test_promotion_package_ollama_uses_default_output_directory() -> None:
    artifact_id = UUID("00000000-0000-4000-8000-000000000001")
    store = FakePromotionStore({})
    store.registry_entries = [
        PromotedArtifactRecord(
            artifact_id=artifact_id,
            artifact_name="trained-adapter",
            artifact_path="runs/latest/adapter",
            artifact_kind="adapter",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            run_dir="runs/latest",
            promotion_report_path="runs/latest/promotion.json",
            evaluation_report_paths=(),
            minimum_score=0.9,
        )
    ]
    workflow = RunPromotionPackageOllamaWorkflow(
        registry_reader=store,
        packager=store,
    )

    result = _run(
        workflow.run(
            RunPromotionPackageOllamaRequest(
                artifact_id=str(artifact_id),
                registry_path=Path("promoted_models.jsonl"),
                model_name="micro-agent-proof:qwen",
            )
        )
    )

    assert result.record.artifact_id == artifact_id
    assert result.package.modelfile_path == (
        ".micro_model_agent/training/ollama/micro-agent-proof-qwen/Modelfile"
    )
    assert store.packaged is not None
    _, model_name, output_dir, ollama_base_model, create = store.packaged
    assert model_name == "micro-agent-proof:qwen"
    assert output_dir == Path(".micro_model_agent/training/ollama/micro-agent-proof-qwen")
    assert ollama_base_model is None
    assert create is False


def test_promotion_package_ollama_uses_explicit_options(tmp_path: Path) -> None:
    artifact_id = UUID("00000000-0000-4000-8000-000000000001")
    store = FakePromotionStore({})
    store.registry_entries = [
        PromotedArtifactRecord(
            artifact_id=artifact_id,
            artifact_name="trained-adapter",
            artifact_path="runs/latest/adapter",
            artifact_kind="adapter",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            run_dir="runs/latest",
            promotion_report_path="runs/latest/promotion.json",
            evaluation_report_paths=(),
            minimum_score=0.9,
        )
    ]
    workflow = RunPromotionPackageOllamaWorkflow(
        registry_reader=store,
        packager=store,
    )

    result = _run(
        workflow.run(
            RunPromotionPackageOllamaRequest(
                artifact_id=str(artifact_id),
                registry_path=tmp_path / "registry.jsonl",
                model_name="micro-agent-proof:qwen",
                output_dir=tmp_path / "ollama",
                ollama_base_model="qwen2.5-coder:7b",
                create=True,
            )
        )
    )

    assert result.package.base_model == "qwen2.5-coder:7b"
    assert result.package.created is True
    assert store.packaged is not None
    _, _, output_dir, ollama_base_model, create = store.packaged
    assert output_dir == tmp_path / "ollama"
    assert ollama_base_model == "qwen2.5-coder:7b"
    assert create is True


def test_promotion_package_ollama_rejects_unknown_artifact_id(tmp_path: Path) -> None:
    store = FakePromotionStore({})
    workflow = RunPromotionPackageOllamaWorkflow(
        registry_reader=store,
        packager=store,
    )

    try:
        _run(
            workflow.run(
                RunPromotionPackageOllamaRequest(
                    artifact_id="missing",
                    registry_path=tmp_path / "promoted_models.jsonl",
                    model_name="micro-agent-proof:qwen",
                )
            )
        )
    except ValueError as exc:
        assert "promoted artifact id not found" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
