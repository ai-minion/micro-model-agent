"""Tests for promotion infrastructure gate store and registry."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from micro_model_agent.promotion.infrastructure.gate import (
    LocalPromotionGateStore,
    PromotionRegistryEntry,
    load_promotion_registry,
    record_promoted_artifact,
    write_promotion_gate_result,
)
from micro_model_agent.shared.domain.value_objects import EvaluationResult
from micro_model_agent.training.domain.value_objects import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from micro_model_agent.training.infrastructure.training_records import (
    model_artifact_to_record,
)
import json


def _artifact(output_dir: str = "/tmp/run") -> ModelArtifact:
    return ModelArtifact(
        name="synthetic-adapter",
        kind=ModelArtifactKind.ADAPTER,
        path=str(Path(output_dir) / "adapter"),
        base_model="tinyllama",
        metrics={"loss": 0.42},
    )


def _result(passed: bool = True, score: float = 0.9) -> EvaluationResult:
    return EvaluationResult(passed=passed, summary="ok" if passed else "fail", score=score)


# ---------------------------------------------------------------------------
# write_promotion_gate_result
# ---------------------------------------------------------------------------


def test_write_gate_result_creates_file(tmp_path: Path) -> None:
    result_path = write_promotion_gate_result(
        tmp_path,
        promoted=True,
        minimum_score=0.8,
        evaluation_reports=[(tmp_path / "eval.json", _result(), True)],
    )
    assert result_path.exists()


def test_write_gate_result_default_path(tmp_path: Path) -> None:
    result_path = write_promotion_gate_result(
        tmp_path,
        promoted=True,
        minimum_score=0.8,
        evaluation_reports=[],
    )
    assert result_path == tmp_path / "promotion.json"


def test_write_gate_result_content(tmp_path: Path) -> None:
    write_promotion_gate_result(
        tmp_path,
        promoted=True,
        minimum_score=0.85,
        evaluation_reports=[(tmp_path / "eval.json", _result(score=0.9), True)],
    )
    content = json.loads((tmp_path / "promotion.json").read_text())
    assert content["promoted"] is True
    assert content["minimum_score"] == 0.85


def test_write_gate_result_not_promoted(tmp_path: Path) -> None:
    write_promotion_gate_result(
        tmp_path,
        promoted=False,
        minimum_score=0.8,
        evaluation_reports=[(tmp_path / "eval.json", _result(passed=False, score=0.5), False)],
    )
    content = json.loads((tmp_path / "promotion.json").read_text())
    assert content["promoted"] is False


# ---------------------------------------------------------------------------
# record_promoted_artifact / load_promotion_registry
# ---------------------------------------------------------------------------


def _write_run_metadata(run_dir: Path, artifact: ModelArtifact) -> None:
    """Write the training artifact metadata that record_promoted_artifact needs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    artifact_path = artifacts_dir / f"{artifact.id}.json"
    artifact_path.write_text(json.dumps(model_artifact_to_record(artifact)), encoding="utf-8")
    # Also write a run.json for load_artifact_from_training_run
    run_json = run_dir / "run.json"
    from micro_model_agent.training.domain.value_objects import TrainingConfig, TrainingRun, TrainingRunKind, TrainingRunStatus
    import dataclasses, datetime
    run = TrainingRun(
        kind=TrainingRunKind.SYNTHETIC,
        config=TrainingConfig(base_model="tinyllama", output_dir=str(run_dir)),
        status=TrainingRunStatus.SUCCEEDED,
        artifacts=(artifact,),
    )
    from micro_model_agent.training.infrastructure.training_records import training_run_to_record
    run_json.write_text(json.dumps(training_run_to_record(run)), encoding="utf-8")


def test_record_and_load_registry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact = _artifact(str(run_dir))
    _write_run_metadata(run_dir, artifact)

    registry_path = tmp_path / "registry.jsonl"
    # Write the promotion report file so record_promoted_artifact can find it
    promotion_report = run_dir / "promotion.json"
    write_promotion_gate_result(run_dir, promoted=True, minimum_score=0.8, evaluation_reports=[])

    entry = record_promoted_artifact(
        registry_path,
        artifact=artifact,
        run_dir=run_dir,
        promotion_report_path=promotion_report,
        reviewer_notes="looks good",
    )
    assert entry is not None

    records = load_promotion_registry(registry_path)
    assert len(records) == 1
    assert str(records[0].artifact_id) == str(artifact.id)


def test_load_registry_empty_file(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text("")
    records = load_promotion_registry(registry_path)
    assert records == []


def test_load_registry_missing_file(tmp_path: Path) -> None:
    records = load_promotion_registry(tmp_path / "nonexistent.jsonl")
    assert records == []


# ---------------------------------------------------------------------------
# LocalPromotionGateStore adapter
# ---------------------------------------------------------------------------


def test_store_load_evaluation_result(tmp_path: Path) -> None:
    from micro_model_agent.evaluation.infrastructure.reports import write_evaluation_result
    result = _result(score=0.88)
    path = write_evaluation_result(tmp_path, result)

    store = LocalPromotionGateStore()
    loaded = store.load_evaluation_result(path)
    assert loaded.score == 0.88


def test_store_write_promotion_gate_result(tmp_path: Path) -> None:
    store = LocalPromotionGateStore()
    path = store.write_promotion_gate_result(
        tmp_path,
        promoted=True,
        minimum_score=0.8,
        evaluation_reports=[],
    )
    assert path.exists()
