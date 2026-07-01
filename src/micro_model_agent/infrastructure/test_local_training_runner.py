"""Tests for local training runner scaffold and promotion policy."""

from __future__ import annotations

import asyncio
from pathlib import Path

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.training import TrainingConfig
from micro_model_agent.infrastructure.promotion.gate import MinimumScorePromotionPolicy
from micro_model_agent.infrastructure.training.local_finetuning import (
    LocalFineTuningResult,
    LocalFineTuningRunner,
    _model_load_kwargs,
)
from micro_model_agent.infrastructure.training_artifacts import (
    load_artifact_from_training_run,
)


class _SuccessfulBackend:
    def train(
        self,
        config: TrainingConfig,
        dataset_path: Path,
        output_dir: Path,
    ) -> LocalFineTuningResult:
        _ = config
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return LocalFineTuningResult(
            artifact_path=adapter_dir,
            metrics={"synthetic_example_count": 2.0, "train_loss": 0.1},
            metadata={"dataset_path": str(dataset_path), "runner": "test_backend"},
        )


class _FailingBackend:
    def train(
        self,
        config: TrainingConfig,
        dataset_path: Path,
        output_dir: Path,
    ) -> LocalFineTuningResult:
        _ = config, dataset_path, output_dir
        raise RuntimeError("backend exploded")


def test_local_finetuning_runner_dry_run_writes_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        output_dir=str(run_dir),
        dry_run=True,
        parameters={"dataset_path": str(tmp_path / "synthetic.jsonl")},
    )

    run = asyncio.run(LocalFineTuningRunner().run(config))
    artifact = load_artifact_from_training_run(run_dir)

    assert run.status.value == "succeeded"
    assert artifact.metadata["runner"] == "local_hf_peft"
    assert (run_dir / "run.json").exists()


def test_local_finetuning_runner_uses_real_backend_when_not_dry_run(tmp_path: Path) -> None:
    run = asyncio.run(
        LocalFineTuningRunner(backend=_SuccessfulBackend()).run(
            TrainingConfig(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                output_dir=str(tmp_path / "run"),
                dry_run=False,
                parameters={"dataset_path": "synthetic.jsonl"},
            )
        )
    )
    artifact = load_artifact_from_training_run(tmp_path / "run")

    assert run.status.value == "succeeded"
    assert artifact.metadata["runner"] == "test_backend"
    assert artifact.metrics["train_loss"] == 0.1


def test_local_finetuning_runner_fails_closed_when_backend_errors(tmp_path: Path) -> None:
    run = asyncio.run(
        LocalFineTuningRunner(backend=_FailingBackend()).run(
            TrainingConfig(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                output_dir=str(tmp_path / "run"),
                dry_run=False,
                parameters={"dataset_path": "synthetic.jsonl"},
            )
        )
    )

    assert run.status.value == "failed"
    assert "backend exploded" in str(run.error)
    assert (tmp_path / "run" / "run.json").exists()


def test_model_load_kwargs_use_transformers_dtype_keyword_for_cuda() -> None:
    dtype = object()

    kwargs = _model_load_kwargs(
        parameters={},
        trust_remote_code=True,
        cuda_available=True,
        dtype=dtype,
    )

    assert kwargs == {"trust_remote_code": True, "device_map": "auto", "dtype": dtype}
    assert "torch_dtype" not in kwargs


def test_minimum_score_promotion_policy() -> None:
    artifact = load_artifact_from_training_run
    policy = MinimumScorePromotionPolicy(minimum_score=0.9)
    evaluation = EvaluationResult(passed=True, summary="ok", score=0.95)

    assert artifact is not None
    assert asyncio.run(policy.can_promote(type("Artifact", (), {})(), evaluation)) is True
