"""Local training artifact storage, fake training, and synthetic evaluation.

This module contains both lightweight dry-run training and the optional
Transformers/PEFT training path. The public runners return the same domain
objects so callers do not need to know which backend was used.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.training import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert Python-only objects into JSON-friendly values."""

    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def training_run_to_record(run: TrainingRun) -> dict[str, Any]:
    """Convert a TrainingRun dataclass into a JSON-ready dictionary."""

    return cast(dict[str, Any], _to_jsonable(asdict(run)))


def model_artifact_to_record(artifact: ModelArtifact) -> dict[str, Any]:
    """Convert a ModelArtifact dataclass into a JSON-ready dictionary."""

    return cast(dict[str, Any], _to_jsonable(asdict(artifact)))


def model_artifact_from_record(record: dict[str, Any]) -> ModelArtifact:
    """Rebuild a ModelArtifact from a JSON dictionary."""

    created_at = record.get("created_at")
    return ModelArtifact(
        id=UUID(record["id"]) if record.get("id") else uuid4(),
        name=record["name"],
        kind=ModelArtifactKind(record["kind"]),
        path=record["path"],
        base_model=record["base_model"],
        metrics=dict(record.get("metrics", {})),
        metadata=dict(record.get("metadata", {})),
        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
    )


class JsonTrainingArtifactStore:
    """Local JSON metadata store for training artifacts."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    async def save(self, artifact: ModelArtifact) -> None:
        path = self._artifact_path(str(artifact.id))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model_artifact_to_record(artifact), indent=2), encoding="utf-8")

    async def get(self, artifact_id: str) -> ModelArtifact | None:
        path = self._artifact_path(artifact_id)
        if not path.exists():
            return None
        return model_artifact_from_record(json.loads(path.read_text(encoding="utf-8")))

    def _artifact_path(self, artifact_id: str) -> Path:
        return self.root_dir / "artifacts" / f"{artifact_id}.json"


class FakeTrainingRunner:
    """Training runner that writes deterministic metadata without touching a GPU."""

    async def run(self, config: TrainingConfig) -> TrainingRun:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # The fake runner creates the same metadata shape as a real run, but it
        # does not load model weights or require a GPU.
        now = datetime.now(UTC)
        example_count = int(config.parameters.get("example_count", 0))
        artifact = ModelArtifact(
            name="synthetic-dry-run-adapter" if config.dry_run else "synthetic-fake-adapter",
            kind=ModelArtifactKind.ADAPTER,
            path=str(output_dir / "adapter"),
            base_model=config.base_model,
            metrics={
                "synthetic_example_count": float(example_count),
                "dry_run": 1.0 if config.dry_run else 0.0,
            },
            metadata={
                "dataset_path": config.parameters.get("dataset_path"),
                "runner": "fake",
            },
        )
        run = TrainingRun(
            kind=TrainingRunKind.SYNTHETIC,
            config=config,
            status=TrainingRunStatus.SUCCEEDED,
            artifacts=(artifact,),
            metrics=artifact.metrics,
            started_at=now,
            finished_at=datetime.now(UTC),
        )

        (output_dir / "artifact.json").write_text(
            json.dumps(model_artifact_to_record(artifact), indent=2),
            encoding="utf-8",
        )
        (output_dir / "run.json").write_text(
            json.dumps(training_run_to_record(run), indent=2),
            encoding="utf-8",
        )
        return run


@dataclass(frozen=True, slots=True)
class LocalFineTuningResult:
    """Result returned by a hardware-dependent fine-tuning backend."""

    artifact_path: Path
    metrics: dict[str, float]
    metadata: dict[str, Any]


class LocalFineTuningBackend(Protocol):
    """Backend capable of running a real local supervised fine-tuning job."""

    def train(
        self,
        config: TrainingConfig,
        dataset_path: Path,
        output_dir: Path,
    ) -> LocalFineTuningResult:
        """Train an adapter and return metadata for the produced artifact."""


class HuggingFacePeftFineTuningBackend:
    """Run supervised fine-tuning with Transformers Trainer and PEFT LoRA."""

    DEFAULT_TARGET_MODULES = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def train(
        self,
        config: TrainingConfig,
        dataset_path: Path,
        output_dir: Path,
    ) -> LocalFineTuningResult:
        try:
            # Heavy ML dependencies are imported lazily so normal CLI commands
            # and tests can run without the optional training environment.
            import torch
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "local fine-tuning requires optional training dependencies; install them with "
                "`uv sync --group training` or "
                "`pip install accelerate datasets peft torch transformers trl`"
            ) from exc

        if not dataset_path.exists():
            raise FileNotFoundError(f"training dataset not found: {dataset_path}")

        parameters = config.parameters
        # Pull options from explicit config first, then from the generic
        # parameters dictionary, then fall back to a small default.
        max_steps = _int_parameter(config.max_steps, parameters, "max_steps", 20)
        batch_size = _int_parameter(config.batch_size, parameters, "batch_size", 1)
        gradient_accumulation_steps = _int_parameter(
            config.gradient_accumulation_steps,
            parameters,
            "gradient_accumulation_steps",
            4,
        )
        learning_rate = _float_parameter(config.learning_rate, parameters, "learning_rate", 2e-4)
        max_seq_length = _int_parameter(None, parameters, "max_seq_length", 1024)
        trust_remote_code = _bool_parameter(parameters, "trust_remote_code", True)
        gradient_checkpointing = _bool_parameter(parameters, "gradient_checkpointing", True)

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Only add CUDA-specific loading options when CUDA is available.
        model_kwargs = _model_load_kwargs(
            parameters=parameters,
            trust_remote_code=trust_remote_code,
            cuda_available=torch.cuda.is_available(),
            dtype=torch.float16,
        )

        model: Any = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        target_modules = _target_modules(parameters.get("lora_target_modules"))
        # LoRA trains a small set of adapter weights instead of all base-model
        # parameters, which keeps local fine-tuning cheaper.
        lora_config = LoraConfig(
            r=_int_parameter(None, parameters, "lora_r", 16),
            lora_alpha=_int_parameter(None, parameters, "lora_alpha", 32),
            lora_dropout=_float_parameter(None, parameters, "lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

        raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        # The exported dataset is chat-shaped JSON. Convert each record into one
        # plain training string, then tokenize those strings.
        text_dataset = raw_dataset.map(
            lambda record: {"text": _training_text_from_record(record, tokenizer.eos_token or "")}
        )
        tokenized_dataset = text_dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_seq_length,
            ),
            batched=True,
            remove_columns=list(text_dataset.column_names),
        )

        checkpoints_dir = output_dir / "checkpoints"
        training_args = TrainingArguments(
            output_dir=str(checkpoints_dir),
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=_int_parameter(None, parameters, "logging_steps", 1),
            save_steps=_int_parameter(None, parameters, "save_steps", max_steps),
            seed=config.seed,
            report_to="none",
            do_train=True,
            fp16=bool(torch.cuda.is_available()),
            gradient_checkpointing=gradient_checkpointing,
        )
        trainer: Any = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        train_output: Any = trainer.train()
        adapter_dir = output_dir / "adapter"
        # Save only the trained adapter/tokenizer files into the run directory.
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        metrics = _numeric_metrics(train_output.metrics)
        metrics["synthetic_example_count"] = float(len(raw_dataset))
        metrics["train_max_steps"] = float(max_steps)
        return LocalFineTuningResult(
            artifact_path=adapter_dir,
            metrics=metrics,
            metadata={
                "dataset_path": str(dataset_path),
                "runner": "local_hf_peft",
                "backend": "transformers_trainer_peft",
                "device": device,
                "max_seq_length": max_seq_length,
                "lora": {
                    "r": lora_config.r,
                    "alpha": lora_config.lora_alpha,
                    "dropout": lora_config.lora_dropout,
                    "target_modules": target_modules,
                },
            },
        )


class SyntheticEvaluationSuite:
    """Evaluate fake or real artifacts against basic synthetic metadata checks."""

    async def evaluate_artifact(self, artifact: ModelArtifact) -> EvaluationResult:
        # For now, the synthetic evaluator checks metadata rather than running a
        # full benchmark. It verifies the artifact came from at least one example.
        example_count = artifact.metrics.get("synthetic_example_count", 0.0)
        passed = example_count > 0
        return EvaluationResult(
            passed=passed,
            summary=(
                "metadata-only synthetic artifact check passed"
                if passed
                else "metadata-only synthetic artifact check has no examples"
            ),
            score=1.0 if passed else 0.0,
            details={
                "metadata_only": True,
                "artifact_id": str(artifact.id),
                "artifact_path": artifact.path,
                "metrics": artifact.metrics,
            },
        )


def load_artifact_from_training_run(run_dir: Path) -> ModelArtifact:
    """Load the artifact metadata written inside a training run directory."""

    artifact_path = run_dir / "artifact.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact metadata not found: {artifact_path}")
    return model_artifact_from_record(json.loads(artifact_path.read_text(encoding="utf-8")))


def write_evaluation_result(run_dir: Path, result: EvaluationResult) -> Path:
    """Write an evaluation report next to the training run metadata."""

    path = run_dir / "evaluation.json"
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


class LocalFineTuningRunner:
    """Hardware-dependent fine-tuning runner for local Qwen-Coder adapter runs."""

    def __init__(self, backend: LocalFineTuningBackend | None = None) -> None:
        self.backend = backend or HuggingFacePeftFineTuningBackend()

    async def run(self, config: TrainingConfig) -> TrainingRun:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        dataset_path = config.parameters.get("dataset_path")
        if not dataset_path:
            # A real training job cannot start without an exported SFT dataset.
            run = TrainingRun(
                kind=TrainingRunKind.SYNTHETIC,
                config=config,
                status=TrainingRunStatus.FAILED,
                error="dataset_path is required for local fine-tuning",
                started_at=now,
                finished_at=datetime.now(UTC),
            )
            self._write_run(output_dir, run)
            return run

        artifact = ModelArtifact(
            name="local-finetune-dry-run-adapter" if config.dry_run else "local-finetune-adapter",
            kind=ModelArtifactKind.ADAPTER,
            path=str(output_dir / "adapter"),
            base_model=config.base_model,
            metrics={"dry_run": 1.0 if config.dry_run else 0.0},
            metadata={
                "dataset_path": str(dataset_path),
                "runner": "local_hf_peft",
                "requires": ["accelerate", "datasets", "peft", "torch", "transformers", "trl"],
            },
        )

        if not config.dry_run:
            try:
                # Delegate hardware-specific training to the backend, then wrap
                # its result in the domain artifact/run objects.
                result = self.backend.train(config, Path(str(dataset_path)), output_dir)
                artifact = ModelArtifact(
                    name="local-finetune-adapter",
                    kind=ModelArtifactKind.ADAPTER,
                    path=str(result.artifact_path),
                    base_model=config.base_model,
                    metrics={**result.metrics, "dry_run": 0.0},
                    metadata={
                        **result.metadata,
                        "source_dataset_path": config.parameters.get("source_dataset_path"),
                    },
                )
                run = TrainingRun(
                    kind=TrainingRunKind.SYNTHETIC,
                    config=config,
                    status=TrainingRunStatus.SUCCEEDED,
                    artifacts=(artifact,),
                    metrics=artifact.metrics,
                    started_at=now,
                    finished_at=datetime.now(UTC),
                )
            except KeyboardInterrupt:
                # Treat Ctrl+C as a cancelled run instead of an unhandled crash.
                run = TrainingRun(
                    kind=TrainingRunKind.SYNTHETIC,
                    config=config,
                    status=TrainingRunStatus.CANCELLED,
                    artifacts=(artifact,),
                    error="training interrupted",
                    started_at=now,
                    finished_at=datetime.now(UTC),
                )
            except BaseException as exc:
                # Catch broad exceptions so a failed training run is still
                # recorded on disk with the error message.
                run = TrainingRun(
                    kind=TrainingRunKind.SYNTHETIC,
                    config=config,
                    status=TrainingRunStatus.FAILED,
                    artifacts=(artifact,),
                    error=f"{type(exc).__name__}: {exc}",
                    started_at=now,
                    finished_at=datetime.now(UTC),
                )
        else:
            # A dry run records what would be trained without invoking the backend.
            run = TrainingRun(
                kind=TrainingRunKind.SYNTHETIC,
                config=config,
                status=TrainingRunStatus.SUCCEEDED,
                artifacts=(artifact,),
                metrics=artifact.metrics,
                started_at=now,
                finished_at=datetime.now(UTC),
            )

        self._write_artifact(output_dir, artifact)
        self._write_run(output_dir, run)
        return run

    def _write_artifact(self, output_dir: Path, artifact: ModelArtifact) -> None:
        """Persist artifact metadata inside the run directory."""

        (output_dir / "artifact.json").write_text(
            json.dumps(model_artifact_to_record(artifact), indent=2),
            encoding="utf-8",
        )

    def _write_run(self, output_dir: Path, run: TrainingRun) -> None:
        """Persist run metadata inside the run directory."""

        (output_dir / "run.json").write_text(
            json.dumps(training_run_to_record(run), indent=2),
            encoding="utf-8",
        )


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


def _int_parameter(
    explicit: int | None,
    parameters: dict[str, Any],
    name: str,
    default: int,
) -> int:
    """Read an integer option from config, parameters, or a default."""

    if explicit is not None:
        return explicit
    value = parameters.get(name, default)
    return int(value)


def _float_parameter(
    explicit: float | None,
    parameters: dict[str, Any],
    name: str,
    default: float,
) -> float:
    """Read a float option from config, parameters, or a default."""

    if explicit is not None:
        return explicit
    value = parameters.get(name, default)
    return float(value)


def _bool_parameter(parameters: dict[str, Any], name: str, default: bool) -> bool:
    """Read a boolean option while accepting common string spellings."""

    value = parameters.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _target_modules(value: Any) -> list[str]:
    """Normalize LoRA target modules from None, comma-separated text, or a list."""

    if value is None:
        return list(HuggingFacePeftFineTuningBackend.DEFAULT_TARGET_MODULES)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (tuple, list)):
        return [str(part) for part in value]
    raise TypeError("lora_target_modules must be a comma-separated string or list")


def _model_load_kwargs(
    parameters: dict[str, Any],
    trust_remote_code: bool,
    cuda_available: bool,
    dtype: Any,
) -> dict[str, Any]:
    """Build keyword arguments for loading a Transformers model."""

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if cuda_available:
        model_kwargs["device_map"] = parameters.get("device_map", "auto")
        model_kwargs["dtype"] = dtype
    return model_kwargs


def _training_text_from_record(record: dict[str, Any], eos_token: str) -> str:
    """Turn one chat-style SFT record into plain text for causal LM training."""

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT dataset records must contain a messages list")

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("SFT dataset messages must be objects")
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts) + eos_token


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep only numeric metrics and convert them to floats for JSON metadata."""

    return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}
