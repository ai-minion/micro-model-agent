"""Local supervised fine-tuning runners and backend helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from micro_model_agent.domain.training import (
    ModelArtifact,
    ModelArtifactKind,
    TrainingConfig,
    TrainingRun,
    TrainingRunKind,
    TrainingRunStatus,
)
from micro_model_agent.infrastructure.persistence.training_records import (
    model_artifact_to_record,
    training_dataset_version,
    training_run_to_record,
)


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
                dataset_version=training_dataset_version(config),
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
                "source_dataset_path": config.parameters.get("source_dataset_path"),
                "source_dataset_sha256": config.parameters.get("source_dataset_sha256"),
                "training_dataset_sha256": config.parameters.get("training_dataset_sha256"),
                "dataset_tool_profile": config.parameters.get("dataset_tool_profile"),
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
                        "source_dataset_sha256": config.parameters.get("source_dataset_sha256"),
                        "training_dataset_sha256": config.parameters.get(
                            "training_dataset_sha256"
                        ),
                        "dataset_tool_profile": config.parameters.get("dataset_tool_profile"),
                    },
                )
                run = TrainingRun(
                    kind=TrainingRunKind.SYNTHETIC,
                    config=config,
                    status=TrainingRunStatus.SUCCEEDED,
                    dataset_version=training_dataset_version(config),
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
                    dataset_version=training_dataset_version(config),
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
                    dataset_version=training_dataset_version(config),
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
                dataset_version=training_dataset_version(config),
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
