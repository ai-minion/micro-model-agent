"""Ollama packaging helpers for promoted PEFT/LoRA adapters."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from micro_model_agent.application.ports import OllamaPackageRecord, PromotedArtifactRecord
from micro_model_agent.infrastructure.training_artifacts import PromotionRegistryEntry


@dataclass(frozen=True, slots=True)
class OllamaPackageResult:
    """Metadata for one Ollama packaging attempt."""

    model_name: str
    base_model: str
    adapter_path: str
    modelfile_path: str
    manifest_path: str
    command: tuple[str, ...]
    created: bool
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    warnings: tuple[str, ...] = ()
    promotion_artifact_id: UUID | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-serializable package record."""

        record = asdict(self)
        record["id"] = str(self.id)
        record["promotion_artifact_id"] = (
            str(self.promotion_artifact_id) if self.promotion_artifact_id else None
        )
        record["created_at"] = self.created_at.isoformat()
        record["command"] = list(self.command)
        record["warnings"] = list(self.warnings)
        return record


class LocalOllamaAdapterPackager:
    """Filesystem adapter for packaging promoted adapters for Ollama."""

    def package_promoted_adapter_for_ollama(
        self,
        *,
        record: PromotedArtifactRecord,
        model_name: str,
        output_dir: Path,
        ollama_base_model: str | None = None,
        create: bool = False,
    ) -> OllamaPackageRecord:
        """Write Ollama package files for one promoted artifact."""

        result = package_promoted_adapter_for_ollama(
            entry=_promotion_entry_from_record(record),
            model_name=model_name,
            output_dir=output_dir,
            ollama_base_model=ollama_base_model,
            create=create,
        )
        return OllamaPackageRecord(
            model_name=result.model_name,
            base_model=result.base_model,
            adapter_path=result.adapter_path,
            modelfile_path=result.modelfile_path,
            manifest_path=result.manifest_path,
            command=result.command,
            created=result.created,
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
            warnings=result.warnings,
        )


def package_promoted_adapter_for_ollama(
    *,
    entry: PromotionRegistryEntry,
    model_name: str,
    output_dir: Path,
    ollama_base_model: str | None = None,
    create: bool = False,
) -> OllamaPackageResult:
    """Write a Modelfile for a promoted adapter and optionally run `ollama create`."""

    adapter_path = Path(entry.artifact_path)
    base_model = ollama_base_model or entry.base_model
    output_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = output_dir / "Modelfile"
    manifest_path = output_dir / "ollama-package.json"
    warnings = _packaging_warnings(
        adapter_path=adapter_path,
        base_model=base_model,
        source_base_model=entry.base_model,
        explicit_ollama_base_model=ollama_base_model,
    )

    modelfile_path.write_text(
        _modelfile_text(base_model=base_model, adapter_path=adapter_path),
        encoding="utf-8",
    )
    command = ("ollama", "create", model_name, "-f", str(modelfile_path))
    result = OllamaPackageResult(
        model_name=model_name,
        base_model=base_model,
        adapter_path=str(adapter_path),
        modelfile_path=str(modelfile_path),
        manifest_path=str(manifest_path),
        command=command,
        created=False,
        warnings=warnings,
        promotion_artifact_id=entry.artifact_id,
    )

    if create:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        result = OllamaPackageResult(
            model_name=model_name,
            base_model=base_model,
            adapter_path=str(adapter_path),
            modelfile_path=str(modelfile_path),
            manifest_path=str(manifest_path),
            command=command,
            created=completed.returncode == 0,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            warnings=warnings,
            promotion_artifact_id=entry.artifact_id,
        )

    manifest_path.write_text(
        json.dumps(result.as_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _promotion_entry_from_record(record: PromotedArtifactRecord) -> PromotionRegistryEntry:
    """Convert the application registry view into the existing infrastructure shape."""

    return PromotionRegistryEntry(
        artifact_id=record.artifact_id,
        artifact_name=record.artifact_name,
        artifact_path=record.artifact_path,
        artifact_kind=record.artifact_kind,
        base_model=record.base_model,
        run_dir=record.run_dir,
        promotion_report_path=record.promotion_report_path,
        evaluation_report_paths=record.evaluation_report_paths,
        minimum_score=record.minimum_score,
        reviewer_notes=record.reviewer_notes,
        approved_by=record.approved_by,
        id=record.id or uuid4(),
        created_at=record.created_at or datetime.now(UTC),
    )


def _modelfile_text(*, base_model: str, adapter_path: Path) -> str:
    """Build a minimal Ollama Modelfile for an adapter-backed model."""

    return "\n".join(
        [
            f"FROM {base_model}",
            f"ADAPTER {adapter_path}",
            'PARAMETER stop "<|im_end|>"',
            'PARAMETER stop "<|endoftext|>"',
            "",
        ]
    )


def _packaging_warnings(
    *,
    adapter_path: Path,
    base_model: str,
    source_base_model: str,
    explicit_ollama_base_model: str | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not adapter_path.exists():
        warnings.append(f"adapter path does not exist yet: {adapter_path}")
    elif adapter_path.is_dir() and not _contains_adapter_weights(adapter_path):
        warnings.append(f"adapter directory does not contain adapter weights: {adapter_path}")
    if explicit_ollama_base_model is None:
        warnings.append(
            "using training base model as Ollama FROM value; verify it is an Ollama model, "
            "GGUF file, or supported local Safetensors model directory"
        )
    elif base_model != source_base_model:
        warnings.append(
            "Ollama base model differs from training base model; verify adapter/base "
            "compatibility before using the packaged model"
        )
    return tuple(warnings)


def _contains_adapter_weights(path: Path) -> bool:
    """Return true when a directory contains common LoRA adapter weight files."""

    expected_names = {
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
    if any((path / name).exists() for name in expected_names):
        return True
    return any(child.suffix == ".gguf" for child in path.iterdir())
