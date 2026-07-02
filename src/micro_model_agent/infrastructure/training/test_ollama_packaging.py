"""Tests for Ollama packaging metadata."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from micro_model_agent.infrastructure.promotion.gate import PromotionRegistryEntry
from micro_model_agent.infrastructure.training.ollama_packaging import (
    package_promoted_adapter_for_ollama,
)


def _entry(adapter_path: Path) -> PromotionRegistryEntry:
    return PromotionRegistryEntry(
        artifact_id=UUID("00000000-0000-4000-8000-000000000001"),
        artifact_name="proof-adapter",
        artifact_path=str(adapter_path),
        artifact_kind="adapter",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        run_dir="training/runs/proof",
        promotion_report_path="training/runs/proof/promotion.json",
        evaluation_report_paths=("synthetic-evaluation.json",),
        minimum_score=0.9,
        approved_by="tests",
    )


def test_package_promoted_adapter_writes_modelfile_and_manifest(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    result = package_promoted_adapter_for_ollama(
        entry=_entry(adapter_dir),
        model_name="micro-agent-proof:qwen",
        output_dir=tmp_path / "ollama",
        ollama_base_model="qwen2.5-coder:7b",
    )

    modelfile = (tmp_path / "ollama" / "Modelfile").read_text(encoding="utf-8")
    manifest = (tmp_path / "ollama" / "ollama-package.json").read_text(encoding="utf-8")
    assert result.created is False
    assert result.command == (
        "ollama",
        "create",
        "micro-agent-proof:qwen",
        "-f",
        str(tmp_path / "ollama" / "Modelfile"),
    )
    assert "FROM qwen2.5-coder:7b" in modelfile
    assert f"ADAPTER {adapter_dir}" in modelfile
    assert '"model_name": "micro-agent-proof:qwen"' in manifest
    assert "differs from training base model" in result.warnings[0]


def test_package_promoted_adapter_warns_for_missing_adapter_path(tmp_path: Path) -> None:
    result = package_promoted_adapter_for_ollama(
        entry=_entry(tmp_path / "missing-adapter"),
        model_name="micro-agent-proof:qwen",
        output_dir=tmp_path / "ollama",
    )

    assert result.created is False
    assert any("adapter path does not exist" in warning for warning in result.warnings)
    assert any(
        "using training base model as Ollama FROM value" in warning
        for warning in result.warnings
    )
