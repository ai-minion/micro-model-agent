"""Shared metadata enrichment helper for evaluation workflow results."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.application.ports.contracts import DatasetToolProfileSummarizer
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import DatasetExample


def _with_evaluation_metadata(
    result: EvaluationResult,
    *,
    run_id: str,
    dataset_path: Path,
    examples: list[DatasetExample],
    provider_kind: str,
    model: str | None,
    base_model: str | None,
    adapter_path: Path | None,
    tool_profile_summarizer: DatasetToolProfileSummarizer,
    default_available_tools: tuple[str, ...],
) -> EvaluationResult:
    """Add report-level metadata without changing evaluator scoring."""

    return EvaluationResult(
        passed=result.passed,
        summary=result.summary,
        score=result.score,
        details={
            **result.details,
            "evaluation_metadata": {
                "run_id": run_id,
                "dataset_path": str(dataset_path),
                "provider": provider_kind,
                "model": model,
                "base_model": base_model,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "tool_profile": (
                    tool_profile_summarizer.summarize_dataset_tool_profiles(
                        examples,
                        default_available_tools=list(default_available_tools),
                    )
                ),
            },
        },
    )
