"""Evaluation suites for persisted model artifacts."""

from __future__ import annotations

from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.training import ModelArtifact


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
