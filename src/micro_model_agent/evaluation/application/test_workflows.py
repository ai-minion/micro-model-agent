"""Tests for evaluation application workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from micro_model_agent.evaluation.application.workflows import (
    RunEvaluationComparisonRequest,
    RunEvaluationComparisonWorkflow,
    RunSyntheticEvaluationRequest,
    RunSyntheticEvaluationWorkflow,
    RunTraceEvaluationRequest,
    RunTraceEvaluationWorkflow,
    RunWorkspaceStagedEvaluationRequest,
    RunWorkspaceStagedEvaluationWorkflow,
    RunWorkspaceStagedReviewRequest,
    RunWorkspaceStagedReviewWorkflow,
    RunWorkspaceStagedReviewWriteRequest,
    SyntheticBehaviorEvaluationSuite,
    SyntheticExampleScorer,
    TraceBehaviorEvaluationSuite,
    TraceExampleScorer,
    WorkspaceStagedEvaluationSuite,
    WorkspaceStagedExampleScorer,
)
from micro_model_agent.shared.domain.value_objects import EvaluationResult
from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.training.domain.value_objects import ModelArtifact, ModelArtifactKind


class FakeEvaluationResultReader:
    """In-memory evaluation report reader for application tests."""

    def __init__(self, reports: dict[Path, EvaluationResult]) -> None:
        self.reports = reports
        self.loaded_paths: list[Path] = []

    def load_evaluation_result(self, path: Path) -> EvaluationResult:
        self.loaded_paths.append(path)
        return self.reports[path]


class FakeEvaluationComparisonReportWriter:
    """In-memory comparison report writer for application tests."""

    def __init__(self) -> None:
        self.written: tuple[Path, dict[str, object]] | None = None

    def write_evaluation_comparison_report(
        self,
        path: Path,
        record: dict[str, object],
    ) -> None:
        self.written = (path, record)


class FakeDatasetReader:
    """In-memory dataset reader for synthetic evaluation tests."""

    def __init__(self, examples: list[DatasetExample]) -> None:
        self.examples = examples
        self.loaded_paths: list[Path] = []

    def load_dataset_examples(self, path: Path) -> list[DatasetExample]:
        self.loaded_paths.append(path)
        return self.examples


class FakeModelProvider:
    """Minimal model provider for synthetic evaluation tests."""

    async def complete(self, prompt: str) -> str:
        return '{"ok": true}'


class RecordingModelProvider:
    """Scripted model provider that records prompts."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


@dataclass(frozen=True, slots=True)
class FakeSyntheticScore:
    """Minimal synthetic score for application evaluator tests."""

    category: str | None
    score: float
    parse_success: bool = True
    correct_tool: bool = True
    valid_arguments: bool = True
    exact_arguments: bool = True
    expects_refusal: bool = False
    safe_refusal: bool = True
    repair_success: bool = True
    unexpected_final_response: bool = False

    def as_record(self) -> dict[str, object]:
        return {
            "category": self.category,
            "score": self.score,
            "safe_refusal": self.safe_refusal,
        }


@dataclass(frozen=True, slots=True)
class FakeTraceScore:
    """Minimal trace score for application evaluator tests."""

    category: str | None
    score: float
    parse_success: bool = True
    final_response_match: bool = True
    patch_match: bool = True
    tool_history_match: bool = True

    def as_record(self) -> dict[str, object]:
        return {
            "category": self.category,
            "score": self.score,
            "tool_history_match": self.tool_history_match,
        }


@dataclass(frozen=True, slots=True)
class FakeWorkspaceStagedScore:
    """Minimal workspace-staged score for application evaluator tests."""

    category: str | None
    score: float
    parse_success: bool = True
    stage_score: float = 1.0
    stage_passed: bool = True

    def as_record(self) -> dict[str, object]:
        return {
            "category": self.category,
            "score": self.score,
            "parse_success": self.parse_success,
            "stages": [
                {
                    "name": "read_search",
                    "score": self.stage_score,
                    "passed": self.stage_passed,
                    "errors": [],
                }
            ],
        }


class FakeBehaviorEvaluationSuite:
    """In-memory behavior suite for synthetic evaluation tests."""

    def __init__(self, evaluation: EvaluationResult) -> None:
        self.evaluation = evaluation
        self.provider: FakeModelProvider | None = None
        self.examples: list[DatasetExample] | None = None

    async def evaluate_model(
        self,
        model_provider: FakeModelProvider,
        examples: list[DatasetExample],
    ) -> EvaluationResult:
        self.provider = model_provider
        self.examples = examples
        return self.evaluation


class FakeArtifactEvaluationSuite:
    """In-memory artifact suite for synthetic evaluation tests."""

    def __init__(self, evaluation: EvaluationResult) -> None:
        self.evaluation = evaluation
        self.artifact: ModelArtifact | None = None

    async def evaluate_artifact(self, artifact: ModelArtifact) -> EvaluationResult:
        self.artifact = artifact
        return self.evaluation


class FakeToolProfileSummarizer:
    """In-memory tool-profile summarizer for synthetic evaluation tests."""

    def __init__(self) -> None:
        self.examples: list[DatasetExample] | None = None
        self.default_available_tools: tuple[str, ...] | None = None

    def summarize_dataset_tool_profiles(
        self,
        examples: list[DatasetExample],
        *,
        default_available_tools: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        self.examples = examples
        self.default_available_tools = tuple(default_available_tools or ())
        return {
            "example_count": len(examples),
            "available_tools": list(self.default_available_tools),
        }


class FakeEvaluationResultWriter:
    """In-memory evaluation report writer for synthetic evaluation tests."""

    def __init__(self) -> None:
        self.written: tuple[Path, EvaluationResult, Path | None] | None = None

    def write_evaluation_result(
        self,
        run_dir: Path,
        result: EvaluationResult,
        output_path: Path | None = None,
    ) -> Path:
        self.written = (run_dir, result, output_path)
        return output_path or run_dir / "evaluation.json"


class FakeWorkspaceStagedReviewBuilder:
    """In-memory review record builder for application tests."""

    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.request: dict[str, object] | None = None

    def build_workspace_staged_review_records(
        self,
        *,
        examples: list[DatasetExample],
        reports: list[tuple[Path, EvaluationResult]],
        simple_failure_threshold: float = 0.4,
        auto_accept_threshold: float = 0.95,
    ) -> list[dict[str, object]]:
        self.request = {
            "examples": examples,
            "reports": reports,
            "simple_failure_threshold": simple_failure_threshold,
            "auto_accept_threshold": auto_accept_threshold,
        }
        return self.records


class FakeWorkspaceStagedReviewQueueWriter:
    """In-memory review queue writer for application tests."""

    def __init__(self) -> None:
        self.written: tuple[Path, list[dict[str, object]]] | None = None

    def write_workspace_staged_review_records(
        self,
        path: Path,
        records: list[dict[str, object]],
    ) -> None:
        self.written = (path, records)


def test_evaluation_comparison_workflow_loads_compares_and_writes_report() -> None:
    baseline_report = Path("base-evaluation.json")
    adapter_report = Path("adapter-evaluation.json")
    reader = FakeEvaluationResultReader(
        {
            baseline_report: EvaluationResult(
                passed=False,
                summary="base",
                score=0.70,
                details={"metrics": {"correct_tool_rate": 0.60}},
            ),
            adapter_report: EvaluationResult(
                passed=True,
                summary="adapter",
                score=0.84,
                details={"metrics": {"correct_tool_rate": 0.74}},
            ),
        }
    )
    writer = FakeEvaluationComparisonReportWriter()
    workflow = RunEvaluationComparisonWorkflow(
        evaluation_reader=reader,
        comparison_writer=writer,
    )

    result = workflow.run(
        RunEvaluationComparisonRequest(
            baseline_report_path=baseline_report,
            adapter_report_path=adapter_report,
            output_path=Path("comparison.json"),
            minimum_score_delta=0.10,
            minimum_metric_deltas={"correct_tool_rate": 0.10},
        )
    )

    assert reader.loaded_paths == [baseline_report, adapter_report]
    assert result.output_path == Path("comparison.json")
    assert result.comparison.passed is True
    assert result.comparison.score_delta == pytest.approx(0.14)
    assert writer.written is not None
    path, record = writer.written
    assert path == Path("comparison.json")
    assert record["passed"] is True
    assert record["score_delta"] == pytest.approx(0.14)
    metric_deltas = {
        delta["name"]: delta
        for delta in record["metric_deltas"]
        if isinstance(delta, dict)
    }
    assert metric_deltas["correct_tool_rate"]["passed"] is True


def test_evaluation_comparison_workflow_records_threshold_failures() -> None:
    baseline_report = Path("base-evaluation.json")
    adapter_report = Path("adapter-evaluation.json")
    reader = FakeEvaluationResultReader(
        {
            baseline_report: EvaluationResult(
                passed=True,
                summary="base",
                score=0.80,
                details={"metrics": {"tool_history_match_rate": 0.80}},
            ),
            adapter_report: EvaluationResult(
                passed=True,
                summary="adapter",
                score=0.82,
                details={"metrics": {"tool_history_match_rate": 0.84}},
            ),
        }
    )
    writer = FakeEvaluationComparisonReportWriter()
    workflow = RunEvaluationComparisonWorkflow(
        evaluation_reader=reader,
        comparison_writer=writer,
    )

    result = workflow.run(
        RunEvaluationComparisonRequest(
            baseline_report_path=baseline_report,
            adapter_report_path=adapter_report,
            output_path=Path("comparison.json"),
            minimum_score_delta=0.05,
            minimum_metric_deltas={"tool_history_match_rate": 0.10},
        )
    )

    assert result.comparison.passed is False
    assert result.comparison.errors == (
        "score delta 0.0200 is below minimum 0.0500",
        "metric 'tool_history_match_rate' delta 0.0400 is below minimum 0.1000",
    )
    assert writer.written is not None
    _, record = writer.written
    assert record["passed"] is False
    assert record["errors"] == list(result.comparison.errors)


def test_synthetic_behavior_suite_calls_model_and_scores_in_application() -> None:
    examples = [
        _example(source="synthetic:one"),
        _example(source="synthetic:two"),
    ]
    provider = RecordingModelProvider(['{"ok": true}', '{"ok": false}'])
    raw_responses: list[str] = []

    def score_example(example: DatasetExample, raw_response: str) -> FakeSyntheticScore:
        raw_responses.append(raw_response)
        return FakeSyntheticScore(
            category=str(example.source).split(":", 1)[1],
            score=1.0 if raw_response == '{"ok": true}' else 0.5,
        )

    suite = SyntheticBehaviorEvaluationSuite(
        score_example=cast(SyntheticExampleScorer, score_example),
        pass_threshold=0.75,
        default_available_tools=("repo.read",),
    )

    result = asyncio.run(suite.evaluate_model(provider, examples))

    assert raw_responses == ['{"ok": true}', '{"ok": false}']
    assert len(provider.prompts) == 2
    assert '"available_tools": ["repo.read"]' in provider.prompts[0]
    assert result.passed is True
    assert result.score == pytest.approx(0.75)
    assert result.details["metrics"]["correct_tool_rate"] == 1.0
    assert result.details["category_metrics"]["one"]["score"] == 1.0
    assert result.details["category_metrics"]["two"]["score"] == 0.5


def test_trace_behavior_suite_calls_model_and_scores_in_application() -> None:
    examples = [
        _example(source="trace:one"),
        _example(source="trace:two"),
    ]
    examples[0].metadata["tool_profile"] = {"available_tools": ["repo.read"]}
    examples[1].input["available_tools"] = ["repo.search"]
    provider = RecordingModelProvider(['{"final_response": "done"}', '{"patch": "diff"}'])
    raw_responses: list[str] = []

    def score_example(example: DatasetExample, raw_response: str) -> FakeTraceScore:
        raw_responses.append(raw_response)
        return FakeTraceScore(
            category=str(example.source).split(":", 1)[1],
            score=1.0 if raw_response == '{"final_response": "done"}' else 0.25,
            tool_history_match=raw_response == '{"final_response": "done"}',
        )

    suite = TraceBehaviorEvaluationSuite(
        score_example=cast(TraceExampleScorer, score_example),
        pass_threshold=0.5,
        default_available_tools=("git.diff",),
    )

    result = asyncio.run(suite.evaluate_model(provider, examples))

    assert raw_responses == ['{"final_response": "done"}', '{"patch": "diff"}']
    assert '"available_tools": ["repo.read"]' in provider.prompts[0]
    assert '"available_tools": ["repo.search"]' in provider.prompts[1]
    assert result.passed is True
    assert result.score == pytest.approx(0.625)
    assert result.details["metrics"]["tool_history_match_rate"] == 0.5
    assert result.details["category_metrics"]["one"]["score"] == 1.0
    assert result.details["category_metrics"]["two"]["score"] == 0.25


def test_workspace_staged_suite_calls_model_and_scores_in_application() -> None:
    examples = [
        _example(source="workspace:one"),
        _example(source="workspace:two"),
    ]
    examples[0].metadata["tool_profile"] = {"available_tools": ["repo.read"]}
    examples[1].input["available_tools"] = ["repo.search"]
    examples[0].input["workspace_files"] = {"src/app.py": "print('hello')"}
    provider = RecordingModelProvider(['{"read_search": {}}', '{"diagnosis": {}}'])
    raw_responses: list[str] = []

    def score_example(
        example: DatasetExample,
        raw_response: str,
    ) -> FakeWorkspaceStagedScore:
        raw_responses.append(raw_response)
        return FakeWorkspaceStagedScore(
            category=str(example.source).split(":", 1)[1],
            score=1.0 if raw_response == '{"read_search": {}}' else 0.25,
            stage_score=1.0 if raw_response == '{"read_search": {}}' else 0.0,
            stage_passed=raw_response == '{"read_search": {}}',
        )

    suite = WorkspaceStagedEvaluationSuite(
        score_example=cast(WorkspaceStagedExampleScorer, score_example),
        pass_threshold=0.5,
        rubric_version="v2",
        default_available_tools=("git.diff",),
    )

    result = asyncio.run(suite.evaluate_model(provider, examples))

    assert raw_responses == ['{"read_search": {}}', '{"diagnosis": {}}']
    assert '"workspace_files": {"src/app.py": "print' in provider.prompts[0]
    assert '"available_tools": ["repo.read"]' in provider.prompts[0]
    assert '"available_tools": ["repo.search"]' in provider.prompts[1]
    assert result.passed is True
    assert result.score == pytest.approx(0.625)
    assert result.details["rubric_version"] == "v2"
    assert result.details["metrics"]["parse_success_rate"] == 1.0
    assert result.details["metrics"]["read_search_score"] == 0.5
    assert result.details["metrics"]["read_search_pass_rate"] == 0.5
    assert result.details["category_metrics"]["one"]["score"] == 1.0
    assert result.details["category_metrics"]["two"]["score"] == 0.25


def test_synthetic_evaluation_workflow_evaluates_provider_and_writes_metadata() -> None:
    examples = [_example(source="synthetic:one"), _example(source="synthetic:two")]
    reader = FakeDatasetReader(examples)
    provider = FakeModelProvider()
    behavior_suite = FakeBehaviorEvaluationSuite(
        EvaluationResult(
            passed=True,
            summary="behavioral synthetic eval scored 1.00 over 1 example(s)",
            score=1.0,
            details={"example_count": 1},
        )
    )
    artifact_suite = FakeArtifactEvaluationSuite(
        EvaluationResult(passed=False, summary="unused", score=0.0)
    )
    summarizer = FakeToolProfileSummarizer()
    writer = FakeEvaluationResultWriter()
    workflow = RunSyntheticEvaluationWorkflow(
        example_reader=reader,
        behavior_suite=behavior_suite,
        artifact_suite=artifact_suite,
        tool_profile_summarizer=summarizer,
        evaluation_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunSyntheticEvaluationRequest(
                run_id="latest",
                run_dir=Path("runs/latest"),
                dataset_path=Path("held-out.jsonl"),
                provider_kind="scripted",
                model_provider=provider,
                model="scripted-model",
                base_model="base-model",
                adapter_path=Path("adapter"),
                max_examples=1,
                output_path=Path("synthetic-evaluation.json"),
                default_available_tools=("repo.read", "repo.write_patch"),
            )
        )
    )

    assert reader.loaded_paths == [Path("held-out.jsonl")]
    assert behavior_suite.provider is provider
    assert behavior_suite.examples == [examples[0]]
    assert artifact_suite.artifact is None
    assert summarizer.examples == [examples[0]]
    assert summarizer.default_available_tools == ("repo.read", "repo.write_patch")
    assert result.report_path == Path("synthetic-evaluation.json")
    assert result.evaluation.details["evaluation_metadata"] == {
        "run_id": "latest",
        "dataset_path": "held-out.jsonl",
        "provider": "scripted",
        "model": "scripted-model",
        "base_model": "base-model",
        "adapter_path": "adapter",
        "tool_profile": {
            "example_count": 1,
            "available_tools": ["repo.read", "repo.write_patch"],
        },
    }
    assert writer.written == (
        Path("runs/latest"),
        result.evaluation,
        Path("synthetic-evaluation.json"),
    )


def test_synthetic_evaluation_workflow_falls_back_to_artifact_evaluation() -> None:
    examples = [_example()]
    artifact = ModelArtifact(
        name="adapter",
        kind=ModelArtifactKind.ADAPTER,
        path="runs/latest/adapter",
        base_model="base-model",
    )
    artifact_suite = FakeArtifactEvaluationSuite(
        EvaluationResult(
            passed=True,
            summary="metadata-only synthetic artifact check passed",
            score=1.0,
            details={"metadata_only": True},
        )
    )
    workflow = RunSyntheticEvaluationWorkflow(
        example_reader=FakeDatasetReader(examples),
        behavior_suite=FakeBehaviorEvaluationSuite(
            EvaluationResult(passed=False, summary="unused", score=0.0)
        ),
        artifact_suite=artifact_suite,
        tool_profile_summarizer=FakeToolProfileSummarizer(),
        evaluation_writer=FakeEvaluationResultWriter(),
    )

    result = asyncio.run(
        workflow.run(
            RunSyntheticEvaluationRequest(
                run_id="dry-run",
                run_dir=Path("runs/dry-run"),
                dataset_path=Path("held-out.jsonl"),
                provider_kind="artifact",
                artifact=artifact,
            )
        )
    )

    assert artifact_suite.artifact is artifact
    assert result.report_path == Path("runs/dry-run/evaluation.json")
    assert result.evaluation.details["evaluation_metadata"]["base_model"] == "base-model"
    assert (
        result.evaluation.details["evaluation_metadata"]["adapter_path"]
        == "runs/latest/adapter"
    )


def test_synthetic_evaluation_workflow_requires_provider_or_artifact() -> None:
    workflow = RunSyntheticEvaluationWorkflow(
        example_reader=FakeDatasetReader([]),
        behavior_suite=FakeBehaviorEvaluationSuite(
            EvaluationResult(passed=False, summary="unused", score=0.0)
        ),
        artifact_suite=FakeArtifactEvaluationSuite(
            EvaluationResult(passed=False, summary="unused", score=0.0)
        ),
        tool_profile_summarizer=FakeToolProfileSummarizer(),
        evaluation_writer=FakeEvaluationResultWriter(),
    )

    with pytest.raises(
        ValueError,
        match="synthetic eval requires a training artifact, runnable model, or scripted response",
    ):
        asyncio.run(
            workflow.run(
                RunSyntheticEvaluationRequest(
                    run_id="latest",
                    run_dir=Path("runs/latest"),
                    dataset_path=Path("held-out.jsonl"),
                    provider_kind="none",
                )
            )
        )


def test_trace_evaluation_workflow_evaluates_provider_and_writes_metadata() -> None:
    examples = [_example(source="trace:one"), _example(source="trace:two")]
    reader = FakeDatasetReader(examples)
    provider = FakeModelProvider()
    behavior_suite = FakeBehaviorEvaluationSuite(
        EvaluationResult(
            passed=True,
            summary="trace behavior eval scored 1.00 over 1 example(s)",
            score=1.0,
            details={"example_count": 1},
        )
    )
    summarizer = FakeToolProfileSummarizer()
    writer = FakeEvaluationResultWriter()
    workflow = RunTraceEvaluationWorkflow(
        example_reader=reader,
        behavior_suite=behavior_suite,
        tool_profile_summarizer=summarizer,
        evaluation_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunTraceEvaluationRequest(
                run_id="latest",
                run_dir=Path("runs/latest"),
                dataset_path=Path("held-out-trace.jsonl"),
                provider_kind="scripted",
                model_provider=provider,
                model="scripted-model",
                base_model="base-model",
                adapter_path=Path("adapter"),
                max_examples=1,
                output_path=Path("trace-evaluation.json"),
                default_available_tools=("repo.read",),
            )
        )
    )

    assert reader.loaded_paths == [Path("held-out-trace.jsonl")]
    assert behavior_suite.provider is provider
    assert behavior_suite.examples == [examples[0]]
    assert summarizer.examples == [examples[0]]
    assert summarizer.default_available_tools == ("repo.read",)
    assert result.report_path == Path("trace-evaluation.json")
    assert result.evaluation.details["evaluation_metadata"] == {
        "run_id": "latest",
        "dataset_path": "held-out-trace.jsonl",
        "provider": "scripted",
        "model": "scripted-model",
        "base_model": "base-model",
        "adapter_path": "adapter",
        "tool_profile": {
            "example_count": 1,
            "available_tools": ["repo.read"],
        },
    }
    assert writer.written == (
        Path("runs/latest"),
        result.evaluation,
        Path("trace-evaluation.json"),
    )


def test_trace_evaluation_workflow_requires_provider() -> None:
    workflow = RunTraceEvaluationWorkflow(
        example_reader=FakeDatasetReader([]),
        behavior_suite=FakeBehaviorEvaluationSuite(
            EvaluationResult(passed=False, summary="unused", score=0.0)
        ),
        tool_profile_summarizer=FakeToolProfileSummarizer(),
        evaluation_writer=FakeEvaluationResultWriter(),
    )

    with pytest.raises(
        ValueError,
        match="trace eval requires a runnable model, adapter, or scripted response",
    ):
        asyncio.run(
            workflow.run(
                RunTraceEvaluationRequest(
                    run_id="latest",
                    run_dir=Path("runs/latest"),
                    dataset_path=Path("held-out-trace.jsonl"),
                    provider_kind="none",
                    model_provider=None,
                )
            )
        )


def test_workspace_staged_evaluation_workflow_evaluates_provider_and_writes_metadata() -> None:
    examples = [_example(source="workspace:one"), _example(source="workspace:two")]
    reader = FakeDatasetReader(examples)
    provider = FakeModelProvider()
    behavior_suite = FakeBehaviorEvaluationSuite(
        EvaluationResult(
            passed=True,
            summary="staged workspace eval scored 1.00 over 1 example(s)",
            score=1.0,
            details={"example_count": 1},
        )
    )
    summarizer = FakeToolProfileSummarizer()
    writer = FakeEvaluationResultWriter()
    workflow = RunWorkspaceStagedEvaluationWorkflow(
        example_reader=reader,
        behavior_suite=behavior_suite,
        tool_profile_summarizer=summarizer,
        evaluation_writer=writer,
    )

    result = asyncio.run(
        workflow.run(
            RunWorkspaceStagedEvaluationRequest(
                run_id="latest",
                run_dir=Path("runs/latest"),
                dataset_path=Path("held-out-workspace.jsonl"),
                provider_kind="scripted",
                model_provider=provider,
                model="scripted-model",
                base_model="base-model",
                adapter_path=Path("adapter"),
                max_examples=1,
                output_path=Path("workspace-evaluation.json"),
                default_available_tools=("repo.read", "repo.write_patch"),
            )
        )
    )

    assert reader.loaded_paths == [Path("held-out-workspace.jsonl")]
    assert behavior_suite.provider is provider
    assert behavior_suite.examples == [examples[0]]
    assert summarizer.examples == [examples[0]]
    assert result.report_path == Path("workspace-evaluation.json")
    assert result.evaluation.details["evaluation_metadata"] == {
        "run_id": "latest",
        "dataset_path": "held-out-workspace.jsonl",
        "provider": "scripted",
        "model": "scripted-model",
        "base_model": "base-model",
        "adapter_path": "adapter",
        "tool_profile": {
            "example_count": 1,
            "available_tools": ["repo.read", "repo.write_patch"],
        },
    }
    assert writer.written == (
        Path("runs/latest"),
        result.evaluation,
        Path("workspace-evaluation.json"),
    )


def test_workspace_staged_evaluation_workflow_requires_provider() -> None:
    workflow = RunWorkspaceStagedEvaluationWorkflow(
        example_reader=FakeDatasetReader([]),
        behavior_suite=FakeBehaviorEvaluationSuite(
            EvaluationResult(passed=False, summary="unused", score=0.0)
        ),
        tool_profile_summarizer=FakeToolProfileSummarizer(),
        evaluation_writer=FakeEvaluationResultWriter(),
    )

    with pytest.raises(
        ValueError,
        match="workspace-staged eval requires a runnable model, adapter, or scripted response",
    ):
        asyncio.run(
            workflow.run(
                RunWorkspaceStagedEvaluationRequest(
                    run_id="latest",
                    run_dir=Path("runs/latest"),
                    dataset_path=Path("held-out-workspace.jsonl"),
                    provider_kind="none",
                    model_provider=None,
                )
            )
        )


def test_workspace_staged_review_workflow_builds_and_writes_records() -> None:
    examples = [_example()]
    report_path = Path("workspace-evaluation.json")
    report = EvaluationResult(
        passed=False,
        summary="staged workspace eval scored 0.50 over 1 example(s)",
        score=0.5,
        details={},
    )
    reader = FakeDatasetReader(examples)
    evaluation_reader = FakeEvaluationResultReader({report_path: report})
    records = [
        {
            "example_id": "example-1",
            "auto_triage": {"decision": "needs_human_review"},
        }
    ]
    builder = FakeWorkspaceStagedReviewBuilder(records)
    writer = FakeWorkspaceStagedReviewQueueWriter()
    workflow = RunWorkspaceStagedReviewWorkflow(
        example_reader=reader,
        evaluation_reader=evaluation_reader,
        review_builder=builder,
        review_writer=writer,
    )

    build_result = workflow.build(
        RunWorkspaceStagedReviewRequest(
            dataset_path=Path("workspace.jsonl"),
            report_paths=(report_path,),
            output_path=Path("review_queue.jsonl"),
            simple_failure_threshold=0.3,
            auto_accept_threshold=0.9,
        )
    )
    write_result = workflow.write(
        RunWorkspaceStagedReviewWriteRequest(
            output_path=build_result.output_path,
            records=build_result.records,
        )
    )

    assert reader.loaded_paths == [Path("workspace.jsonl")]
    assert evaluation_reader.loaded_paths == [report_path]
    assert builder.request is not None
    assert builder.request["examples"] == examples
    assert builder.request["reports"] == [(report_path, report)]
    assert builder.request["simple_failure_threshold"] == 0.3
    assert builder.request["auto_accept_threshold"] == 0.9
    assert build_result.auto_triage_counts == {"needs_human_review": 1}
    assert writer.written == (Path("review_queue.jsonl"), records)
    assert write_result.record_count == 1
    assert write_result.auto_triage_counts == {"needs_human_review": 1}


def test_workspace_staged_review_workflow_requires_report_paths() -> None:
    workflow = RunWorkspaceStagedReviewWorkflow(
        example_reader=FakeDatasetReader([]),
        evaluation_reader=FakeEvaluationResultReader({}),
        review_builder=FakeWorkspaceStagedReviewBuilder([]),
        review_writer=FakeWorkspaceStagedReviewQueueWriter(),
    )

    with pytest.raises(ValueError, match="at least one --report is required"):
        workflow.build(
            RunWorkspaceStagedReviewRequest(
                dataset_path=Path("workspace.jsonl"),
                report_paths=(),
                output_path=Path("review_queue.jsonl"),
            )
        )


def _example(source: str = "synthetic:test") -> DatasetExample:
    return DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "Fix tests"},
        target={"final_response": "Tests fixed."},
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
        source=source,
    )
