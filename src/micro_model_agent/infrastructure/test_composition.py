"""Tests for shared runtime composition helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.application.datasets import (
    RunDatasetExportWorkflow,
    RunDatasetMergeWorkflow,
    RunDatasetRelabelWorkflow,
    RunDatasetSynthesisWorkflow,
    RunDatasetValidationWorkflow,
    RunTraceDatasetExportWorkflow,
    RunTraceReviewWorkflow,
)
from micro_model_agent.application.evaluation import (
    RunEvaluationComparisonWorkflow,
    RunSyntheticEvaluationWorkflow,
    RunTraceEvaluationWorkflow,
    RunWorkspaceStagedEvaluationWorkflow,
    RunWorkspaceStagedReviewWorkflow,
)
from micro_model_agent.application.promotion import (
    RunPromotionGateWorkflow,
    RunPromotionListWorkflow,
    RunPromotionPackageOllamaWorkflow,
    RunPromotionRecordWorkflow,
    RunPromotionSelectWorkflow,
)
from micro_model_agent.infrastructure.composition import (
    allowed_test_commands,
    build_builtin_tool_executor,
    build_dataset_export_workflow,
    build_dataset_merge_workflow,
    build_dataset_relabel_workflow,
    build_dataset_synthesis_workflow,
    build_dataset_validation_workflow,
    build_evaluation_comparison_workflow,
    build_jsonl_dataset_example_store,
    build_promotion_gate_workflow,
    build_promotion_list_workflow,
    build_promotion_package_ollama_workflow,
    build_promotion_record_workflow,
    build_promotion_select_workflow,
    build_synthetic_evaluation_workflow,
    build_synthetic_training_workflow,
    build_trace_dataset_export_workflow,
    build_trace_evaluation_workflow,
    build_trace_review_workflow,
    build_workspace_staged_evaluation_workflow,
    build_workspace_staged_review_workflow,
    comparison_trace_store,
    default_evaluation_available_tools,
    initialize_local_repository,
    local_repository_initialized,
    register_workspace_record,
    registered_workspace_path,
    resolve_model_options,
    select_evaluation_model,
    workflow_trace_store,
    workspace_registry,
    write_local_repository_index,
)
from micro_model_agent.infrastructure.models.fake import ScriptedModelProvider
from micro_model_agent.infrastructure.models.ollama import OllamaModelProvider
from micro_model_agent.infrastructure.models.transformers import (
    TransformersPeftModelProvider,
)
from micro_model_agent.infrastructure.persistence.dataset_store import JsonlDatasetExampleStore
from micro_model_agent.infrastructure.repositories.local_index import LocalIndexResult
from micro_model_agent.infrastructure.repositories.metadata import (
    initialize_repository,
    update_model_configuration,
)
from micro_model_agent.infrastructure.tools.executor import BuiltinToolExecutor
from micro_model_agent.infrastructure.training.artifacts import FakeTrainingRunner
from micro_model_agent.infrastructure.training.local_finetuning import LocalFineTuningRunner


def test_resolve_model_options_uses_selected_repository_config(tmp_path: Path) -> None:
    update_model_configuration(
        tmp_path,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=tmp_path / "training" / "runs" / "proof" / "adapter",
        selected_promotion={"artifact_id": "00000000-0000-4000-8000-000000000001"},
    )

    options = resolve_model_options(
        repository_root=tmp_path,
        env={},
    )

    assert options.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert options.adapter_path == tmp_path / "training" / "runs" / "proof" / "adapter"
    assert options.selected_promotion_artifact_id == "00000000-0000-4000-8000-000000000001"


def test_resolve_model_options_prefer_args_over_env_and_config(tmp_path: Path) -> None:
    initialize_repository(
        tmp_path,
        default_model="configured-ollama",
        base_model="configured-base",
        adapter_path="configured-adapter",
    )

    options = resolve_model_options(
        repository_root=tmp_path,
        model="explicit-ollama",
        base_model="explicit-base",
        adapter_path=Path("explicit-adapter"),
        env={
            "MICRO_MODEL_AGENT_DEFAULT_MODEL": "env-ollama",
            "MICRO_MODEL_AGENT_BASE_MODEL": "env-base",
            "MICRO_MODEL_AGENT_ADAPTER_PATH": "env-adapter",
        },
    )

    assert options.model == "explicit-ollama"
    assert options.base_model == "explicit-base"
    assert options.adapter_path == Path("explicit-adapter")


def test_resolve_model_options_can_disable_adapter(tmp_path: Path) -> None:
    update_model_configuration(
        tmp_path,
        base_model="configured-base",
        adapter_path="configured-adapter",
    )

    options = resolve_model_options(
        repository_root=tmp_path,
        use_adapter=False,
        env={},
    )

    assert options.base_model == "configured-base"
    assert options.adapter_path is None


def test_allowed_test_commands_adds_default_pytest() -> None:
    commands = allowed_test_commands(None, None, use_default_pytest=True)

    assert list(commands) == ["pytest"]
    assert commands["pytest"].args == ("python3", "-m", "pytest", "-q")


def test_runtime_factories_use_standard_repository_paths(tmp_path: Path) -> None:
    executor = build_builtin_tool_executor(
        tmp_path,
        allowed_test_commands("unit", ["python", "-m", "pytest"]),
    )

    assert isinstance(executor, BuiltinToolExecutor)
    assert workflow_trace_store(tmp_path).path == tmp_path / ".traces" / "workflows.jsonl"
    assert comparison_trace_store(tmp_path).path == (
        tmp_path / ".traces" / "comparison_sessions.jsonl"
    )
    assert workspace_registry(tmp_path).path == (
        tmp_path / ".micro_model_agent" / "workspaces.jsonl"
    )


def test_repository_cli_factories_use_local_adapters(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def run() -> None:\n    pass\n", encoding="utf-8")

    init_result = initialize_local_repository(tmp_path, default_model="qwen:test")
    index_result = write_local_repository_index(tmp_path, max_file_bytes=1_000_000)
    dataset_store = build_jsonl_dataset_example_store(tmp_path / "datasets" / "tasks.jsonl")

    assert init_result.ok
    assert init_result.config["model"]["default_model"] == "qwen:test"
    assert isinstance(index_result, LocalIndexResult)
    assert index_result.indexed_file_count >= 1
    assert isinstance(dataset_store, JsonlDatasetExampleStore)


def test_workspace_factories_register_and_resolve_records(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    workspace_root = tmp_path / "workspace"

    assert local_repository_initialized(workspace_root) is False

    initialize_result = initialize_local_repository(workspace_root)
    record = asyncio.run(
        register_workspace_record(
            registry_root=registry_root,
            workspace_path=workspace_root,
            name="chat-workspace",
            metadata={"source": "test"},
        )
    )
    resolved_path = asyncio.run(
        registered_workspace_path(
            registry_root=registry_root,
            workspace_id=record["id"],
        )
    )

    assert initialize_result.ok
    assert local_repository_initialized(workspace_root) is True
    assert record["path"] == str(workspace_root)
    assert record["metadata"] == {"source": "test"}
    assert resolved_path == workspace_root


def test_build_synthetic_training_workflow_selects_runner_by_dry_run(
    tmp_path: Path,
) -> None:
    dry_run_workflow = build_synthetic_training_workflow(
        dry_run=True,
        artifact_store_root=tmp_path / "artifacts",
    )
    real_workflow = build_synthetic_training_workflow(
        dry_run=False,
        artifact_store_root=tmp_path / "artifacts",
    )

    assert isinstance(dry_run_workflow.runner, FakeTrainingRunner)
    assert isinstance(real_workflow.runner, LocalFineTuningRunner)


def test_dataset_workflow_factories_return_standard_workflows(tmp_path: Path) -> None:
    assert isinstance(
        build_dataset_synthesis_workflow(template_dir=tmp_path),
        RunDatasetSynthesisWorkflow,
    )
    assert isinstance(build_dataset_validation_workflow(), RunDatasetValidationWorkflow)
    assert isinstance(build_dataset_export_workflow(), RunDatasetExportWorkflow)
    assert isinstance(
        build_trace_dataset_export_workflow(
            trace_path=tmp_path / "workflows.jsonl",
            review_path=tmp_path / "reviews.jsonl",
        ),
        RunTraceDatasetExportWorkflow,
    )
    assert isinstance(build_trace_review_workflow(), RunTraceReviewWorkflow)
    assert isinstance(build_dataset_relabel_workflow(), RunDatasetRelabelWorkflow)
    assert isinstance(build_dataset_merge_workflow(), RunDatasetMergeWorkflow)


def test_promotion_workflow_factories_return_standard_workflows() -> None:
    assert isinstance(build_promotion_gate_workflow(), RunPromotionGateWorkflow)
    assert isinstance(build_promotion_record_workflow(), RunPromotionRecordWorkflow)
    assert isinstance(build_promotion_list_workflow(), RunPromotionListWorkflow)
    assert isinstance(build_promotion_select_workflow(), RunPromotionSelectWorkflow)
    assert isinstance(
        build_promotion_package_ollama_workflow(),
        RunPromotionPackageOllamaWorkflow,
    )


def test_evaluation_workflow_factories_return_standard_workflows() -> None:
    assert default_evaluation_available_tools()
    assert isinstance(
        build_synthetic_evaluation_workflow(pass_threshold=0.8),
        RunSyntheticEvaluationWorkflow,
    )
    assert isinstance(
        build_trace_evaluation_workflow(pass_threshold=0.8),
        RunTraceEvaluationWorkflow,
    )
    assert isinstance(
        build_workspace_staged_evaluation_workflow(
            pass_threshold=0.8,
            rubric_version="legacy",
        ),
        RunWorkspaceStagedEvaluationWorkflow,
    )
    assert isinstance(build_workspace_staged_review_workflow(), RunWorkspaceStagedReviewWorkflow)
    assert isinstance(build_evaluation_comparison_workflow(), RunEvaluationComparisonWorkflow)


def test_select_evaluation_model_prefers_scripted_responses(tmp_path: Path) -> None:
    selection = select_evaluation_model(
        run_dir=tmp_path,
        model="ignored-ollama",
        base_model="metadata-base",
        adapter_path=tmp_path / "missing-adapter",
        scripted_responses=['{"ok": true}'],
        max_new_tokens=32,
    )

    assert selection.provider_kind == "scripted"
    assert isinstance(selection.provider, ScriptedModelProvider)
    assert selection.model == "ignored-ollama"
    assert selection.base_model == "metadata-base"
    assert selection.adapter_path == tmp_path / "missing-adapter"


def test_select_evaluation_model_uses_explicit_adapter(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "configured-base"}',
        encoding="utf-8",
    )

    selection = select_evaluation_model(
        run_dir=tmp_path,
        adapter_path=adapter_path,
        scripted_responses=[],
        max_new_tokens=48,
    )

    assert selection.provider_kind == "transformers_peft"
    assert isinstance(selection.provider, TransformersPeftModelProvider)
    assert selection.base_model == "configured-base"
    assert selection.adapter_path == adapter_path
    assert selection.provider.max_new_tokens == 48


def test_select_evaluation_model_uses_ollama_options(tmp_path: Path) -> None:
    selection = select_evaluation_model(
        run_dir=tmp_path,
        model="qwen:test",
        scripted_responses=[],
        max_new_tokens=64,
        env={"MICRO_MODEL_AGENT_OLLAMA_BASE_URL": "http://localhost:11434"},
    )

    assert selection.provider_kind == "ollama"
    assert isinstance(selection.provider, OllamaModelProvider)
    assert selection.provider.model_name == "qwen:test"
    assert selection.provider.options == {"num_predict": 64}


def test_select_evaluation_model_uses_runnable_training_artifact(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-4000-8000-000000000001",
                "name": "trained-adapter",
                "kind": "adapter",
                "path": str(adapter_path),
                "base_model": "artifact-base",
                "metrics": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    selection = select_evaluation_model(
        run_dir=run_dir,
        scripted_responses=[],
        max_new_tokens=96,
    )

    assert selection.provider_kind == "training_artifact"
    assert isinstance(selection.provider, TransformersPeftModelProvider)
    assert selection.base_model == "artifact-base"
    assert selection.adapter_path == adapter_path
    assert selection.artifact is not None
    assert selection.artifact.name == "trained-adapter"


def test_select_evaluation_model_returns_metadata_fallback_for_dry_run_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact_path = tmp_path / "missing-adapter"
    (run_dir / "artifact.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-4000-8000-000000000001",
                "name": "dry-run-adapter",
                "kind": "adapter",
                "path": str(artifact_path),
                "base_model": "artifact-base",
                "metrics": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    selection = select_evaluation_model(
        run_dir=run_dir,
        scripted_responses=[],
        max_new_tokens=96,
    )

    assert selection.provider is None
    assert selection.provider_kind == "metadata"
    assert selection.artifact is not None
    assert selection.artifact.path == str(artifact_path)
