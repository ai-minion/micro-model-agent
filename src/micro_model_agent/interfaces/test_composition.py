"""Tests for the interface-facing runtime composition facade."""

from __future__ import annotations

from types import ModuleType

from micro_model_agent.interfaces import composition
from micro_model_agent.execution.infrastructure import agents_runtime as agent_runtime
from micro_model_agent.execution.infrastructure import composition as execution_composition
from micro_model_agent.dataset.infrastructure import runtime as dataset_runtime
from micro_model_agent.evaluation.infrastructure import runtime as evaluation_runtime
from micro_model_agent.execution.infrastructure.models import runtime as model_runtime
from micro_model_agent.execution.infrastructure import persistence_runtime
from micro_model_agent.promotion.infrastructure import runtime as promotion_runtime
from micro_model_agent.repository_ops.infrastructure import runtime as repository_runtime
from micro_model_agent.repository_ops.infrastructure import tools_runtime as tool_runtime
from micro_model_agent.training.infrastructure import runtime as training_runtime

FACADE_EXPORTS_BY_OWNER: dict[ModuleType, tuple[str, ...]] = {
    agent_runtime: (
        "build_static_coding_workflow",
    ),
    execution_composition: (
        "build_coding_workflow",
        "build_event_pipeline",
    ),
    dataset_runtime: (
        "build_dataset_export_workflow",
        "build_dataset_merge_workflow",
        "build_dataset_relabel_workflow",
        "build_dataset_synthesis_workflow",
        "build_dataset_validation_workflow",
        "build_jsonl_dataset_example_store",
        "build_trace_dataset_export_workflow",
        "build_trace_review_workflow",
    ),
    evaluation_runtime: (
        "build_evaluation_comparison_workflow",
        "build_synthetic_evaluation_workflow",
        "build_trace_evaluation_workflow",
        "build_workspace_staged_evaluation_workflow",
        "build_workspace_staged_review_workflow",
        "default_evaluation_available_tools",
    ),
    model_runtime: (
        "ConfiguredToolLoopResult",
        "EvaluationModelSelection",
        "RuntimeModelOptions",
        "base_model_from_adapter",
        "build_loop_model_provider",
        "build_model_provider",
        "loop_budget_response",
        "path_config_value",
        "path_env",
        "path_or_none",
        "resolve_mcp_model_settings",
        "resolve_model_options",
        "run_cli_tool_loop",
        "run_configured_tool_loop",
        "run_mcp_agent_loop",
        "runtime_model_metadata",
        "select_evaluation_model",
        "string_config_value",
        "tool_prompt_schemas",
    ),
    persistence_runtime: (
        "DEFAULT_TRACE_DIR",        "append_comparison_trace_event",
        "comparison_trace_store",
        "register_workspace_record",
        "registered_workspace_path",
        "review_comparison_trace_session",
        "start_comparison_trace_session",
        "stop_comparison_trace_session",
        "trace_dir",
        "workflow_trace_record",
        "workflow_trace_store",
        "workspace_registry",
    ),
    promotion_runtime: (
        "build_promotion_gate_workflow",
        "build_promotion_list_workflow",
        "build_promotion_package_ollama_workflow",
        "build_promotion_record_workflow",
        "build_promotion_select_workflow",
    ),
    repository_runtime: (
        "initialize_local_repository",
        "local_repository_initialized",
        "write_local_repository_index",
    ),
    tool_runtime: (
        "PatchPolicyToolExecutor",
        "allowed_test_commands",
        "build_builtin_tool_executor",
        "builtin_tool_exists",
        "builtin_tool_names",
        "builtin_tool_summaries",
        "builtin_tools_response",
        "execute_builtin_tool_request",
    ),
    training_runtime: (
        "build_synthetic_training_workflow",
    ),
}


def test_composition_exports_match_owner_modules() -> None:
    expected_exports = {
        name for names in FACADE_EXPORTS_BY_OWNER.values() for name in names
    }

    assert set(composition.__all__) == expected_exports
    assert sorted(composition.__all__) == composition.__all__
    assert all(not name.startswith("_") for name in composition.__all__)

    for owner_module, names in FACADE_EXPORTS_BY_OWNER.items():
        for name in names:
            assert getattr(composition, name) is getattr(owner_module, name)
