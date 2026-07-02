"""Architecture boundary characterization tests."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
PROJECT_IMPORT_PREFIX = "micro_model_agent."
APPLICATION_BANNED_PREFIXES = (
    "micro_model_agent.agents",
    "micro_model_agent.infrastructure",
    "micro_model_agent.interfaces",
)
INTERFACE_BANNED_PREFIXES = ("micro_model_agent.agents",)
INTERFACE_ALLOWED_INFRASTRUCTURE_PREFIXES = (
    "micro_model_agent.infrastructure.composition",
)
INTERFACE_BANNED_CONCRETE_ADAPTER_PREFIXES = (
    "micro_model_agent.infrastructure.models",
    "micro_model_agent.infrastructure.fake_model_provider",
    "micro_model_agent.infrastructure.ollama_model_provider",
    "micro_model_agent.infrastructure.transformers_model_provider",
    "micro_model_agent.infrastructure.tools.catalog",
    "micro_model_agent.infrastructure.tools.command_runner",
    "micro_model_agent.infrastructure.tools.executor",
    "micro_model_agent.infrastructure.tool_executor",
    "micro_model_agent.infrastructure.persistence.comparison_trace",
    "micro_model_agent.infrastructure.persistence.dataset_store",
    "micro_model_agent.infrastructure.persistence.trace_store",
    "micro_model_agent.infrastructure.persistence.workspace_registry",
    "micro_model_agent.infrastructure.promotion.gate",
    "micro_model_agent.infrastructure.repositories.local_index",
    "micro_model_agent.infrastructure.repositories.metadata",
    "micro_model_agent.infrastructure.training.artifacts",
    "micro_model_agent.infrastructure.training.local_finetuning",
    "micro_model_agent.infrastructure.training.ollama_packaging",
    "micro_model_agent.infrastructure.datasets",
    "micro_model_agent.infrastructure.evaluation",
    "micro_model_agent.infrastructure.traces",
)
APPLICATION_BANNED_EXTERNALS = ("typer", "mcp", "pydantic")
DOMAIN_BANNED_EXTERNALS = (
    "pydantic",
    "typer",
    "mcp",
    "fastmcp",
    "datasets",
    "peft",
    "torch",
    "transformers",
)
PURE_RUBRIC_BANNED_PREFIXES = (
    "micro_model_agent.agents",
    "micro_model_agent.infrastructure",
    "micro_model_agent.interfaces",
)
PURE_RUBRIC_BANNED_EXTERNALS = (
    "mcp",
    "fastmcp",
    "pydantic",
    "typer",
    "datasets",
    "peft",
    "torch",
    "transformers",
)
FLAT_MODEL_PROVIDER_MODULES = (
    "micro_model_agent.infrastructure.fake_model_provider",
    "micro_model_agent.infrastructure.ollama_model_provider",
    "micro_model_agent.infrastructure.transformers_model_provider",
)
FLAT_REPOSITORY_MODULES = (
    "micro_model_agent.infrastructure.local_index",
    "micro_model_agent.infrastructure.local_retrieval",
    "micro_model_agent.infrastructure.repository_metadata",
    "micro_model_agent.infrastructure.repository_paths",
)
FLAT_PERSISTENCE_MODULES = (
    "micro_model_agent.infrastructure.comparison_trace",
    "micro_model_agent.infrastructure.dataset_store",
    "micro_model_agent.infrastructure.trace_store",
    "micro_model_agent.infrastructure.training_records",
    "micro_model_agent.infrastructure.workspace_registry",
)
FLAT_TRAINING_MODULES = (
    "micro_model_agent.infrastructure.local_finetuning",
    "micro_model_agent.infrastructure.ollama_packaging",
)
FLAT_PROMOTION_MODULES = ("micro_model_agent.infrastructure.promotion_gate",)
FLAT_EVALUATION_MODULES = (
    "micro_model_agent.infrastructure.artifact_evaluation",
    "micro_model_agent.infrastructure.evaluation_comparison",
    "micro_model_agent.infrastructure.evaluation_reports",
    "micro_model_agent.infrastructure.evaluation_response_parsing",
    "micro_model_agent.infrastructure.synthetic_rubrics",
    "micro_model_agent.infrastructure.synthetic_evaluation",
    "micro_model_agent.infrastructure.trace_rubrics",
    "micro_model_agent.infrastructure.trace_evaluation",
    "micro_model_agent.infrastructure.workspace_staged_rubrics",
    "micro_model_agent.infrastructure.workspace_staged_evaluation",
    "micro_model_agent.infrastructure.workspace_staged_review",
)
FLAT_TOOL_MODULES = ("micro_model_agent.infrastructure.tool_executor",)
FLAT_DATASET_MODULES = (
    "micro_model_agent.infrastructure.dataset_curation",
    "micro_model_agent.infrastructure.dataset_metadata",
    "micro_model_agent.infrastructure.dataset_prompting",
    "micro_model_agent.infrastructure.dataset_validation",
    "micro_model_agent.infrastructure.synthetic_data",
)
FLAT_TRACE_MODULES = (
    "micro_model_agent.infrastructure.trace_export",
    "micro_model_agent.infrastructure.trace_review",
)
FLAT_FACADE_MODULES = ("micro_model_agent.infrastructure.training_artifacts",)
RETIRED_CLI_COMPAT_NAMES = (
    "_load_dotenv",
    "_resolve_loop_model_options",
)
RETIRED_CLI_COMMON_NAMES = (
    "_base_model_from_adapter",
    "_resolve_loop_model_options",
)
RETIRED_MCP_SERVER_COMPAT_NAMES = (
    "_allowed_test_commands",
    "_allowed_tool_names",
    "_base_model_from_adapter",
    "_model_provider",
    "_path_from_user_input",
    "_required_tool_names",
    "_resolve_model_settings",
    "_run_profile_settings",
    "_string_config_value",
    "_tool_prompt_schemas",
    "_workflow_trace_store",
)
RETIRED_MCP_POLICY_PATHS = {
    PACKAGE_ROOT / "interfaces" / "mcp" / "policy" / "patch_policy.py",
    PACKAGE_ROOT / "interfaces" / "mcp" / "policy" / "tool_names.py",
}
RETIRED_MCP_TRACE_HELPER_NAMES = (
    "comparison_trace_store",
    "trace_dir",
    "workflow_trace_store",
)
PERSISTENCE_RUNTIME_HELPER_NAMES = (
    "append_comparison_trace_event",
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
)
MODEL_RUNTIME_HELPER_NAMES = (
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
)
INFRASTRUCTURE_RUNTIME_FACTORY_NAMES = (
    "build_static_coding_workflow",
    "build_dataset_export_workflow",
    "build_dataset_merge_workflow",
    "build_dataset_relabel_workflow",
    "build_dataset_synthesis_workflow",
    "build_dataset_validation_workflow",
    "build_evaluation_comparison_workflow",
    "build_jsonl_dataset_example_store",
    "build_promotion_gate_workflow",
    "build_promotion_list_workflow",
    "build_promotion_package_ollama_workflow",
    "build_promotion_record_workflow",
    "build_promotion_select_workflow",
    "build_synthetic_evaluation_workflow",
    "build_synthetic_training_workflow",
    "build_trace_dataset_export_workflow",
    "build_trace_evaluation_workflow",
    "build_trace_review_workflow",
    "build_workspace_staged_evaluation_workflow",
    "build_workspace_staged_review_workflow",
    "default_evaluation_available_tools",
    "initialize_local_repository",
    "local_repository_initialized",
    "write_local_repository_index",
)
MODEL_PROVIDER_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "fake_model_provider.py",
    PACKAGE_ROOT / "infrastructure" / "ollama_model_provider.py",
    PACKAGE_ROOT / "infrastructure" / "transformers_model_provider.py",
}
REPOSITORY_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "local_index.py",
    PACKAGE_ROOT / "infrastructure" / "local_retrieval.py",
    PACKAGE_ROOT / "infrastructure" / "repository_metadata.py",
    PACKAGE_ROOT / "infrastructure" / "repository_paths.py",
}
PERSISTENCE_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "comparison_trace.py",
    PACKAGE_ROOT / "infrastructure" / "dataset_store.py",
    PACKAGE_ROOT / "infrastructure" / "trace_store.py",
    PACKAGE_ROOT / "infrastructure" / "training_records.py",
    PACKAGE_ROOT / "infrastructure" / "workspace_registry.py",
}
TRAINING_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "local_finetuning.py",
    PACKAGE_ROOT / "infrastructure" / "ollama_packaging.py",
}
TRAINING_ARTIFACTS_FACADE_PATH = PACKAGE_ROOT / "infrastructure" / "training_artifacts.py"
PROMOTION_SHIM_PATHS = {PACKAGE_ROOT / "infrastructure" / "promotion_gate.py"}
EVALUATION_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "artifact_evaluation.py",
    PACKAGE_ROOT / "infrastructure" / "evaluation_comparison.py",
    PACKAGE_ROOT / "infrastructure" / "evaluation_reports.py",
    PACKAGE_ROOT / "infrastructure" / "evaluation_response_parsing.py",
    PACKAGE_ROOT / "infrastructure" / "synthetic_rubrics.py",
    PACKAGE_ROOT / "infrastructure" / "synthetic_evaluation.py",
    PACKAGE_ROOT / "infrastructure" / "trace_rubrics.py",
    PACKAGE_ROOT / "infrastructure" / "trace_evaluation.py",
    PACKAGE_ROOT / "infrastructure" / "workspace_staged_rubrics.py",
    PACKAGE_ROOT / "infrastructure" / "workspace_staged_evaluation.py",
    PACKAGE_ROOT / "infrastructure" / "workspace_staged_review.py",
}
TOOL_SHIM_PATHS = {PACKAGE_ROOT / "infrastructure" / "tool_executor.py"}
DATASET_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "dataset_curation.py",
    PACKAGE_ROOT / "infrastructure" / "dataset_metadata.py",
    PACKAGE_ROOT / "infrastructure" / "dataset_prompting.py",
    PACKAGE_ROOT / "infrastructure" / "dataset_validation.py",
    PACKAGE_ROOT / "infrastructure" / "synthetic_data.py",
}
TRACE_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "trace_export.py",
    PACKAGE_ROOT / "infrastructure" / "trace_review.py",
}
ALLOWED_TOP_LEVEL_INFRASTRUCTURE_MODULES = (
    MODEL_PROVIDER_SHIM_PATHS
    | REPOSITORY_SHIM_PATHS
    | PERSISTENCE_SHIM_PATHS
    | TRAINING_SHIM_PATHS
    | PROMOTION_SHIM_PATHS
    | EVALUATION_SHIM_PATHS
    | TOOL_SHIM_PATHS
    | DATASET_SHIM_PATHS
    | TRACE_SHIM_PATHS
    | {TRAINING_ARTIFACTS_FACADE_PATH, PACKAGE_ROOT / "infrastructure" / "composition.py"}
)
RUNTIME_MODULE_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "agents" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "datasets" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "evaluation" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "models" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "persistence" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "promotion" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "repositories" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "tools" / "runtime.py",
    PACKAGE_ROOT / "infrastructure" / "training" / "runtime.py",
}
APPLICATION_USE_CASE_FACADES = (
    (
        PACKAGE_ROOT / "application" / "agent" / "__init__.py",
        "micro_model_agent.application.agent.workflows",
    ),
    (
        PACKAGE_ROOT / "application" / "agent_workflows.py",
        "micro_model_agent.application.agent",
    ),
    (
        PACKAGE_ROOT / "application" / "datasets" / "__init__.py",
        "micro_model_agent.application.datasets.workflows",
    ),
    (
        PACKAGE_ROOT / "application" / "evaluation" / "__init__.py",
        "micro_model_agent.application.evaluation.workflows",
    ),
    (
        PACKAGE_ROOT / "application" / "evaluation_workflows.py",
        "micro_model_agent.application.evaluation",
    ),
    (
        PACKAGE_ROOT / "application" / "promotion" / "__init__.py",
        "micro_model_agent.application.promotion.workflows",
    ),
    (
        PACKAGE_ROOT / "application" / "ports" / "__init__.py",
        "micro_model_agent.application.ports.contracts",
    ),
    (
        PACKAGE_ROOT / "application" / "training" / "__init__.py",
        "micro_model_agent.application.training.workflows",
    ),
    (
        PACKAGE_ROOT / "application" / "tool_loop" / "__init__.py",
        "micro_model_agent.application.tool_loop.workflows",
    ),
    (
        PACKAGE_ROOT / "application" / "workflows.py",
        "micro_model_agent.application.agent",
    ),
)
APPLICATION_FACADE_MODULES = (
    "micro_model_agent.application.agent",
    "micro_model_agent.application.agent_workflows",
    "micro_model_agent.application.datasets",
    "micro_model_agent.application.evaluation",
    "micro_model_agent.application.evaluation_synthetic_rubric",
    "micro_model_agent.application.evaluation_trace_rubric",
    "micro_model_agent.application.evaluation_workspace_staged_rubric",
    "micro_model_agent.application.evaluation_workflows",
    "micro_model_agent.application.ports",
    "micro_model_agent.application.promotion",
    "micro_model_agent.application.tool_loop",
    "micro_model_agent.application.training",
    "micro_model_agent.application.workflows",
)
APPLICATION_RUBRIC_PATHS = {
    PACKAGE_ROOT / "application" / "evaluation_rubrics" / "synthetic.py",
    PACKAGE_ROOT / "application" / "evaluation_rubrics" / "trace.py",
    PACKAGE_ROOT / "application" / "evaluation_rubrics" / "workspace_staged.py",
}
APPLICATION_RUBRIC_SHIM_PATHS = {
    PACKAGE_ROOT / "application" / "evaluation_synthetic_rubric.py",
    PACKAGE_ROOT / "application" / "evaluation_trace_rubric.py",
    PACKAGE_ROOT / "application" / "evaluation_workspace_staged_rubric.py",
}
APPLICATION_FACADE_PATHS = {
    path for path, _owner_prefix in APPLICATION_USE_CASE_FACADES
} | APPLICATION_RUBRIC_SHIM_PATHS
INFRASTRUCTURE_RUBRIC_SHIM_PATHS = {
    PACKAGE_ROOT / "infrastructure" / "evaluation" / "synthetic_rubric.py",
    PACKAGE_ROOT / "infrastructure" / "evaluation" / "trace_rubric.py",
    PACKAGE_ROOT / "infrastructure" / "evaluation" / "workspace_staged_rubric.py",
}


def _production_modules(package: str) -> list[Path]:
    package_path = PACKAGE_ROOT / package
    return sorted(
        path
        for path in package_path.rglob("*.py")
        if not path.name.startswith("test_") and path.name != "__init__.py"
    )


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.partition(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _top_level_definition_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _all_export_values(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        assigns_all = any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
        if not assigns_all:
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            raise AssertionError(f"{path.relative_to(PACKAGE_ROOT)} has dynamic __all__")
        exports: list[str] = []
        for element in node.value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                raise AssertionError(f"{path.relative_to(PACKAGE_ROOT)} has dynamic __all__")
            exports.append(element.value)
        return exports
    raise AssertionError(f"{path.relative_to(PACKAGE_ROOT)} does not define __all__")


def _all_exports(path: Path) -> set[str]:
    return set(_all_export_values(path))


def test_application_layer_does_not_import_outer_layers() -> None:
    violations: list[str] = []
    for path in _production_modules("application"):
        for imported in _imports(path):
            if imported.startswith(APPLICATION_BANNED_PREFIXES) or imported in (
                *APPLICATION_BANNED_EXTERNALS,
            ):
                relative_path = path.relative_to(PACKAGE_ROOT)
                violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_domain_layer_has_no_project_or_framework_imports() -> None:
    violations: list[str] = []
    for path in _production_modules("domain"):
        for imported in _imports(path):
            if imported.startswith(PROJECT_IMPORT_PREFIX) or imported in DOMAIN_BANNED_EXTERNALS:
                relative_path = path.relative_to(PACKAGE_ROOT)
                violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_cli_package_exports_only_public_app() -> None:
    path = PACKAGE_ROOT / "interfaces" / "cli" / "__init__.py"

    assert _all_exports(path) == {"app"}
    assert set(RETIRED_CLI_COMPAT_NAMES).isdisjoint(_top_level_names(path))


def test_cli_common_does_not_own_runtime_resolution_helpers() -> None:
    path = PACKAGE_ROOT / "interfaces" / "cli" / "common.py"

    assert set(RETIRED_CLI_COMMON_NAMES).isdisjoint(_top_level_names(path))


def test_mcp_server_compatibility_module_does_not_export_private_helpers() -> None:
    path = PACKAGE_ROOT / "interfaces" / "mcp_server.py"
    exports = _all_exports(path)

    assert all(not name.startswith("_") for name in exports)
    assert set(RETIRED_MCP_SERVER_COMPAT_NAMES).isdisjoint(_top_level_names(path))


def test_mcp_policy_runtime_modules_stay_retired() -> None:
    assert [path for path in RETIRED_MCP_POLICY_PATHS if path.exists()] == []


def test_mcp_trace_module_does_not_own_store_factories() -> None:
    path = PACKAGE_ROOT / "interfaces" / "mcp" / "traces.py"

    assert set(RETIRED_MCP_TRACE_HELPER_NAMES).isdisjoint(_top_level_names(path))


def test_persistence_runtime_helpers_are_not_defined_in_composition() -> None:
    path = PACKAGE_ROOT / "infrastructure" / "composition.py"

    assert set(PERSISTENCE_RUNTIME_HELPER_NAMES).isdisjoint(
        _top_level_definition_names(path)
    )


def test_model_runtime_helpers_are_not_defined_in_composition() -> None:
    path = PACKAGE_ROOT / "infrastructure" / "composition.py"

    assert set(MODEL_RUNTIME_HELPER_NAMES).isdisjoint(_top_level_definition_names(path))


def test_infrastructure_runtime_factories_are_not_defined_in_composition() -> None:
    path = PACKAGE_ROOT / "infrastructure" / "composition.py"

    assert set(INFRASTRUCTURE_RUNTIME_FACTORY_NAMES).isdisjoint(
        _top_level_definition_names(path)
    )


def test_composition_facade_exports_are_explicit_and_public() -> None:
    path = PACKAGE_ROOT / "infrastructure" / "composition.py"
    exports = _all_exports(path)

    assert all(not name.startswith("_") for name in exports)
    assert exports.issubset(_top_level_names(path))


def test_runtime_modules_define_explicit_sorted_public_exports() -> None:
    violations: list[str] = []
    for path in RUNTIME_MODULE_PATHS:
        violations.extend(_public_export_violations(path, require_sorted=True))

    assert violations == []


def test_interface_modules_do_not_import_concrete_agents() -> None:
    violations: list[str] = []
    for path in _production_modules("interfaces"):
        for imported in _imports(path):
            if imported.startswith(INTERFACE_BANNED_PREFIXES):
                relative_path = path.relative_to(PACKAGE_ROOT)
                violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_interface_modules_do_not_import_low_level_concrete_adapters() -> None:
    violations: list[str] = []
    for path in _production_modules("interfaces"):
        for imported in _imports(path):
            if imported.startswith(INTERFACE_BANNED_CONCRETE_ADAPTER_PREFIXES):
                relative_path = path.relative_to(PACKAGE_ROOT)
                violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_interface_modules_import_infrastructure_only_through_composition() -> None:
    violations: list[str] = []
    for path in _production_modules("interfaces"):
        for imported in _imports(path):
            if imported.startswith("micro_model_agent.infrastructure") and not imported.startswith(
                INTERFACE_ALLOWED_INFRASTRUCTURE_PREFIXES
            ):
                relative_path = path.relative_to(PACKAGE_ROOT)
                violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_pure_rubrics_do_not_import_concrete_infrastructure() -> None:
    violations: list[str] = []
    for path in APPLICATION_RUBRIC_PATHS:
        violations.extend(
            f"{path.relative_to(PACKAGE_ROOT)}: {imported}"
            for imported in _imports(path)
            if imported.startswith(PURE_RUBRIC_BANNED_PREFIXES)
            or imported in PURE_RUBRIC_BANNED_EXTERNALS
        )

    assert violations == []


def test_application_rubric_exports_are_explicit_sorted_and_public() -> None:
    violations: list[str] = []
    for path in APPLICATION_RUBRIC_PATHS:
        violations.extend(_public_export_violations(path, require_sorted=True))

    assert violations == []


def test_application_use_case_modules_are_compatibility_facades() -> None:
    violations: list[str] = []
    for path, owner_prefix in APPLICATION_USE_CASE_FACADES:
        violations.extend(
            _shim_violations({path}, allowed_import_prefix=owner_prefix)
        )

    assert violations == []


def test_flat_application_rubric_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        APPLICATION_RUBRIC_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.application.evaluation_rubrics",
    )

    assert violations == []


def test_internal_production_imports_use_application_owner_modules() -> None:
    violations: list[str] = []
    for package in ("agents", "application", "infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in APPLICATION_FACADE_PATHS:
                continue
            for imported in _imports(path):
                if imported in APPLICATION_FACADE_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_infrastructure_rubric_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        INFRASTRUCTURE_RUBRIC_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.application",
    )

    assert violations == []


def test_internal_production_imports_use_model_provider_package() -> None:
    violations: list[str] = []
    for package in ("agents", "infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in MODEL_PROVIDER_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_MODEL_PROVIDER_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_repository_package() -> None:
    violations: list[str] = []
    for package in ("infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in REPOSITORY_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_REPOSITORY_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_persistence_package() -> None:
    violations: list[str] = []
    for package in ("agents", "infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in PERSISTENCE_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_PERSISTENCE_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_training_package() -> None:
    violations: list[str] = []
    for package in ("infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in TRAINING_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_TRAINING_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_promotion_package() -> None:
    violations: list[str] = []
    for package in ("infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in PROMOTION_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_PROMOTION_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_evaluation_package() -> None:
    violations: list[str] = []
    for package in ("infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in EVALUATION_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_EVALUATION_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_tool_package() -> None:
    violations: list[str] = []
    for package in ("agents", "infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in TOOL_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_TOOL_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_dataset_package() -> None:
    violations: list[str] = []
    for package in ("infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in DATASET_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_DATASET_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_use_trace_package() -> None:
    violations: list[str] = []
    for package in ("infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path in TRACE_SHIM_PATHS:
                continue
            for imported in _imports(path):
                if imported in FLAT_TRACE_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_internal_production_imports_avoid_flat_facade_modules() -> None:
    violations: list[str] = []
    for package in ("infrastructure", "interfaces"):
        for path in _production_modules(package):
            if path == TRAINING_ARTIFACTS_FACADE_PATH:
                continue
            for imported in _imports(path):
                if imported in FLAT_FACADE_MODULES:
                    relative_path = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_top_level_infrastructure_modules_are_composition_or_compatibility() -> None:
    top_level_modules = {
        path
        for path in (PACKAGE_ROOT / "infrastructure").glob("*.py")
        if not path.name.startswith("test_") and path.name != "__init__.py"
    }

    assert top_level_modules == ALLOWED_TOP_LEVEL_INFRASTRUCTURE_MODULES


def test_flat_model_provider_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        MODEL_PROVIDER_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.models",
    )

    assert violations == []


def test_flat_repository_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        REPOSITORY_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.repositories",
    )

    assert violations == []


def test_flat_persistence_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        PERSISTENCE_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.persistence",
    )

    assert violations == []


def test_flat_training_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        TRAINING_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.training",
    )

    assert violations == []


def test_training_artifacts_module_is_a_compatibility_facade() -> None:
    violations = _shim_violations(
        {TRAINING_ARTIFACTS_FACADE_PATH},
        allowed_import_prefix=(
            "micro_model_agent.infrastructure.evaluation",
            "micro_model_agent.infrastructure.persistence",
            "micro_model_agent.infrastructure.promotion",
            "micro_model_agent.infrastructure.training",
        ),
    )

    assert violations == []


def test_flat_promotion_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        PROMOTION_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.promotion",
    )

    assert violations == []


def test_flat_evaluation_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        EVALUATION_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.evaluation",
    )

    assert violations == []


def test_flat_tool_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        TOOL_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.tools",
    )

    assert violations == []


def test_flat_dataset_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        DATASET_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.datasets",
    )

    assert violations == []


def test_flat_trace_modules_are_compatibility_shims() -> None:
    violations = _shim_violations(
        TRACE_SHIM_PATHS,
        allowed_import_prefix="micro_model_agent.infrastructure.traces",
    )

    assert violations == []


def _shim_violations(
    paths: set[Path],
    *,
    allowed_import_prefix: str | tuple[str, ...],
) -> list[str]:
    violations: list[str] = []
    for path in paths:
        violations.extend(_public_export_violations(path))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = (
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for name in definitions:
            violations.append(f"{path.relative_to(PACKAGE_ROOT)} defines {name}")
        for imported in _imports(path):
            if not imported.startswith(allowed_import_prefix):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)} imports {imported}")
    return violations


def _public_export_violations(path: Path, *, require_sorted: bool = False) -> list[str]:
    violations: list[str] = []
    try:
        exports = _all_export_values(path)
    except AssertionError as exc:
        return [str(exc)]

    relative_path = path.relative_to(PACKAGE_ROOT)
    if not exports:
        violations.append(f"{relative_path} exports nothing")
    if any(name.startswith("_") for name in exports):
        violations.append(f"{relative_path} exports private names")
    if len(exports) != len(set(exports)):
        violations.append(f"{relative_path} has duplicate __all__ names")
    if require_sorted and exports != sorted(exports):
        violations.append(f"{relative_path} has unsorted __all__")
    missing_exports = set(exports) - _top_level_names(path)
    if missing_exports:
        violations.append(f"{relative_path} exports missing names: {sorted(missing_exports)}")
    return violations
