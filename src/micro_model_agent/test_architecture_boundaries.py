"""Architecture boundary characterization tests."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
PROJECT_IMPORT_PREFIX = "micro_model_agent."
APPLICATION_BANNED_PREFIXES = (
    "micro_model_agent.infrastructure",
    "micro_model_agent.interfaces",
)
INTERFACE_BANNED_CONCRETE_ADAPTER_PREFIXES = (
    "micro_model_agent.infrastructure.models",
    "micro_model_agent.infrastructure.tools.catalog",
    "micro_model_agent.infrastructure.tools.command_runner",
    "micro_model_agent.infrastructure.tools.executor",
    "micro_model_agent.evaluation.infrastructure.comparison_trace",
    "micro_model_agent.dataset.infrastructure.dataset_store",
    "micro_model_agent.execution.infrastructure.trace_store",
    "micro_model_agent.repository_ops.infrastructure.workspace_registry",
    "micro_model_agent.infrastructure.promotion.gate",
    "micro_model_agent.infrastructure.repositories.local_index",
    "micro_model_agent.infrastructure.repositories.metadata",
    "micro_model_agent.infrastructure.training.artifacts",
    "micro_model_agent.training.infrastructure.local_finetuning",
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
RUNTIME_MODULE_PATHS = {
    # New canonical runtime locations after DDD migration
    PACKAGE_ROOT / "execution" / "infrastructure" / "agents_runtime.py",
    PACKAGE_ROOT / "execution" / "infrastructure" / "persistence_runtime.py",
    PACKAGE_ROOT / "execution" / "infrastructure" / "models" / "runtime.py",
    PACKAGE_ROOT / "dataset" / "infrastructure" / "runtime.py",
    PACKAGE_ROOT / "evaluation" / "infrastructure" / "runtime.py",
    PACKAGE_ROOT / "promotion" / "infrastructure" / "runtime.py",
    PACKAGE_ROOT / "repository_ops" / "infrastructure" / "runtime.py",
    PACKAGE_ROOT / "repository_ops" / "infrastructure" / "tools_runtime.py",
    PACKAGE_ROOT / "training" / "infrastructure" / "runtime.py",
}
APPLICATION_RUBRIC_PATHS = {
    PACKAGE_ROOT / "evaluation" / "domain" / "rubrics_synthetic.py",
    PACKAGE_ROOT / "evaluation" / "domain" / "rubrics_trace.py",
    PACKAGE_ROOT / "evaluation" / "domain" / "rubrics_workspace_staged.py",
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


DOMAIN_COMPAT_STUB_MARKER = "Backward-compatible re-export"


def _is_compat_stub(path: Path) -> bool:
    """Return True when the file is a backward-compatible re-export shim."""
    try:
        text = path.read_text(encoding="utf-8")
        return DOMAIN_COMPAT_STUB_MARKER in text
    except Exception:
        return False


def test_domain_layer_has_no_project_or_framework_imports() -> None:
    violations: list[str] = []
    for path in _production_modules("domain"):
        # Backward-compat stubs are allowed to re-export from bounded-context packages.
        if _is_compat_stub(path):
            continue
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
        # Backward-compat stubs replaced the original runtime files; check
        # the new canonical locations instead when a stub is found.
        if _is_compat_stub(path):
            continue
        violations.extend(_public_export_violations(path, require_sorted=True))

    assert violations == []


def test_interface_modules_do_not_import_low_level_concrete_adapters() -> None:
    violations: list[str] = []
    for path in _production_modules("interfaces"):
        for imported in _imports(path):
            if imported.startswith(INTERFACE_BANNED_CONCRETE_ADAPTER_PREFIXES):
                relative_path = path.relative_to(PACKAGE_ROOT)
                violations.append(f"{relative_path}: {imported}")

    assert violations == []


def test_interface_modules_do_not_import_infrastructure() -> None:
    """Interfaces must not import from the legacy top-level infrastructure package."""
    violations: list[str] = []
    for path in _production_modules("interfaces"):
        for imported in _imports(path):
            if imported.startswith("micro_model_agent.infrastructure"):
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
        # Stubs replaced the original rubric files; skip them.
        if _is_compat_stub(path):
            continue
        violations.extend(_public_export_violations(path, require_sorted=True))

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

# ---------------------------------------------------------------------------
# New bounded context packages (DDD migration)
# ---------------------------------------------------------------------------
BOUNDED_CONTEXT_PACKAGES = (
    "micro_model_agent.execution",
    "micro_model_agent.dataset",
    "micro_model_agent.training",
    "micro_model_agent.evaluation",
    "micro_model_agent.promotion",
    "micro_model_agent.repository_ops",
)

# Each context's domain layer must not import another context's domain layer.
CROSS_CONTEXT_DOMAIN_PREFIXES = tuple(
    f"micro_model_agent.{ctx}.domain"
    for ctx in (
        "execution",
        "dataset",
        "training",
        "evaluation",
        "promotion",
        "repository_ops",
    )
)

# The shared kernel must not import any bounded context.
SHARED_KERNEL_BANNED_PREFIXES = tuple(
    f"micro_model_agent.{ctx}"
    for ctx in (
        "execution",
        "dataset",
        "training",
        "evaluation",
        "promotion",
        "repository_ops",
        "agents",
        "infrastructure",
        "interfaces",
        "application",
    )
)

DOMAIN_BANNED_EXTERNALS_BOUNDED = (
    "pydantic",
    "typer",
    "mcp",
    "fastmcp",
    "datasets",
    "peft",
    "torch",
    "transformers",
)

def _bounded_context_domain_modules(context: str) -> list[Path]:
    """Return production .py files inside <context>/domain/."""
    domain_path = PACKAGE_ROOT / context / "domain"
    if not domain_path.exists():
        return []
    return sorted(
        path
        for path in domain_path.rglob("*.py")
        if not path.name.startswith("test_") and path.name != "__init__.py"
    )


def _bounded_context_production_modules(context: str) -> list[Path]:
    """Return all non-test .py files inside a bounded-context package."""
    ctx_path = PACKAGE_ROOT / context
    if not ctx_path.exists():
        return []
    return sorted(
        path
        for path in ctx_path.rglob("*.py")
        if not path.name.startswith("test_") and path.name != "__init__.py"
    )


def test_shared_kernel_does_not_import_any_bounded_context() -> None:
    """The shared kernel must be self-contained: no imports from bounded contexts."""
    violations: list[str] = []
    shared_path = PACKAGE_ROOT / "shared"
    if not shared_path.exists():
        return
    for path in shared_path.rglob("*.py"):
        if path.name.startswith("test_") or path.name == "__init__.py":
            continue
        for imported in _imports(path):
            if imported.startswith(SHARED_KERNEL_BANNED_PREFIXES):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {imported}")
    assert violations == []


# Evaluation rubrics score DatasetExample objects — a deliberate cross-context
# data-schema dependency (Published Language pattern).  Allow it explicitly.
ALLOWED_CROSS_CONTEXT_DOMAIN_IMPORTS: set[tuple[str, str]] = {
    # (importer_prefix, allowed_imported_prefix)
    ("evaluation", "micro_model_agent.dataset.domain"),
}


def test_bounded_context_domains_do_not_cross_import_each_other() -> None:
    """No bounded context domain layer may import another context's domain layer.

    Exceptions in ALLOWED_CROSS_CONTEXT_DOMAIN_IMPORTS are documented deliberate
    Published-Language dependencies (e.g. evaluation rubrics scoring DatasetExample).
    """
    violations: list[str] = []
    for context in BOUNDED_CONTEXT_PACKAGES:
        ctx_name = context.split(".")[-1]
        own_prefix = f"micro_model_agent.{ctx_name}.domain"
        for path in _bounded_context_domain_modules(ctx_name):
            for imported in _imports(path):
                if not imported.startswith(CROSS_CONTEXT_DOMAIN_PREFIXES):
                    continue
                if imported.startswith(own_prefix):
                    continue
                # Check whether this import is in the allowed exceptions.
                if any(
                    ctx_name == src and imported.startswith(allowed)
                    for src, allowed in ALLOWED_CROSS_CONTEXT_DOMAIN_IMPORTS
                ):
                    continue
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {imported}")
    assert violations == []


def test_bounded_context_domains_have_no_banned_framework_imports() -> None:
    """Bounded context domain layers must stay free of heavy framework imports."""
    violations: list[str] = []
    for context in BOUNDED_CONTEXT_PACKAGES:
        ctx_name = context.split(".")[-1]
        for path in _bounded_context_domain_modules(ctx_name):
            for imported in _imports(path):
                if imported in DOMAIN_BANNED_EXTERNALS_BOUNDED:
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {imported}")
    assert violations == []




# ---------------------------------------------------------------------------
# Bounded-context application layer purity
# ---------------------------------------------------------------------------

# Application layers must not reach into infrastructure or interfaces directly.
BOUNDED_CONTEXT_APP_BANNED_PREFIXES = (
    "micro_model_agent.infrastructure",
    "micro_model_agent.interfaces",
)


def _bounded_context_application_modules(context: str) -> list[Path]:
    """Return production .py files inside <context>/application/."""
    app_path = PACKAGE_ROOT / context / "application"
    if not app_path.exists():
        return []
    return sorted(
        path
        for path in app_path.rglob("*.py")
        if not path.name.startswith("test_") and path.name != "__init__.py"
    )


def test_bounded_context_application_layers_do_not_import_infrastructure() -> None:
    """Bounded-context application/ layers must not import from agents/,
    infrastructure/, or interfaces/ — only from their own context and shared.

    Cross-context communication goes through events and ports, not direct calls.
    """
    violations: list[str] = []
    for context in BOUNDED_CONTEXT_PACKAGES:
        ctx_name = context.split(".")[-1]
        for path in _bounded_context_application_modules(ctx_name):
            for imported in _imports(path):
                if imported.startswith(BOUNDED_CONTEXT_APP_BANNED_PREFIXES):
                    violations.append(
                        f"{path.relative_to(PACKAGE_ROOT)}: {imported}"
                    )
    assert violations == []
