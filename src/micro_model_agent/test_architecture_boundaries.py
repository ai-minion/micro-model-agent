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
