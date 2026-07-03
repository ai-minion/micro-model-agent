"""Catalog metadata for built-in tools.

The catalog is the single place that maps public tool names to descriptions and
Pydantic argument contracts. Agents use it to build model prompts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from micro_model_agent.repository_ops.infrastructure.contracts import (
    GitDiffRequest,
    RepoReadRequest,
    RepoSearchRequest,
    RepoWriteFilesRequest,
    RepoWritePatchRequest,
    SemanticSearchRequest,
    TestRunRequest,
)


@dataclass(frozen=True, slots=True)
class BuiltinToolSpec:
    """Prompt and validation metadata for a built-in tool."""

    name: str
    description: str
    argument_contract: type[BaseModel]


BUILTIN_TOOL_SPECS: dict[str, BuiltinToolSpec] = {
    "repo.search": BuiltinToolSpec(
        name="repo.search",
        description="Find repository files by text, glob, or Python symbol name.",
        argument_contract=RepoSearchRequest,
    ),
    "repo.read": BuiltinToolSpec(
        name="repo.read",
        description="Read one or more repository-relative text files, optionally by line range.",
        argument_contract=RepoReadRequest,
    ),
    "repo.semantic_search": BuiltinToolSpec(
        name="repo.semantic_search",
        description="Retrieve code or documentation context using structured lexical search.",
        argument_contract=SemanticSearchRequest,
    ),
    "repo.write_patch": BuiltinToolSpec(
        name="repo.write_patch",
        description="Preview or apply a unified diff constrained to repository files.",
        argument_contract=RepoWritePatchRequest,
    ),
    "repo.write_files": BuiltinToolSpec(
        name="repo.write_files",
        description=(
            "Create or replace repository-relative text files from explicit path/content "
            "entries. Prefer this for greenfield scaffolds and new files."
        ),
        argument_contract=RepoWriteFilesRequest,
    ),
    "test.run": BuiltinToolSpec(
        name="test.run",
        description="Run one allowlisted verification command by command name.",
        argument_contract=TestRunRequest,
    ),
    "git.diff": BuiltinToolSpec(
        name="git.diff",
        description="Inspect the current working tree diff without mutating files.",
        argument_contract=GitDiffRequest,
    ),
}

# Fast lookup from tool name to its request model, used by validation/export code.
TOOL_ARGUMENT_CONTRACTS: dict[str, type[BaseModel]] = {
    name: spec.argument_contract for name, spec in BUILTIN_TOOL_SPECS.items()
}


def builtin_tool_prompt_schemas(tool_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Return JSON-serializable tool descriptions and argument schemas for prompting."""

    selected_names = list(tool_names) if tool_names is not None else list(BUILTIN_TOOL_SPECS)
    schemas: dict[str, Any] = {}
    for name in selected_names:
        spec = BUILTIN_TOOL_SPECS.get(name)
        if spec is None:
            # Unknown names are ignored here because callers may filter a larger
            # user-provided list before exposing schemas to the model.
            continue
        schemas[name] = {
            "description": spec.description,
            "arguments_schema": spec.argument_contract.model_json_schema(),
        }
    return schemas
