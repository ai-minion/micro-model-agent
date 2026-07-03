"""Tests for the built-in tool catalog."""

from __future__ import annotations

from micro_model_agent.repository_ops.infrastructure.catalog import (
    BUILTIN_TOOL_SPECS,
    TOOL_ARGUMENT_CONTRACTS,
    builtin_tool_prompt_schemas,
)

# ---------------------------------------------------------------------------
# BUILTIN_TOOL_SPECS / TOOL_ARGUMENT_CONTRACTS
# ---------------------------------------------------------------------------


def test_all_canonical_tool_names_present() -> None:
    expected = {
        "repo.search",
        "repo.read",
        "repo.semantic_search",
        "repo.write_patch",
        "repo.write_files",
        "test.run",
        "git.diff",
    }
    assert set(BUILTIN_TOOL_SPECS.keys()) == expected


def test_tool_argument_contracts_match_specs() -> None:
    assert set(TOOL_ARGUMENT_CONTRACTS.keys()) == set(BUILTIN_TOOL_SPECS.keys())
    for name, contract in TOOL_ARGUMENT_CONTRACTS.items():
        assert BUILTIN_TOOL_SPECS[name].argument_contract is contract


def test_each_spec_has_description() -> None:
    for name, spec in BUILTIN_TOOL_SPECS.items():
        assert isinstance(spec.description, str) and len(spec.description) > 0, \
            f"{name} has no description"


def test_each_spec_has_argument_contract() -> None:
    for name, spec in BUILTIN_TOOL_SPECS.items():
        assert spec.argument_contract is not None, f"{name} has no argument_contract"


# ---------------------------------------------------------------------------
# builtin_tool_prompt_schemas
# ---------------------------------------------------------------------------


def test_schemas_returns_all_tools_when_none() -> None:
    schemas = builtin_tool_prompt_schemas()
    assert set(schemas.keys()) == set(BUILTIN_TOOL_SPECS.keys())


def test_schemas_filtered_by_names() -> None:
    schemas = builtin_tool_prompt_schemas(["repo.read", "git.diff"])
    assert set(schemas.keys()) == {"repo.read", "git.diff"}


def test_schemas_ignores_unknown_names() -> None:
    schemas = builtin_tool_prompt_schemas(["repo.read", "unknown.tool"])
    assert "repo.read" in schemas
    assert "unknown.tool" not in schemas


def test_each_schema_has_description() -> None:
    schemas = builtin_tool_prompt_schemas()
    for name, schema in schemas.items():
        assert "description" in schema, f"{name} schema missing description"
        assert isinstance(schema["description"], str)


def test_each_schema_has_arguments_schema() -> None:
    schemas = builtin_tool_prompt_schemas()
    for name, schema in schemas.items():
        assert "arguments_schema" in schema, f"{name} schema missing arguments_schema"
        assert isinstance(schema["arguments_schema"], dict)


def test_schema_is_json_serializable() -> None:
    import json
    schemas = builtin_tool_prompt_schemas()
    json.dumps(schemas)  # should not raise


def test_empty_name_list_returns_empty_dict() -> None:
    schemas = builtin_tool_prompt_schemas([])
    assert schemas == {}


def test_repo_read_schema_has_expected_fields() -> None:
    schemas = builtin_tool_prompt_schemas(["repo.read"])
    schema = schemas["repo.read"]["arguments_schema"]
    # The repo.read contract has a 'files' property in its JSON schema
    properties = schema.get("properties", {})
    assert "files" in properties
