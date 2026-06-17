"""Metadata helpers for dataset tool profiles.

Tool profiles are stored as plain JSON metadata so datasets, training runs, and
evaluation reports can be compared without adding a new domain abstraction yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from micro_model_agent.domain.datasets import DatasetExample

DEFAULT_TOOL_PROFILE_NAME = "coding-agent-v1"


def tool_profile_for_example(
    example: DatasetExample,
    *,
    default_available_tools: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return JSON-ready tool-profile metadata for one dataset example."""

    available_tools = _available_tools(example, default_available_tools)
    tools_used = _tools_used(example)
    profile_name = _metadata_profile_name(example)
    if profile_name is None:
        profile_name = DEFAULT_TOOL_PROFILE_NAME if default_available_tools else "custom"

    return {
        "name": profile_name,
        "tool_schema_version": example.tool_schema_version,
        "available_tools": list(available_tools),
        "tools_used": list(tools_used),
    }


def summarize_tool_profiles(
    examples: Iterable[DatasetExample],
    *,
    default_available_tools: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize tool-profile metadata across a dataset or evaluation suite."""

    profiles = [
        tool_profile_for_example(example, default_available_tools=default_available_tools)
        for example in examples
    ]
    available_tools = sorted(
        {
            tool
            for profile in profiles
            for tool in profile["available_tools"]
            if isinstance(tool, str)
        }
    )
    tools_used = sorted(
        {
            tool
            for profile in profiles
            for tool in profile["tools_used"]
            if isinstance(tool, str)
        }
    )
    schema_versions = sorted(
        {
            version
            for profile in profiles
            if isinstance((version := profile.get("tool_schema_version")), str)
        }
    )
    profile_counts: dict[str, int] = {}
    for profile in profiles:
        key = ",".join(profile["available_tools"]) or "unspecified"
        profile_counts[key] = profile_counts.get(key, 0) + 1

    return {
        "name": DEFAULT_TOOL_PROFILE_NAME if default_available_tools else "dataset",
        "example_count": len(profiles),
        "tool_schema_versions": schema_versions,
        "available_tools": available_tools,
        "tools_used": tools_used,
        "profile_counts": dict(sorted(profile_counts.items())),
    }


def metadata_with_tool_profile(
    example: DatasetExample,
    *,
    default_available_tools: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return example metadata with a computed tool_profile when absent."""

    metadata = dict(example.metadata)
    metadata.setdefault(
        "tool_profile",
        tool_profile_for_example(example, default_available_tools=default_available_tools),
    )
    return metadata


def _available_tools(
    example: DatasetExample,
    default_available_tools: Sequence[str] | None,
) -> tuple[str, ...]:
    input_tools = example.input.get("available_tools")
    if isinstance(input_tools, list) and all(isinstance(tool, str) for tool in input_tools):
        return _dedupe(input_tools)

    metadata_profile = example.metadata.get("tool_profile")
    if isinstance(metadata_profile, dict):
        metadata_tools = metadata_profile.get("available_tools")
        if isinstance(metadata_tools, list) and all(
            isinstance(tool, str) for tool in metadata_tools
        ):
            return _dedupe(metadata_tools)

    if default_available_tools is not None:
        return _dedupe(default_available_tools)
    return ()


def _tools_used(example: DatasetExample) -> tuple[str, ...]:
    tools: list[str] = []
    target_tool = example.target.get("tool_name")
    if isinstance(target_tool, str):
        tools.append(target_tool)

    history = example.input.get("tool_history")
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            tool_call = item.get("tool_call")
            if isinstance(tool_call, dict) and isinstance(tool_call.get("tool_name"), str):
                tools.append(tool_call["tool_name"])

    return _dedupe(tools)


def _metadata_profile_name(example: DatasetExample) -> str | None:
    metadata_profile = example.metadata.get("tool_profile")
    if not isinstance(metadata_profile, dict):
        return None
    name = metadata_profile.get("name")
    return name if isinstance(name, str) and name else None


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
