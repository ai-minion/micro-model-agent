"""Prompt-history compaction helpers for ToolLoopAgent."""

from __future__ import annotations

import json
from typing import Any

from micro_model_agent.execution.domain.value_objects import ToolResult, WorkflowStep


def tool_result_message(
    tool_result: ToolResult,
    *,
    max_prompt_chars: int,
) -> dict[str, Any]:
    """Convert a ToolResult into the compact transcript message format."""

    return {
        "role": "tool",
        "tool_call_id": str(tool_result.tool_call_id),
        "tool_name": tool_result.tool_name,
        "ok": tool_result.ok,
        "output": truncate_tool_output(tool_result.output, max_prompt_chars),
        "error": tool_result.error,
    }


def tool_history(
    steps: list[WorkflowStep],
    *,
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    """Return prior tool calls/results, compacting outputs near the prompt budget."""

    history: list[dict[str, Any]] = []
    for step in steps:
        if step.tool_call is None or step.tool_result is None:
            continue
        history.append(
            {
                "step": step.name,
                "tool_call_id": str(step.tool_call.id),
                "tool_name": step.tool_call.tool_name,
                "arguments": summarize_tool_arguments(
                    step.tool_call.tool_name,
                    step.tool_call.arguments,
                ),
                "ok": step.tool_result.ok,
                "output": step.tool_result.output,
                "error": step.tool_result.error,
            }
        )

    if not history:
        return []

    history = compact_path_tool_history(history)
    history_json = json.dumps(history, sort_keys=True)
    if max_prompt_chars < 1 or len(history_json) <= max_prompt_chars:
        return history

    compacted: list[dict[str, Any]] = []
    for entry in history:
        compacted.append(
            {
                **entry,
                "output": summarize_tool_output(entry["output"]),
            }
        )

    compacted_json = json.dumps(compacted, sort_keys=True)
    if len(compacted_json) <= max_prompt_chars:
        return compacted

    # Keep the most recent calls with full call metadata. Older calls are
    # summarized into a count so repetition is still visible under pressure.
    kept: list[dict[str, Any]] = []
    omitted = 0
    for entry in reversed(compacted):
        candidate = [*reversed(kept), entry]
        candidate_json = json.dumps(candidate, sort_keys=True)
        if len(candidate_json) <= max_prompt_chars or not kept:
            kept.append(entry)
        else:
            omitted += 1
    result = list(reversed(kept))
    if omitted:
        result.insert(
            0,
            {
                "compacted_history": True,
                "omitted_older_tool_calls": omitted,
            },
        )
    return result


def compact_path_tool_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only the latest read/write context for each repository path."""

    compacted_reversed: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for entry in reversed(history):
        paths = history_entry_paths(entry)
        if not paths:
            compacted_reversed.append(entry)
            continue

        unseen_paths = [path for path in paths if path not in seen_paths]
        if not unseen_paths:
            continue

        compacted_reversed.append(history_entry_with_paths(entry, unseen_paths))
        seen_paths.update(unseen_paths)
    return list(reversed(compacted_reversed))


def history_entry_paths(entry: dict[str, Any]) -> list[str]:
    """Return repository paths represented by one prompt history entry."""

    # Dedup-blocked entries contain no real content — don't let them displace
    # the actual read/write result that holds the file content.
    output = entry.get("output")
    if isinstance(output, dict) and output.get("duplicate"):
        return []
    if entry.get("error") == "duplicate_read_or_search":
        return []

    tool_name = entry.get("tool_name")
    arguments = entry.get("arguments")
    if not isinstance(arguments, dict):
        return []
    if tool_name == "repo.write_files":
        paths = arguments.get("paths")
        if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
            return list(paths)
    if tool_name == "repo.read":
        files = arguments.get("files")
        if isinstance(files, list):
            return [
                file["path"]
                for file in files
                if isinstance(file, dict) and isinstance(file.get("path"), str)
            ]
    return []


def history_entry_with_paths(
    entry: dict[str, Any],
    paths: list[str],
) -> dict[str, Any]:
    """Return a history entry narrowed to the requested path subset."""

    tool_name = entry.get("tool_name")
    allowed_paths = set(paths)
    narrowed = dict(entry)
    arguments = entry.get("arguments")
    if isinstance(arguments, dict):
        narrowed["arguments"] = history_arguments_with_paths(
            tool_name,
            arguments,
            allowed_paths,
        )
    output = entry.get("output")
    if isinstance(output, dict):
        narrowed["output"] = history_output_with_paths(output, allowed_paths)
    return narrowed


def history_arguments_with_paths(
    tool_name: object,
    arguments: dict[str, Any],
    allowed_paths: set[str],
) -> dict[str, Any]:
    narrowed = dict(arguments)
    if tool_name == "repo.write_files":
        narrowed["paths"] = [
            path
            for path in arguments.get("paths", [])
            if isinstance(path, str) and path in allowed_paths
        ]
        narrowed["file_count"] = len(narrowed["paths"])
    elif tool_name == "repo.read":
        files = arguments.get("files")
        if isinstance(files, list):
            narrowed["files"] = [
                file
                for file in files
                if isinstance(file, dict) and file.get("path") in allowed_paths
            ]
    return narrowed


def history_output_with_paths(
    output: dict[str, Any],
    allowed_paths: set[str],
) -> dict[str, Any]:
    narrowed = dict(output)
    files = output.get("files")
    if isinstance(files, list):
        narrowed["files"] = [
            file for file in files if isinstance(file, dict) and file.get("path") in allowed_paths
        ]
        if "file_count" in narrowed:
            narrowed["file_count"] = len(narrowed["files"])
    paths = output.get("paths")
    if isinstance(paths, list):
        narrowed["paths"] = [
            path for path in paths if isinstance(path, str) and path in allowed_paths
        ]
    changed_files = output.get("changed_files")
    if isinstance(changed_files, list):
        narrowed["changed_files"] = [
            path for path in changed_files if isinstance(path, str) and path in allowed_paths
        ]
    return narrowed


def summarize_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Keep prior tool arguments useful without inviting content copying."""

    if tool_name != "repo.write_files":
        return arguments
    paths = write_file_paths(arguments)
    return {
        "dry_run": arguments.get("dry_run"),
        "file_count": len(paths),
        "paths": list(paths),
    }


def summarize_tool_output(output: dict[str, Any]) -> dict[str, Any]:
    """Summarize bulky tool output while preserving decision-relevant facts."""

    summary: dict[str, Any] = {}
    if "matches" in output and isinstance(output["matches"], list):
        summary["match_count"] = len(output["matches"])
        summary["truncated"] = output.get("truncated", False)
    if "results" in output and isinstance(output["results"], list):
        summary["result_count"] = len(output["results"])
    if "files" in output and isinstance(output["files"], list):
        summary["file_count"] = len(output["files"])
        summary["paths"] = [
            item.get("path")
            for item in output["files"]
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        ][:20]
    if "changed_files" in output:
        summary["changed_files"] = output["changed_files"]
    if "errors" in output and output["errors"]:
        summary["errors"] = output["errors"]
    if "ok" in output:
        summary["ok"] = output["ok"]
    return summary or truncate_tool_output(output, 500)


def truncate_tool_output(
    output: dict[str, Any],
    max_prompt_chars: int,
) -> dict[str, Any]:
    """Shorten large tool outputs before they are placed back in the prompt."""

    output_json = json.dumps(output, sort_keys=True)
    if max_prompt_chars < 1 or len(output_json) <= max_prompt_chars:
        return output
    return {
        "truncated_for_prompt": True,
        "preview": output_json[:max_prompt_chars],
        "original_character_count": len(output_json),
    }


def write_file_paths(arguments: dict[str, Any]) -> tuple[str, ...]:
    paths = arguments.get("paths")
    if isinstance(paths, list):
        return tuple(path for path in paths if isinstance(path, str))
    files = arguments.get("files")
    if isinstance(files, list):
        return tuple(
            file["path"]
            for file in files
            if isinstance(file, dict) and isinstance(file.get("path"), str)
        )
    return ()
