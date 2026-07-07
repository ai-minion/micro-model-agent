"""Portable workflow policy helpers for ToolLoopAgent."""

from __future__ import annotations

from typing import Any

from micro_model_agent.execution.application.tool_loop import ToolLoopAgentTask
from micro_model_agent.execution.domain.value_objects import ToolCall, WorkflowStep


def tool_budget_exhausted(
    task: ToolLoopAgentTask,
    tool_calls_made: int,
    steps: list[WorkflowStep],
) -> bool:
    """Return true when the task has used all allowed tool calls."""

    return (
        task.max_tool_calls is not None
        and tool_calls_made >= task.max_tool_calls
        and not missing_required_tools(task, steps)
    )


def discovery_sufficient_for_final_response(
    task: ToolLoopAgentTask,
    steps: list[WorkflowStep],
) -> bool:
    """Return true when a dry-run creation/proposal task has enough discovery."""

    return (
        looks_like_creation_task(task)
        and looks_like_dry_run_or_proposal_task(task)
        and not missing_required_tools(task, steps)
        and any(is_successful_search_step(step) for step in steps)
    )


def should_allow_extra_finalization_turn(
    task: ToolLoopAgentTask,
    tool_calls_made: int,
    steps: list[WorkflowStep],
) -> bool:
    """Allow exactly one final-answer turn after a useful last action."""

    return (
        tool_budget_exhausted(task, tool_calls_made, steps)
        or (bool(steps) and steps[-1].tool_result is not None)
        or is_duplicate_write_block_step(steps[-1] if steps else None)
    )


def is_duplicate_write_block_step(step: WorkflowStep | None) -> bool:
    """Return whether a step blocked a repeated write and should now finalize."""

    return (
        step is not None
        and step.tool_call is not None
        and step.tool_call.tool_name in {"repo.write_files", "repo.write_patch"}
        and step.output.get("error") == "duplicate_successful_write"
    )


def has_unresolved_failed_tool_step(steps: list[WorkflowStep]) -> bool:
    """Check whether a tool failure still represents the final task state."""

    last_success_by_tool = {
        step.tool_call.tool_name: index
        for index, step in enumerate(steps)
        if step.tool_call is not None and step.tool_result is not None and step.tool_result.ok
    }
    latest_successful_write = max(
        (
            index
            for index, step in enumerate(steps)
            if step.tool_call is not None
            and step.tool_call.tool_name in {"repo.write_patch", "repo.write_files"}
            and step.tool_result is not None
            and step.tool_result.ok
        ),
        default=None,
    )
    for index, step in enumerate(steps):
        if (
            step.tool_call is None
            or step.tool_result is None
            or step.tool_result.ok
            or is_repaired_tool_failure(
                step,
                step_index=index,
                last_success_by_tool=last_success_by_tool,
                latest_successful_write=latest_successful_write,
            )
        ):
            continue
        return True
    return False


def is_repaired_tool_failure(
    step: WorkflowStep,
    *,
    step_index: int,
    last_success_by_tool: dict[str, int],
    latest_successful_write: int | None,
) -> bool:
    """Return true when a later action supersedes a failed tool attempt."""

    if step.tool_call is None:
        return False
    # Calls to tools that are simply not available, or that were blocked by
    # orchestration policy, are not real task failures.
    if step.tool_result is not None and isinstance(step.tool_result.error, str):
        if step.tool_result.error.startswith("tool is not available:"):
            return True
        if step.tool_result.error == "duplicate_read_or_search":
            return True
        if step.tool_result.error == "repeated_unavailable_tool":
            return True
    tool_name = step.tool_call.tool_name
    if last_success_by_tool.get(tool_name, -1) > step_index:
        return True
    return tool_name == "test.run" and (
        latest_successful_write is not None and latest_successful_write > step_index
    )


def is_duplicate_successful_write(
    tool_call: ToolCall,
    steps: list[WorkflowStep],
) -> bool:
    """Return true when repo.write_files or repo.write_patch repeats a prior successful write."""

    if tool_call.tool_name == "repo.write_files":
        requested_paths = write_file_paths(tool_call.arguments)
        if not requested_paths:
            return False
        for step in steps:
            if (
                step.tool_call is None
                or step.tool_result is None
                or not step.tool_result.ok
                or step.tool_call.tool_name != "repo.write_files"
            ):
                continue
            if write_file_paths(step.tool_call.arguments) == requested_paths:
                return True
        return False

    if tool_call.tool_name == "repo.write_patch":
        requested_paths = write_patch_paths(tool_call.arguments)
        requested_patch = str(tool_call.arguments.get("patch", ""))
        # Need at least one identity signal to avoid false positives.
        if not requested_paths and not requested_patch:
            return False
        for step in steps:
            if (
                step.tool_call is None
                or step.tool_result is None
                or not step.tool_result.ok
                or step.tool_call.tool_name != "repo.write_patch"
            ):
                continue
            # Block if applied=True (real write) or preview_complete=True
            # (forced dry_run via apply_patches=False).
            prev_output = step.tool_result.output or {}
            if not prev_output.get("applied", False) and not prev_output.get(
                "preview_complete", False
            ):
                continue
            # Same file paths alone are not enough: a repair loop may need a
            # second, distinct patch to the same file after verification. Only
            # block when the patch body is the same too.
            if str(step.tool_call.arguments.get("patch", "")) != requested_patch:
                continue
            if requested_paths:
                if write_patch_paths(step.tool_call.arguments) == requested_paths:
                    return True
                continue
            return True
        return False

    return False



def is_duplicate_read_or_search(
    tool_call: ToolCall,
    steps: list[WorkflowStep],
) -> bool:
    """Return true when repo.read or repo.search repeats an already-seen query.

    Only blocks if no write has occurred since the last identical call, so the
    model can legitimately re-read a file it just modified.
    """

    if tool_call.tool_name not in {"repo.read", "repo.search"}:
        return False

    # Stable key for this call (sorted JSON of arguments).
    import json as _json
    try:
        call_key = _json.dumps(tool_call.arguments, sort_keys=True)
    except (TypeError, ValueError):
        return False

    last_matching_index: int | None = None
    last_write_index: int | None = None

    for idx, step in enumerate(steps):
        if step.tool_call is None or step.tool_result is None or not step.tool_result.ok:
            continue
        if step.tool_call.tool_name in {"repo.write_patch", "repo.write_files"}:
            last_write_index = idx
        if step.tool_call.tool_name == tool_call.tool_name:
            try:
                prev_key = _json.dumps(step.tool_call.arguments, sort_keys=True)
            except (TypeError, ValueError):
                continue
            if prev_key == call_key:
                last_matching_index = idx

    if last_matching_index is None:
        return False
    # Allow re-read/re-search if a write happened after the previous matching call.
    return not (last_write_index is not None and last_write_index > last_matching_index)


def dedup_block_count_for_call(
    tool_call: ToolCall,
    steps: list[WorkflowStep],
) -> int:
    """Count how many times this exact (tool_name, args) call has already been
    dedup-blocked (error='duplicate_read_or_search') in the step history."""
    import json as _json

    try:
        call_key = _json.dumps(tool_call.arguments, sort_keys=True)
    except (TypeError, ValueError):
        return 0

    count = 0
    for step in steps:
        if step.tool_call is None or step.tool_result is None:
            continue
        if step.tool_result.error != "duplicate_read_or_search":
            continue
        if step.tool_call.tool_name != tool_call.tool_name:
            continue
        try:
            prev_key = _json.dumps(step.tool_call.arguments, sort_keys=True)
        except (TypeError, ValueError):
            continue
        if prev_key == call_key:
            count += 1
    return count


def is_repeated_unavailable_tool(
    tool_call: ToolCall,
    steps: list[WorkflowStep],
) -> bool:
    """Return true when the model calls a tool it has already been told is not available."""

    for step in steps:
        if (
            step.tool_call is not None
            and step.tool_call.tool_name == tool_call.tool_name
            and step.tool_result is not None
            and not step.tool_result.ok
            and isinstance(step.tool_result.error, str)
            and step.tool_result.error.startswith("tool is not available:")
        ):
            return True
    return False


def write_patch_paths(arguments: dict[str, Any]) -> tuple[str, ...]:
    """Extract a stable file-path tuple from repo.write_patch expected_changed_files."""

    files = arguments.get("expected_changed_files")
    if not isinstance(files, list):
        return ()
    paths = []
    for f in files:
        if not isinstance(f, str):
            return ()
        paths.append(f)
    return tuple(sorted(paths))


def write_file_paths(arguments: dict[str, Any]) -> tuple[str, ...]:
    """Extract a stable file-path tuple from repo.write_files arguments."""

    files = arguments.get("files")
    if not isinstance(files, list):
        return ()
    paths = []
    for file in files:
        if not isinstance(file, dict) or not isinstance(file.get("path"), str):
            return ()
        paths.append(file["path"])
    return tuple(sorted(paths))


def orchestration_hints(
    task: ToolLoopAgentTask,
    steps: list[WorkflowStep],
) -> list[str]:
    """Return compact process hints derived from repeated tool outcomes."""

    hints: list[str] = []
    if looks_like_creation_task(task):
        empty_searches = sum(1 for step in steps if is_empty_search_step(step))
        if empty_searches >= 2 and "repo.write_files" in task.available_tools:
            hints.append(
                "This appears to be a creation/scaffolding task in an empty workspace. "
                "Repeated search calls returned no matches; stop searching and use "
                "repo.write_files to create the requested files."
            )

    _PATCH_WRITE_ERRORS = {
        "tool argument validation failed",
        "patch does not contain changed file headers",
    }
    failed_patch_writes = [
        step
        for step in steps
        if step.tool_call is not None
        and step.tool_call.tool_name == "repo.write_patch"
        and step.tool_result is not None
        and not step.tool_result.ok
        and step.tool_result.error in _PATCH_WRITE_ERRORS
    ]
    if failed_patch_writes:
        if "repo.write_files" in task.available_tools:
            hints.append(
                "A repo.write_patch call failed. For new files and scaffolds, "
                "use repo.write_files with explicit path/content entries instead of "
                "hand-authoring unified diffs."
            )
        else:
            hints.append(
                "A repo.write_patch call failed. Ensure the patch starts with "
                "file headers: '--- a/path/to/file' and '+++ b/path/to/file' "
                "before the @@ hunk lines."
            )
    failed_write_files = [
        step
        for step in steps
        if step.tool_call is not None
        and step.tool_call.tool_name == "repo.write_files"
        and step.tool_result is not None
        and not step.tool_result.ok
        and step.tool_result.error == "tool argument validation failed"
    ]
    if failed_write_files and "repo.write_files" in task.available_tools:
        hints.append(
            'A repo.write_files call failed validation. Use arguments shaped exactly like '
            '{"files":[{"path":"README.md","content":"# Title\\n"}],'
            '"dry_run":false,"require_approval":false}; do not use a paths map, '
            "filename keys, or top-level content."
        )
    if missing_verification_after_write(task, steps):
        hints.append(
            "This task is about a failing test/import/verification issue. A file write "
            "has succeeded, but no test.run has passed after the latest write. Run "
            "test.run before answering."
        )
    successful_writes = [
        step
        for step in steps
        if step.tool_call is not None
        and step.tool_call.tool_name == "repo.write_files"
        and step.tool_result is not None
        and step.tool_result.ok
    ]
    if successful_writes:
        hints.append(
            "A repo.write_files call already succeeded. Do not repeat the same write. "
            "Answer the user, or run a distinct verification tool if one is available."
        )
    return hints


def looks_like_creation_task(task: ToolLoopAgentTask) -> bool:
    """Heuristic for greenfield creation/scaffolding requests."""

    text = f"{task.goal} {task.context}".lower()
    return any(
        term in text
        for term in (
            "create",
            "scaffold",
            "skeleton",
            "greenfield",
            "empty repository",
            "empty workspace",
            "new file",
            "new project",
        )
    )


def is_empty_search_step(step: WorkflowStep) -> bool:
    """Return true for successful search-style calls with zero results."""

    if step.tool_call is None or step.tool_result is None or not step.tool_result.ok:
        return False
    if step.tool_call.tool_name == "repo.search":
        matches = step.tool_result.output.get("matches")
        return isinstance(matches, list) and len(matches) == 0
    if step.tool_call.tool_name == "repo.semantic_search":
        results = step.tool_result.output.get("results")
        return isinstance(results, list) and len(results) == 0
    return False


def is_successful_search_step(step: WorkflowStep) -> bool:
    """Return true for any successful search-style call."""

    return (
        step.tool_call is not None
        and step.tool_result is not None
        and step.tool_result.ok
        and step.tool_call.tool_name in {"repo.search", "repo.semantic_search"}
    )


def missing_required_tools(
    task: ToolLoopAgentTask,
    steps: list[WorkflowStep],
) -> tuple[str, ...]:
    """List required tools that have not successfully appeared in the trace."""

    used_tools = {
        step.tool_call.tool_name
        for step in steps
        if step.tool_call is not None and step.tool_result is not None and step.tool_result.ok
    }
    return tuple(
        tool_name
        for tool_name in task.required_tools
        if not required_tool_satisfied(tool_name, used_tools)
    )


def required_tool_satisfied(required_tool: str, used_tools: set[str]) -> bool:
    """Treat successful repository writes as equivalent required write evidence."""

    if required_tool in {"repo.write_patch", "repo.write_files"}:
        return bool({"repo.write_patch", "repo.write_files"} & used_tools)
    return required_tool in used_tools


def missing_verification_after_write(
    task: ToolLoopAgentTask,
    steps: list[WorkflowStep],
) -> bool:
    """Require test.run after writes for explicit test/import repair tasks."""

    if "test.run" not in task.available_tools or not looks_like_verification_task(task):
        return False
    latest_write_index: int | None = None
    latest_passing_test_index: int | None = None
    for index, step in enumerate(steps):
        if step.tool_call is None or step.tool_result is None or not step.tool_result.ok:
            continue
        if step.tool_call.tool_name in {"repo.write_patch", "repo.write_files"}:
            latest_write_index = index
        if step.tool_call.tool_name == "test.run":
            output_ok = step.tool_result.output.get("ok")
            exit_code = step.tool_result.output.get("exit_code")
            if output_ok is True or exit_code == 0:
                latest_passing_test_index = index
    return latest_write_index is not None and (
        latest_passing_test_index is None or latest_passing_test_index < latest_write_index
    )


def looks_like_verification_task(task: ToolLoopAgentTask) -> bool:
    """Heuristic for tasks whose success depends on running verification."""

    text = f"{task.goal} {task.context}".lower()
    return any(
        term in text
        for term in (
            "pytest",
            "test failure",
            "tests fail",
            "test fails",
            "failing test",
            "failing import",
            "import error",
            "modulenotfounderror",
            "runs successfully",
            "verification",
        )
    )


def looks_like_dry_run_or_proposal_task(task: ToolLoopAgentTask) -> bool:
    """Heuristic for tasks that ask the loop to propose rather than write."""

    text = f"{task.goal} {task.context}".lower()
    return any(
        term in text
        for term in (
            "do not apply",
            "without applying",
            "do not change",
            "no changes",
            "no files changed",
            "propose",
            "proposal",
            "dry-run",
            "dry run",
        )
    )


def consecutive_tool_failure_count(steps: list[WorkflowStep]) -> int:
    """Count consecutive real tool failures at the tail of the step list.

    Only counts actual executor failures (tool argument validation failed,
    tool execution errors, etc.) — not policy rejections like dedup blocks or
    'tool is not available' which are handled separately.
    """
    real_failure_errors = {
        "tool argument validation failed",
    }
    count = 0
    for step in reversed(steps):
        if step.tool_result is None:
            break
        err = step.tool_result.error or ""
        if step.tool_result.ok:
            break
        if err in real_failure_errors or err.startswith("executor error"):
            count += 1
        else:
            # Policy rejection — stop counting
            break
    return count
