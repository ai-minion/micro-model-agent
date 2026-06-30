"""Portable workflow policy helpers for ToolLoopAgent."""

from __future__ import annotations

from typing import Any

from micro_model_agent.application.tool_loop import ToolLoopAgentTask
from micro_model_agent.domain.contracts import ToolCall, WorkflowStep


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
        and step.tool_call.tool_name == "repo.write_files"
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
    """Return true when repo.write_files repeats a prior successful file set."""

    if tool_call.tool_name != "repo.write_files":
        return False
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

    failed_patch_writes = [
        step
        for step in steps
        if step.tool_call is not None
        and step.tool_call.tool_name == "repo.write_patch"
        and step.tool_result is not None
        and not step.tool_result.ok
        and step.tool_result.error == "tool argument validation failed"
    ]
    if failed_patch_writes and "repo.write_files" in task.available_tools:
        hints.append(
            "A repo.write_patch call failed validation. For new files and scaffolds, "
            "use repo.write_files with explicit path/content entries instead of "
            "hand-authoring unified diffs."
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
            "test.run before final_response."
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
            "Return final_response, or run a distinct verification tool if one is available."
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
