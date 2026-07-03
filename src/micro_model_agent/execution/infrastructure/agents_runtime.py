"""Runtime composition helpers for concrete agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from micro_model_agent.agents.coding_agent import CodingAgent
from micro_model_agent.execution.application.workflows import RunAgentWorkflow
from micro_model_agent.dataset.application.ports import DatasetExampleStore
from micro_model_agent.execution.infrastructure.models.fake import StaticModelProvider
from micro_model_agent.execution.infrastructure.persistence_runtime import workflow_trace_store
from micro_model_agent.repository_ops.infrastructure.command_runner import AllowedTestCommand
from micro_model_agent.repository_ops.infrastructure.tools_runtime import build_builtin_tool_executor
from micro_model_agent.execution.infrastructure.composition import build_coding_workflow  # noqa: F401

__all__ = ["build_static_coding_workflow"]


def build_static_coding_workflow(
    *,
    repository_root: str | Path,
    patch: str,
    allowed_commands: Mapping[str, AllowedTestCommand | Sequence[str]],
    dataset_store: DatasetExampleStore | None = None,
) -> RunAgentWorkflow:
    """Build the fixed coding workflow. Prefer ``build_coding_workflow`` for new code."""

    return build_coding_workflow(
        repository_root=repository_root,
        patch=patch,
        allowed_commands=allowed_commands,
        dataset_store=dataset_store,
    )
