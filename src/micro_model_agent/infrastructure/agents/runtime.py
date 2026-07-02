"""Runtime composition helpers for concrete agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from micro_model_agent.agents.coding_agent import CodingAgent
from micro_model_agent.application.agent_workflows import RunAgentWorkflow
from micro_model_agent.application.ports import DatasetExampleStore
from micro_model_agent.infrastructure.models.fake import StaticModelProvider
from micro_model_agent.infrastructure.persistence.runtime import workflow_trace_store
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.infrastructure.tools.runtime import build_builtin_tool_executor

__all__ = ["build_static_coding_workflow"]


def build_static_coding_workflow(
    *,
    repository_root: str | Path,
    patch: str,
    allowed_commands: Mapping[str, AllowedTestCommand | Sequence[str]],
    dataset_store: DatasetExampleStore | None = None,
) -> RunAgentWorkflow:
    """Build the fixed coding workflow with a command-line supplied patch."""

    repository = Path(repository_root)
    agent = CodingAgent(
        model_provider=StaticModelProvider(patch),
        tool_executor=build_builtin_tool_executor(repository, allowed_commands),
        trace_store=workflow_trace_store(repository),
    )
    return RunAgentWorkflow(agent=agent, dataset_store=dataset_store)
