"""Reference agents for MicroModelAgent."""

from micro_model_agent.agents.coding_agent import CodingAgent, CodingAgentResult, CodingAgentTask
from micro_model_agent.agents.tool_loop_agent import (
    ToolLoopAgent,
    ToolLoopAgentResult,
    ToolLoopAgentTask,
)

__all__ = [
    "CodingAgent",
    "CodingAgentResult",
    "CodingAgentTask",
    "ToolLoopAgent",
    "ToolLoopAgentResult",
    "ToolLoopAgentTask",
]
