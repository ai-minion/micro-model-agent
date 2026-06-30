"""Compatibility constants for the MCP boundary."""

from __future__ import annotations

import re
from typing import Literal

DEFAULT_MCP_AVAILABLE_TOOLS: tuple[str, ...] = (
    "repo.search",
    "repo.read",
    "repo.semantic_search",
    "repo.write_patch",
    "repo.write_files",
    "git.diff",
)
CANONICAL_TOOL_NAMES_TEXT = (
    "repo.search, repo.read, repo.semantic_search, repo.write_patch, "
    "repo.write_files, test.run, git.diff"
)
DEFAULT_7B_ADAPTER_PATH = (
    ".micro_model_agent/training/runs/qwen-coder-7b-tool-schema-20260613-205520/adapter"
)
COMPAT_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    # Common Codex/legacy names accepted at the MCP boundary. The model still
    # only sees MicroModelAgent built-in tool names after normalization.
    "apply_patch": ("repo.write_files", "repo.write_patch"),
    "list_files": ("repo.search",),
    "read_file": ("repo.read",),
    "run_tests": ("test.run",),
    "shell": (),
    "write_file": ("repo.write_files",),
}
COMPAT_REQUIRED_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    # Treat a legacy apply_patch requirement as "perform a write"; for
    # greenfield tasks, repo.write_files is the preferred write primitive.
    "apply_patch": ("repo.write_files",),
    **{
        alias: canonical
        for alias, canonical in COMPAT_TOOL_ALIASES.items()
        if alias != "apply_patch"
    },
}
MCP_DEBUG_TOOLS_ENV = "MICRO_MODEL_AGENT_MCP_DEBUG_TOOLS"
MCP_EXPOSE_INIT_ENV = "MICRO_MODEL_AGENT_MCP_EXPOSE_INIT"
MCP_REPOSITORY_ROOT_ENV = "MICRO_MODEL_AGENT_REPOSITORY_ROOT"
MCP_INIT_TOOL_NAME = "micro_agent_init"
TRACE_DIR_NAME = ".traces"
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
type McpTransport = Literal["stdio", "sse", "streamable-http"]
type RunProfile = Literal["quick", "standard", "extended"]
