"""Compatibility imports for workspace registry persistence adapters."""

from micro_model_agent.infrastructure.persistence.workspace_registry import (
    JsonlWorkspaceRegistry,
    WorkspaceRecord,
    workspace_record_from_dict,
    workspace_record_to_dict,
)

__all__ = [
    "JsonlWorkspaceRegistry",
    "WorkspaceRecord",
    "workspace_record_from_dict",
    "workspace_record_to_dict",
]
