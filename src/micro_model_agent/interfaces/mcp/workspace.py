"""Workspace helpers for MCP tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp.exceptions import ToolError as FastMcpToolError

from micro_model_agent.infrastructure.composition import (
    workspace_registry as build_workspace_registry,
)
from micro_model_agent.infrastructure.persistence.workspace_registry import (
    WorkspaceRecord,
    workspace_record_to_dict,
)
from micro_model_agent.infrastructure.repositories.metadata import initialize_repository
from micro_model_agent.interfaces.mcp.compat import WINDOWS_ABSOLUTE_PATH_RE


async def init_workspace(
    *,
    registry_root: str | Path,
    path: str | None = None,
    name: str | None = None,
    create: bool = True,
    initialize: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or register a workspace and optionally initialize metadata there."""

    registry_base = Path(registry_root).resolve()
    if path:
        workspace_path = path_from_user_input(path).resolve()
    else:
        workspace_id = uuid4()
        directory_name = path_safe_name(name or f"workspace-{workspace_id}")
        workspace_path = registry_base / ".micro_model_agent" / "workspaces" / directory_name

    if workspace_path.exists() and not workspace_path.is_dir():
        return {"ok": False, "error": f"workspace path is not a directory: {workspace_path}"}
    if not workspace_path.exists():
        if not create:
            return {"ok": False, "error": f"workspace path does not exist: {workspace_path}"}
        workspace_path.mkdir(parents=True, exist_ok=True)

    init_result = None
    if initialize:
        init_result = initialize_repository(workspace_path).to_dict()
        if init_result.get("ok") is not True:
            return {"ok": False, "error": init_result.get("error"), "init": init_result}

    record = WorkspaceRecord(
        path=str(workspace_path),
        name=name,
        metadata=dict(metadata or {}),
    )
    await workspace_registry(registry_base).save(record)
    return {
        "ok": True,
        "workspace": workspace_record_to_dict(record),
        "repository_root": str(workspace_path),
        "init": init_result,
    }


def init_repository(
    *,
    repository_root: str = ".",
    default_model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    """Initialize repository metadata through MCP."""

    return initialize_repository(
        repository_root,
        default_model=default_model,
        base_model=base_model,
        adapter_path=adapter_path,
    ).to_dict()


async def resolve_workspace_root(
    *,
    registry_root: Path,
    repository_root: str,
    workspace_id: str | None,
) -> Path:
    """Resolve a workspace id or direct repository root into a filesystem path."""

    if not workspace_id:
        return Path(repository_root)
    workspace = await workspace_registry(registry_root.resolve()).get(workspace_id)
    if workspace is None:
        raise FastMcpToolError(f"workspace_id not found: {workspace_id}")
    return Path(workspace.path)


def workspace_registry(registry_root: Path):
    """Return the workspace registry for the server's default root."""

    return build_workspace_registry(registry_root)


def path_safe_name(value: str) -> str:
    """Return a conservative directory name for generated workspaces."""

    safe = "".join(character if character.isalnum() else "-" for character in value.lower())
    return "-".join(part for part in safe.split("-") if part) or "workspace"


def path_from_user_input(value: str, *, wsl_mount_root: Path = Path("/mnt")) -> Path:
    """Convert Windows absolute paths to WSL mount paths when running on POSIX."""

    use_wsl_mounts = os.name != "nt" or wsl_mount_root != Path("/mnt")
    if use_wsl_mounts and (match := WINDOWS_ABSOLUTE_PATH_RE.match(value)):
        drive = match.group("drive").lower()
        rest = match.group("rest").replace("\\", "/")
        return wsl_mount_root / drive / rest
    return Path(value).expanduser()
