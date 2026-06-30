"""Shared helpers for CLI command modules."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Never

import typer

from micro_model_agent.infrastructure.composition import (
    EvaluationModelSelection,
    resolve_model_options,
    select_evaluation_model,
)
from micro_model_agent.infrastructure.composition import (
    base_model_from_adapter as read_base_model_from_adapter,
)

DEFAULT_TRACE_DIR = Path(".traces")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async workflow from a synchronous Typer command."""

    return asyncio.run(coro)


def _fail(message: str) -> Never:
    """Print an error and stop the current CLI command."""

    typer.echo(message, err=True)
    raise typer.Exit(1)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without overriding the process environment."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Ignore blank lines and comments, like common .env parsers do.
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _format_count_distribution(counts: object) -> str:
    """Format a JSON-ready count mapping for compact CLI output."""

    if not isinstance(counts, dict) or not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _format_tool_profile(profile: object) -> str:
    """Format a tool-profile summary for compact CLI output."""

    if not isinstance(profile, dict) or not profile:
        return "none"
    available_tools = profile.get("available_tools")
    schema_versions = profile.get("tool_schema_versions")
    tools = ", ".join(available_tools) if isinstance(available_tools, list) else "none"
    versions = ", ".join(schema_versions) if isinstance(schema_versions, list) else "none"
    return f"available_tools=[{tools}]; schema_versions=[{versions}]"


def _base_model_from_adapter(adapter_path: Path | None) -> str:
    """Read the base model name from a PEFT adapter config file."""

    try:
        return read_base_model_from_adapter(adapter_path)
    except ValueError:
        _fail("--base-model is required when no --adapter-path is provided")
    except FileNotFoundError as exc:
        _fail(str(exc))


def _resolve_loop_model_options(
    *,
    repository_root: Path,
    model: str | None,
    base_model: str | None,
    adapter_path: Path | None,
    use_adapter: bool = True,
) -> dict[str, str | Path | None]:
    """Resolve CLI loop model settings from args, env vars, and local config."""

    options = resolve_model_options(
        repository_root=repository_root,
        model=model,
        base_model=base_model,
        adapter_path=adapter_path,
        use_adapter=use_adapter,
    )
    return {
        "adapter_path": options.adapter_path,
        "base_model": options.base_model,
        "model": options.model,
    }


def _read_scripted_responses(
    scripted_response: list[str] | None,
    scripted_response_file: Path | None,
) -> list[str]:
    """Read scripted model responses from repeated CLI options and JSONL files."""

    responses = list(scripted_response or [])
    if scripted_response_file:
        if not scripted_response_file.exists():
            _fail(f"scripted response file does not exist: {scripted_response_file}")
        responses.extend(
            line
            for line in scripted_response_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return responses


def _select_evaluation_model(
    *,
    run_dir: Path,
    model: str | None,
    base_model: str | None,
    adapter_path: Path | None,
    scripted_response: list[str] | None,
    scripted_response_file: Path | None,
    max_new_tokens: int,
    ollama_base_url: str | None,
) -> EvaluationModelSelection:
    """Resolve the eval provider while preserving CLI validation messages."""

    responses = _read_scripted_responses(scripted_response, scripted_response_file)
    try:
        return select_evaluation_model(
            run_dir=run_dir,
            model=model,
            base_model=base_model,
            adapter_path=adapter_path,
            scripted_responses=responses,
            max_new_tokens=max_new_tokens,
            ollama_base_url=ollama_base_url,
        )
    except ValueError:
        _fail("--base-model is required when no --adapter-path is provided")
    except FileNotFoundError as exc:
        _fail(str(exc))
