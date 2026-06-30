"""Shared runtime composition helpers for interface adapters."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from micro_model_agent.application.ports import ModelProvider
from micro_model_agent.domain.training import ModelArtifact
from micro_model_agent.infrastructure.comparison_trace import JsonlComparisonTraceStore
from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
from micro_model_agent.infrastructure.ollama_model_provider import OllamaModelProvider
from micro_model_agent.infrastructure.repository_metadata import load_repository_config
from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor
from micro_model_agent.infrastructure.tools.catalog import builtin_tool_prompt_schemas
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.infrastructure.trace_store import JsonlTraceStore
from micro_model_agent.infrastructure.training_artifacts import load_artifact_from_training_run
from micro_model_agent.infrastructure.transformers_model_provider import (
    TransformersPeftModelProvider,
)
from micro_model_agent.infrastructure.workspace_registry import JsonlWorkspaceRegistry

DEFAULT_TRACE_DIR = Path(".traces")
_MODEL_CACHE: dict[tuple[str, str | None, int], TransformersPeftModelProvider] = {}


@dataclass(frozen=True, slots=True)
class RuntimeModelOptions:
    """Resolved model options shared by CLI and MCP composition."""

    model: str | None
    base_model: str | None
    adapter_path: Path | None
    selected_promotion_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationModelSelection:
    """Resolved provider and metadata for CLI evaluation commands."""

    provider: ModelProvider | None
    provider_kind: str
    model: str | None
    base_model: str | None
    adapter_path: Path | None
    artifact: ModelArtifact | None = None


def resolve_model_options(
    *,
    repository_root: str | Path,
    model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | Path | None = None,
    use_adapter: bool = True,
    default_adapter_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RuntimeModelOptions:
    """Resolve model settings from explicit args, environment, repository config."""

    environment = env if env is not None else os.environ
    repository_config = load_repository_config(repository_root) or {}
    model_config = repository_config.get("model")
    if not isinstance(model_config, dict):
        model_config = {}

    resolved_adapter_path = None
    if use_adapter:
        resolved_adapter_path = (
            path_or_none(adapter_path)
            or path_env(environment, "MICRO_MODEL_AGENT_ADAPTER_PATH")
            or path_config_value(model_config, "adapter_path")
            or path_or_none(default_adapter_path)
        )
    resolved_base_model = (
        base_model
        or environment.get("MICRO_MODEL_AGENT_BASE_MODEL")
        or string_config_value(model_config, "base_model")
    )
    resolved_model = (
        model
        or environment.get("MICRO_MODEL_AGENT_DEFAULT_MODEL")
        or string_config_value(model_config, "default_model")
    )
    selected_promotion = model_config.get("selected_promotion")
    selected_artifact_id = None
    if isinstance(selected_promotion, dict):
        raw_artifact_id = selected_promotion.get("artifact_id")
        selected_artifact_id = raw_artifact_id if isinstance(raw_artifact_id, str) else None

    return RuntimeModelOptions(
        model=resolved_model,
        base_model=resolved_base_model,
        adapter_path=resolved_adapter_path,
        selected_promotion_artifact_id=selected_artifact_id,
    )


def build_model_provider(
    *,
    options: RuntimeModelOptions,
    max_new_tokens: int,
    scripted_responses: Sequence[str] | None = None,
    ollama_base_url: str | None = None,
    offline: bool = False,
    require_model: bool = True,
) -> ModelProvider | None:
    """Build a model provider from resolved runtime options."""

    if scripted_responses:
        return ScriptedModelProvider(list(scripted_responses))

    if options.adapter_path or options.base_model:
        if offline:
            # Offline defaults keep local adapter runs from unexpectedly reaching
            # out to the Hugging Face hub.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        resolved_base_model = options.base_model or base_model_from_adapter(options.adapter_path)
        cache_key = (
            resolved_base_model,
            str(options.adapter_path) if options.adapter_path else None,
            max_new_tokens,
        )
        provider = _MODEL_CACHE.get(cache_key)
        if provider is None:
            provider = TransformersPeftModelProvider(
                base_model=resolved_base_model,
                adapter_path=options.adapter_path,
                max_new_tokens=max_new_tokens,
            )
            _MODEL_CACHE[cache_key] = provider
        return provider

    if options.model:
        return OllamaModelProvider(
            model_name=options.model,
            base_url=ollama_base_url,
            options={"num_predict": max_new_tokens},
        )

    if require_model:
        raise ValueError("a model, base model, adapter path, or scripted responses are required")
    return None


def select_evaluation_model(
    *,
    run_dir: str | Path,
    model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | Path | None = None,
    scripted_responses: Sequence[str] | None = None,
    max_new_tokens: int,
    ollama_base_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> EvaluationModelSelection:
    """Select the runnable model provider or metadata fallback for evaluation."""

    responses = list(scripted_responses or [])
    normalized_adapter_path = path_or_none(adapter_path)
    resolved_base_model = base_model
    artifact: ModelArtifact | None = None
    provider_kind = "metadata"

    if responses:
        provider_kind = "scripted"
        provider = build_model_provider(
            options=RuntimeModelOptions(
                model=model,
                base_model=base_model,
                adapter_path=normalized_adapter_path,
            ),
            scripted_responses=responses,
            max_new_tokens=max_new_tokens,
        )
    elif normalized_adapter_path or base_model:
        provider_kind = "transformers_peft"
        resolved_base_model = base_model or base_model_from_adapter(normalized_adapter_path)
        provider = build_model_provider(
            options=RuntimeModelOptions(
                model=None,
                base_model=resolved_base_model,
                adapter_path=normalized_adapter_path,
            ),
            max_new_tokens=max_new_tokens,
        )
    elif model:
        provider_kind = "ollama"
        environment = env if env is not None else os.environ
        provider = build_model_provider(
            options=RuntimeModelOptions(
                model=model,
                base_model=None,
                adapter_path=None,
            ),
            max_new_tokens=max_new_tokens,
            ollama_base_url=ollama_base_url
            or environment.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
        )
    else:
        artifact = load_artifact_from_training_run(Path(run_dir))
        artifact_path = Path(artifact.path)
        if artifact_path.exists():
            provider_kind = "training_artifact"
            resolved_base_model = artifact.base_model
            normalized_adapter_path = artifact_path
            provider = build_model_provider(
                options=RuntimeModelOptions(
                    model=None,
                    base_model=artifact.base_model,
                    adapter_path=artifact_path,
                ),
                max_new_tokens=max_new_tokens,
            )
        else:
            provider = None

    return EvaluationModelSelection(
        provider=provider,
        provider_kind=provider_kind,
        model=model,
        base_model=resolved_base_model,
        adapter_path=normalized_adapter_path,
        artifact=artifact,
    )


def base_model_from_adapter(adapter_path: Path | None) -> str:
    """Read the base model name stored in a PEFT adapter config."""

    if adapter_path is None:
        raise ValueError("base_model is required when adapter_path is not set")
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter config does not exist: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError(f"adapter config does not contain base_model_name_or_path: {config_path}")
    return base_model


def allowed_test_commands(
    test_command_name: str | None,
    test_command_args: Sequence[str] | None,
    *,
    use_default_pytest: bool = False,
) -> dict[str, AllowedTestCommand]:
    """Build the allowlist consumed by the test.run tool."""

    if use_default_pytest:
        return {"pytest": AllowedTestCommand(("python3", "-m", "pytest", "-q"))}
    if not test_command_name or not test_command_args:
        return {}
    return {test_command_name: AllowedTestCommand(tuple(test_command_args))}


def build_builtin_tool_executor(
    repository_root: str | Path,
    allowed_commands: Mapping[str, AllowedTestCommand | Sequence[str]],
) -> BuiltinToolExecutor:
    """Build the standard repository tool executor."""

    return BuiltinToolExecutor(repository_root, allowed_commands)


def workflow_trace_store(
    repository_root: str | Path,
    *,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> JsonlTraceStore:
    """Return the workflow trace store for one repository."""

    return JsonlTraceStore(
        trace_dir(repository_root, trace_dir_name=trace_dir_name) / "workflows.jsonl"
    )


def comparison_trace_store(
    repository_root: str | Path,
    *,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> JsonlComparisonTraceStore:
    """Return the comparison trace store for one repository."""

    return JsonlComparisonTraceStore(
        trace_dir(repository_root, trace_dir_name=trace_dir_name) / "comparison_sessions.jsonl"
    )


def trace_dir(
    repository_root: str | Path,
    *,
    trace_dir_name: str | Path = DEFAULT_TRACE_DIR,
) -> Path:
    """Return the centralized trace/log directory for one repository."""

    return Path(repository_root) / trace_dir_name


def workspace_registry(registry_root: str | Path) -> JsonlWorkspaceRegistry:
    """Return the workspace registry for the server's default root."""

    return JsonlWorkspaceRegistry(Path(registry_root) / ".micro_model_agent" / "workspaces.jsonl")


def tool_prompt_schemas(
    allowed_tool_names: tuple[str, ...],
    *,
    allowed_test_commands: Mapping[str, AllowedTestCommand],
) -> dict[str, Any]:
    """Return tool schemas enriched with runtime command allowlists."""

    schemas = builtin_tool_prompt_schemas(allowed_tool_names)
    test_schema = schemas.get("test.run")
    if test_schema is not None:
        test_schema["allowed_command_names"] = list(allowed_test_commands)
        test_schema["description"] = (
            str(test_schema["description"])
            + " Use command_name from allowed_command_names exactly."
        )
    return schemas


def path_env(environment: Mapping[str, str], name: str) -> Path | None:
    """Read one path environment variable."""

    value = environment.get(name)
    return Path(value) if value else None


def path_config_value(config: dict[str, Any], key: str) -> Path | None:
    """Read one path value from repository config."""

    value = string_config_value(config, key)
    return Path(value) if value else None


def path_or_none(value: str | Path | None) -> Path | None:
    """Convert a string/path value to a path when present."""

    return Path(value) if value else None


def string_config_value(config: dict[str, Any], key: str) -> str | None:
    """Read one string value from repository config."""

    value = config.get(key)
    return value if isinstance(value, str) and value else None
