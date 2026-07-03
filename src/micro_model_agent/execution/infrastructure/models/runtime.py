"""Runtime model and tool-loop assembly helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from micro_model_agent.execution.application.ports import ModelProvider, ToolExecutor
from micro_model_agent.execution.application.tool_loop import (
    DEFAULT_TOOL_NAMES,
    PrepareToolLoopRequest,
    RunProfile,
    RunToolLoopResult,
    RunToolLoopWorkflow,
    ToolLoopBudget,
    normalized_tool_names,
    prepare_tool_loop_run,
)
from micro_model_agent.execution.infrastructure.models.fake import ScriptedModelProvider
from micro_model_agent.execution.infrastructure.tool_loop_agent import ToolLoopAgent
from micro_model_agent.training.domain.value_objects import ModelArtifact

try:
    from micro_model_agent.execution.infrastructure.models.ollama import OllamaModelProvider
except ImportError:
    OllamaModelProvider = None  # type: ignore[assignment,misc]

try:
    from micro_model_agent.execution.infrastructure.models.transformers import (
        TransformersPeftModelProvider,
    )
except ImportError:
    TransformersPeftModelProvider = None  # type: ignore[assignment,misc]
from micro_model_agent.execution.infrastructure.persistence_runtime import (
    append_comparison_trace_event,
    workflow_trace_store,
)
from micro_model_agent.repository_ops.infrastructure.catalog import (
    BUILTIN_TOOL_SPECS,
    builtin_tool_prompt_schemas,
)
from micro_model_agent.repository_ops.infrastructure.command_runner import AllowedTestCommand
from micro_model_agent.repository_ops.infrastructure.metadata import load_repository_config
from micro_model_agent.repository_ops.infrastructure.tools_runtime import (
    PatchPolicyToolExecutor,
    allowed_test_commands,
    build_builtin_tool_executor,
)
from micro_model_agent.training.infrastructure.training_records import (
    load_artifact_from_training_run,
)

__all__ = [
    "ConfiguredToolLoopResult",
    "EvaluationModelSelection",
    "RuntimeModelOptions",
    "base_model_from_adapter",
    "build_loop_model_provider",
    "build_model_provider",
    "loop_budget_response",
    "path_config_value",
    "path_env",
    "path_or_none",
    "resolve_mcp_model_settings",
    "resolve_model_options",
    "run_cli_tool_loop",
    "run_configured_tool_loop",
    "run_mcp_agent_loop",
    "runtime_model_metadata",
    "select_evaluation_model",
    "string_config_value",
    "tool_prompt_schemas",
]

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


@dataclass(frozen=True, slots=True)
class ConfiguredToolLoopResult:
    """Result and runtime decisions for a configured tool-loop run."""

    result: RunToolLoopResult
    budget: ToolLoopBudget
    model: dict[str, str | None]
    available_tools: tuple[str, ...]
    required_tools: tuple[str, ...]
    allowed_test_command_names: tuple[str, ...]


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


def runtime_model_metadata(options: RuntimeModelOptions) -> dict[str, str | None]:
    """Return JSON-ready metadata for resolved model options."""

    return {
        "model": options.model,
        "base_model": options.base_model,
        "adapter_path": str(options.adapter_path) if options.adapter_path else None,
        "selected_promotion_artifact_id": options.selected_promotion_artifact_id,
    }


def resolve_mcp_model_settings(
    *,
    repository_root: str | Path,
    adapter_path: str | None,
    base_model: str | None,
    use_adapter: bool,
    default_adapter_path: str | Path | None,
    allow_missing_base_model: bool = False,
) -> dict[str, str | None]:
    """Resolve MCP model settings from explicit args, repository config, and defaults."""

    options = resolve_model_options(
        repository_root=repository_root,
        base_model=base_model,
        adapter_path=adapter_path,
        use_adapter=use_adapter,
        default_adapter_path=default_adapter_path,
    )
    resolved_base_model = options.base_model
    if resolved_base_model is None and not allow_missing_base_model:
        if options.adapter_path is None:
            raise ValueError("--base-model is required when use_adapter is false")
        resolved_base_model = base_model_from_adapter(options.adapter_path)

    return {
        "base_model": resolved_base_model,
        "adapter_path": str(options.adapter_path) if options.adapter_path else None,
        "selected_promotion_artifact_id": options.selected_promotion_artifact_id,
    }


def build_loop_model_provider(
    *,
    adapter_path: str | None,
    base_model: str | None,
    max_new_tokens: int,
    scripted_responses: Sequence[str] | None,
    offline: bool,
) -> ModelProvider:
    """Return the direct model provider used by loop compatibility helpers."""

    provider = build_model_provider(
        options=RuntimeModelOptions(
            model=None,
            base_model=base_model,
            adapter_path=Path(adapter_path) if adapter_path else None,
        ),
        max_new_tokens=max_new_tokens,
        scripted_responses=scripted_responses,
        offline=offline,
    )
    assert provider is not None
    return provider


def loop_budget_response(
    configured: ConfiguredToolLoopResult,
    *,
    run_profile: RunProfile | None,
) -> dict[str, int | float | str | None]:
    """Return the compact loop budget response used by MCP helpers."""

    loop_budget: dict[str, int | float | str | None] = {
        "max_turns": configured.budget.max_turns,
        "max_tool_calls": configured.budget.max_tool_calls,
        "max_new_tokens": configured.budget.max_new_tokens,
        "max_tool_result_prompt_chars": configured.budget.max_tool_result_prompt_chars,
    }
    if configured.budget.model_timeout_seconds is not None:
        loop_budget["model_timeout_seconds"] = configured.budget.model_timeout_seconds
    if run_profile is not None:
        loop_budget["run_profile"] = run_profile
    return loop_budget


async def run_configured_tool_loop(
    *,
    goal: str,
    repository_root: str | Path,
    model_options: RuntimeModelOptions,
    max_new_tokens: int,
    scripted_responses: Sequence[str] | None = None,
    ollama_base_url: str | None = None,
    offline: bool = False,
    available_tools: Sequence[str] | None = None,
    required_tools: Sequence[str] | None = None,
    default_tools: tuple[str, ...] = DEFAULT_TOOL_NAMES,
    max_turns: int = 8,
    max_tool_calls: int | None = None,
    max_tool_result_prompt_chars: int = 12_000,
    model_timeout_seconds: float | None = None,
    run_profile: RunProfile | None = None,
    context: str = "",
    schema_prompt: bool = True,
    require_tool_call: bool = True,
    capture_prompts: bool = False,
    allowed_commands: Mapping[str, AllowedTestCommand | Sequence[str]] | None = None,
    trace_repository_root: str | Path | None = None,
    executor_wrapper: Callable[[ToolExecutor], ToolExecutor] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> ConfiguredToolLoopResult:
    """Build and run the standard model-driven tool loop."""

    repository = Path(repository_root)
    command_allowlist = allowed_commands or {}
    prepared = prepare_tool_loop_run(
        PrepareToolLoopRequest(
            goal=goal,
            available_tools=tuple(available_tools) if available_tools is not None else None,
            required_tools=tuple(required_tools) if required_tools is not None else None,
            default_tools=default_tools,
            known_tools=tuple(BUILTIN_TOOL_SPECS),
            repository_has_git=(repository / ".git").exists(),
            allow_test_run=bool(command_allowlist),
            run_profile=run_profile,
            budget=ToolLoopBudget(
                max_turns=max_turns,
                max_tool_calls=max_tool_calls,
                max_new_tokens=max_new_tokens,
                max_tool_result_prompt_chars=max_tool_result_prompt_chars,
                model_timeout_seconds=model_timeout_seconds,
            ),
            context=context,
            schema_prompt=schema_prompt,
            require_tool_call=require_tool_call,
            capture_prompts=capture_prompts,
            run_metadata=run_metadata or {},
        ),
        tool_schema_builder=lambda tool_names: tool_prompt_schemas(
            tool_names,
            allowed_test_commands=command_allowlist,
        ),
    )
    model_provider = build_model_provider(
        options=model_options,
        max_new_tokens=prepared.budget.max_new_tokens,
        scripted_responses=scripted_responses,
        ollama_base_url=ollama_base_url,
        offline=offline,
    )
    assert model_provider is not None

    executor: ToolExecutor = build_builtin_tool_executor(repository, command_allowlist)
    if executor_wrapper is not None:
        executor = executor_wrapper(executor)
    agent = ToolLoopAgent(
        model_provider=model_provider,
        tool_executor=executor,
        trace_store=workflow_trace_store(trace_repository_root or repository),
    )
    workflow = RunToolLoopWorkflow(agent)
    result = await workflow.run(prepared.request)

    return ConfiguredToolLoopResult(
        result=result,
        budget=prepared.budget,
        model=runtime_model_metadata(model_options),
        available_tools=prepared.available_tools,
        required_tools=prepared.required_tools,
        allowed_test_command_names=tuple(command_allowlist),
    )


async def run_mcp_agent_loop(
    *,
    goal: str,
    repository_root: str | Path = ".",
    context: str = "",
    adapter_path: str | None = None,
    base_model: str | None = None,
    use_adapter: bool = True,
    available_tools: Sequence[str] | None = None,
    required_tools: Sequence[str] | None = None,
    max_turns: int = 4,
    max_tool_calls: int | None = 1,
    max_new_tokens: int = 350,
    max_tool_result_prompt_chars: int = 2500,
    model_timeout_seconds: float | None = None,
    run_profile: RunProfile | None = None,
    schema_prompt: bool = True,
    capture_prompts: bool = False,
    apply_patches: bool = False,
    allow_test_run: bool = False,
    test_command_name: str | None = None,
    test_command_args: Sequence[str] | None = None,
    scripted_responses: Sequence[str] | None = None,
    comparison_session_id: str | None = None,
    comparison_repository_root: str | Path | None = None,
    offline: bool = True,
    default_adapter_path: str | Path | None = None,
    default_available_tools: tuple[str, ...] = DEFAULT_TOOL_NAMES,
    tool_aliases: Mapping[str, Sequence[str]] | None = None,
    required_tool_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Run the MCP-flavored model tool loop and return its response record."""

    repository = Path(repository_root)
    test_commands = allowed_test_commands(
        test_command_name,
        test_command_args,
        use_default_pytest=allow_test_run and not test_command_name,
    )
    loop_tool_names = normalized_tool_names(
        available_tools,
        default_tools=default_available_tools,
        aliases=tool_aliases,
    )
    loop_required_tool_names = normalized_tool_names(
        required_tools,
        default_tools=(),
        aliases=required_tool_aliases,
    )
    model_settings = resolve_mcp_model_settings(
        repository_root=repository,
        adapter_path=adapter_path,
        base_model=base_model,
        use_adapter=use_adapter,
        default_adapter_path=default_adapter_path,
        allow_missing_base_model=bool(scripted_responses),
    )
    configured = await run_configured_tool_loop(
        goal=goal,
        repository_root=repository,
        model_options=RuntimeModelOptions(
            model=None,
            base_model=model_settings["base_model"],
            adapter_path=Path(model_settings["adapter_path"])
            if model_settings["adapter_path"]
            else None,
            selected_promotion_artifact_id=model_settings["selected_promotion_artifact_id"],
        ),
        max_new_tokens=max_new_tokens,
        scripted_responses=scripted_responses,
        offline=offline,
        available_tools=loop_tool_names,
        required_tools=loop_required_tool_names,
        default_tools=default_available_tools,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_tool_result_prompt_chars=max_tool_result_prompt_chars,
        model_timeout_seconds=model_timeout_seconds,
        run_profile=run_profile,
        context=context,
        schema_prompt=schema_prompt,
        capture_prompts=capture_prompts,
        allowed_commands=test_commands,
        trace_repository_root=comparison_repository_root or repository,
        executor_wrapper=lambda executor: PatchPolicyToolExecutor(
            executor,
            apply_patches=apply_patches,
        ),
        run_metadata={
            "interface": "mcp",
            "apply_patches": apply_patches,
            "allowed_test_commands": list(test_commands),
            "model": model_settings,
        },
    )
    result = configured.result
    if comparison_session_id:
        await append_comparison_trace_event(
            repository_root=Path(comparison_repository_root or repository),
            session_id=comparison_session_id,
            event_type="local_model_run",
            actor="micro_model_agent",
            payload={
                "trace_id": str(result.trace_id),
                "ok": result.ok,
                "response": result.response,
                "turns_used": result.turns_used,
                "tool_calls_made": result.tool_calls_made,
                "model": configured.model,
            },
        )

    return {
        "ok": result.ok,
        "response": result.response,
        "trace_id": str(result.trace_id),
        "turns_used": result.turns_used,
        "tool_calls_made": result.tool_calls_made,
        "loop_budget": loop_budget_response(configured, run_profile=run_profile),
        "model": configured.model,
        "steps": [
            {
                "name": step.name,
                "status": step.status.value,
                "tool_name": step.tool_call.tool_name if step.tool_call else None,
                "tool_ok": step.tool_result.ok if step.tool_result else None,
                "tool_error": step.tool_result.error if step.tool_result else None,
            }
            for step in result.trace.steps
        ],
    }


async def run_cli_tool_loop(
    *,
    goal: str,
    repository_root: str | Path,
    model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | Path | None = None,
    use_adapter: bool = True,
    ollama_base_url: str | None = None,
    max_new_tokens: int,
    scripted_responses: Sequence[str] | None = None,
    available_tools: Sequence[str] | None = None,
    required_tools: Sequence[str] | None = None,
    max_turns: int,
    max_tool_calls: int | None,
    max_tool_result_prompt_chars: int,
    context: str = "",
    schema_prompt: bool = True,
    capture_prompts: bool = False,
    require_tool_call: bool = True,
    verification_command: str | None = None,
    test_command: Sequence[str] | None = None,
) -> ConfiguredToolLoopResult:
    """Run the CLI-flavored model tool loop with standard runtime metadata."""

    responses = list(scripted_responses or [])
    model_options = resolve_model_options(
        repository_root=repository_root,
        model=model,
        base_model=base_model,
        adapter_path=adapter_path,
        use_adapter=use_adapter,
    )
    if not responses and not (
        model_options.adapter_path or model_options.base_model or model_options.model
    ):
        raise ValueError(
            "--model, MICRO_MODEL_AGENT_DEFAULT_MODEL, or configured model default is "
            "required without scripted responses"
        )
    if verification_command and not test_command:
        raise ValueError("--test-command is required when --verification-command is set")

    allowed_commands = allowed_test_commands(verification_command, test_command)
    return await run_configured_tool_loop(
        goal=goal,
        repository_root=repository_root,
        model_options=model_options,
        max_new_tokens=max_new_tokens,
        scripted_responses=responses,
        ollama_base_url=ollama_base_url
        or os.environ.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
        available_tools=tuple(available_tools or DEFAULT_TOOL_NAMES),
        required_tools=tuple(required_tools or ()),
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_tool_result_prompt_chars=max_tool_result_prompt_chars,
        context=context,
        schema_prompt=schema_prompt,
        require_tool_call=require_tool_call,
        capture_prompts=capture_prompts,
        allowed_commands=allowed_commands,
        run_metadata={
            "interface": "cli.loop",
            "use_adapter": use_adapter,
            "model": {
                "model": model_options.model,
                "base_model": model_options.base_model,
                "adapter_path": str(model_options.adapter_path)
                if model_options.adapter_path
                else None,
            },
        },
    )


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


def tool_prompt_schemas(
    allowed_tool_names: tuple[str, ...],
    *,
    allowed_test_commands: Mapping[str, object],
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
