"""Shared runtime composition helpers for interface adapters."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from micro_model_agent.agents.coding_agent import CodingAgent
from micro_model_agent.agents.tool_loop_agent import ToolLoopAgent
from micro_model_agent.application.datasets import (
    RunDatasetExportWorkflow,
    RunDatasetMergeWorkflow,
    RunDatasetRelabelWorkflow,
    RunDatasetSynthesisWorkflow,
    RunDatasetValidationWorkflow,
    RunTraceDatasetExportWorkflow,
    RunTraceReviewWorkflow,
)
from micro_model_agent.application.evaluation import (
    RunEvaluationComparisonWorkflow,
    RunSyntheticEvaluationWorkflow,
    RunTraceEvaluationWorkflow,
    RunWorkspaceStagedEvaluationWorkflow,
    RunWorkspaceStagedReviewWorkflow,
)
from micro_model_agent.application.ports import DatasetExampleStore, ModelProvider, ToolExecutor
from micro_model_agent.application.promotion import (
    RunPromotionGateWorkflow,
    RunPromotionListWorkflow,
    RunPromotionPackageOllamaWorkflow,
    RunPromotionRecordWorkflow,
    RunPromotionSelectWorkflow,
)
from micro_model_agent.application.tool_loop import (
    DEFAULT_TOOL_NAMES,
    PrepareToolLoopRequest,
    RunProfile,
    RunToolLoopResult,
    RunToolLoopWorkflow,
    ToolLoopBudget,
    prepare_tool_loop_run,
)
from micro_model_agent.application.training import RunSyntheticTrainingWorkflow
from micro_model_agent.application.workflows import RunAgentWorkflow
from micro_model_agent.domain.training import ModelArtifact
from micro_model_agent.infrastructure.datasets.curation import (
    LocalDatasetMerger,
    LocalDatasetRelabeler,
)
from micro_model_agent.infrastructure.datasets.metadata import (
    LocalDatasetFileHasher,
    LocalDatasetToolProfileSummarizer,
)
from micro_model_agent.infrastructure.datasets.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.datasets.validation import (
    LocalDatasetValidator,
    SftJsonlDatasetExporter,
)
from micro_model_agent.infrastructure.evaluation.artifact import SyntheticEvaluationSuite
from micro_model_agent.infrastructure.evaluation.comparison import (
    LocalEvaluationComparisonReportWriter,
)
from micro_model_agent.infrastructure.evaluation.reports import (
    LocalEvaluationResultReader,
    LocalEvaluationResultWriter,
)
from micro_model_agent.infrastructure.evaluation.synthetic_behavior import (
    SyntheticBehaviorEvaluationSuite,
)
from micro_model_agent.infrastructure.evaluation.trace_behavior import (
    TraceBehaviorEvaluationSuite,
)
from micro_model_agent.infrastructure.evaluation.workspace_staged import (
    WorkspaceStagedEvaluationSuite,
)
from micro_model_agent.infrastructure.evaluation.workspace_staged_review import (
    LocalWorkspaceStagedReviewBuilder,
    LocalWorkspaceStagedReviewQueueWriter,
)
from micro_model_agent.infrastructure.models.fake import ScriptedModelProvider, StaticModelProvider
from micro_model_agent.infrastructure.models.ollama import OllamaModelProvider
from micro_model_agent.infrastructure.models.transformers import (
    TransformersPeftModelProvider,
)
from micro_model_agent.infrastructure.persistence.comparison_trace import JsonlComparisonTraceStore
from micro_model_agent.infrastructure.persistence.dataset_store import (
    JsonlDatasetExampleStore,
    LocalDatasetExampleReader,
    LocalDatasetExampleWriter,
)
from micro_model_agent.infrastructure.persistence.trace_store import (
    JsonlTraceStore,
    LocalWorkflowTraceReader,
)
from micro_model_agent.infrastructure.persistence.training_records import (
    load_artifact_from_training_run,
)
from micro_model_agent.infrastructure.persistence.workspace_registry import (
    JsonlWorkspaceRegistry,
    WorkspaceRecord,
    workspace_record_to_dict,
)
from micro_model_agent.infrastructure.promotion.gate import (
    LocalPromotionGateStore,
    MinimumScorePromotionPolicy,
)
from micro_model_agent.infrastructure.repositories.local_index import (
    LocalIndexResult,
    LocalLexicalIndexWriter,
)
from micro_model_agent.infrastructure.repositories.metadata import (
    LocalRepositoryModelConfigurationStore,
    RepositoryInitializationResult,
    initialize_repository,
    is_repository_initialized,
    load_repository_config,
)
from micro_model_agent.infrastructure.tools.catalog import (
    BUILTIN_TOOL_SPECS,
    TOOL_ARGUMENT_CONTRACTS,
    builtin_tool_prompt_schemas,
)
from micro_model_agent.infrastructure.tools.command_runner import AllowedTestCommand
from micro_model_agent.infrastructure.tools.executor import BuiltinToolExecutor
from micro_model_agent.infrastructure.traces.export import (
    LocalTraceDatasetExporter,
    LocalTraceDatasetExportValidator,
)
from micro_model_agent.infrastructure.traces.review import (
    LocalTraceReviewReader,
    LocalTraceReviewWriter,
)
from micro_model_agent.infrastructure.training.artifacts import (
    FakeTrainingRunner,
    JsonTrainingArtifactStore,
)
from micro_model_agent.infrastructure.training.local_finetuning import LocalFineTuningRunner
from micro_model_agent.infrastructure.training.ollama_packaging import (
    LocalOllamaAdapterPackager,
)

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


def build_jsonl_dataset_example_store(path: str | Path) -> DatasetExampleStore:
    """Build the local JSONL dataset store used by coding task capture."""

    return JsonlDatasetExampleStore(path)


def initialize_local_repository(
    repository_root: str | Path,
    *,
    default_model: str | None = None,
    base_model: str | None = None,
    adapter_path: str | Path | None = None,
) -> RepositoryInitializationResult:
    """Initialize local MicroModelAgent repository metadata."""

    return initialize_repository(
        repository_root,
        default_model=default_model,
        base_model=base_model,
        adapter_path=adapter_path,
    )


def local_repository_initialized(repository_root: str | Path) -> bool:
    """Return whether local MicroModelAgent metadata is initialized."""

    return is_repository_initialized(repository_root)


def write_local_repository_index(
    repository_root: str | Path,
    *,
    max_file_bytes: int,
) -> LocalIndexResult:
    """Write the standard local lexical repository index."""

    return LocalLexicalIndexWriter(
        repository_root,
        max_file_bytes=max_file_bytes,
    ).write()


async def register_workspace_record(
    *,
    registry_root: str | Path,
    workspace_path: str | Path,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a workspace record and return its JSON-ready representation."""

    record = WorkspaceRecord(
        path=str(workspace_path),
        name=name,
        metadata=dict(metadata or {}),
    )
    await workspace_registry(registry_root).save(record)
    return workspace_record_to_dict(record)


async def registered_workspace_path(
    *,
    registry_root: str | Path,
    workspace_id: str,
) -> Path | None:
    """Resolve a registered workspace id to its repository path."""

    workspace = await workspace_registry(registry_root).get(workspace_id)
    if workspace is None:
        return None
    return Path(workspace.path)


def build_synthetic_training_workflow(
    *,
    dry_run: bool,
    artifact_store_root: str | Path = Path(".micro_model_agent/training"),
) -> RunSyntheticTrainingWorkflow:
    """Build the standard synthetic training workflow for CLI entrypoints."""

    runner = FakeTrainingRunner() if dry_run else LocalFineTuningRunner()
    return RunSyntheticTrainingWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
        exporter=SftJsonlDatasetExporter(),
        file_hasher=LocalDatasetFileHasher(),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        runner=runner,
        artifact_store=JsonTrainingArtifactStore(artifact_store_root),
    )


def build_dataset_synthesis_workflow(
    *,
    template_dir: str | Path,
) -> RunDatasetSynthesisWorkflow:
    """Build the standard synthetic dataset generation workflow."""

    return RunDatasetSynthesisWorkflow(
        generator=SyntheticTemplateGenerator(template_dir),
        validator=LocalDatasetValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )


def build_dataset_validation_workflow() -> RunDatasetValidationWorkflow:
    """Build the standard persisted dataset validation workflow."""

    return RunDatasetValidationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
    )


def build_dataset_export_workflow() -> RunDatasetExportWorkflow:
    """Build the standard dataset export workflow."""

    return RunDatasetExportWorkflow(
        example_reader=LocalDatasetExampleReader(),
        validator=LocalDatasetValidator(),
        exporter=SftJsonlDatasetExporter(),
    )


def build_trace_dataset_export_workflow(
    *,
    trace_path: str | Path,
    review_path: str | Path,
) -> RunTraceDatasetExportWorkflow:
    """Build the standard trace-to-dataset export workflow."""

    return RunTraceDatasetExportWorkflow(
        trace_reader=LocalWorkflowTraceReader(trace_path),
        review_reader=LocalTraceReviewReader(review_path),
        trace_exporter=LocalTraceDatasetExporter(),
        trace_export_validator=LocalTraceDatasetExportValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )


def build_trace_review_workflow() -> RunTraceReviewWorkflow:
    """Build the standard human trace review recording workflow."""

    return RunTraceReviewWorkflow(review_writer=LocalTraceReviewWriter())


def build_dataset_relabel_workflow() -> RunDatasetRelabelWorkflow:
    """Build the standard dataset relabeling workflow."""

    return RunDatasetRelabelWorkflow(
        example_reader=LocalDatasetExampleReader(),
        relabeler=LocalDatasetRelabeler(),
        example_writer=LocalDatasetExampleWriter(),
        validator=LocalDatasetValidator(),
    )


def build_dataset_merge_workflow() -> RunDatasetMergeWorkflow:
    """Build the standard dataset merge workflow."""

    return RunDatasetMergeWorkflow(
        example_reader=LocalDatasetExampleReader(),
        merger=LocalDatasetMerger(),
        validator=LocalDatasetValidator(),
        example_writer=LocalDatasetExampleWriter(),
    )


def default_evaluation_available_tools() -> tuple[str, ...]:
    """Return the default tool names used in behavior evaluation prompts."""

    return tuple(TOOL_ARGUMENT_CONTRACTS)


def build_synthetic_evaluation_workflow(
    *,
    pass_threshold: float,
) -> RunSyntheticEvaluationWorkflow:
    """Build the standard synthetic evaluation workflow."""

    return RunSyntheticEvaluationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        behavior_suite=SyntheticBehaviorEvaluationSuite(pass_threshold=pass_threshold),
        artifact_suite=SyntheticEvaluationSuite(),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        evaluation_writer=LocalEvaluationResultWriter(),
    )


def build_trace_evaluation_workflow(
    *,
    pass_threshold: float,
) -> RunTraceEvaluationWorkflow:
    """Build the standard trace-derived evaluation workflow."""

    return RunTraceEvaluationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        behavior_suite=TraceBehaviorEvaluationSuite(pass_threshold=pass_threshold),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        evaluation_writer=LocalEvaluationResultWriter(),
    )


def build_workspace_staged_evaluation_workflow(
    *,
    pass_threshold: float,
    rubric_version: str,
) -> RunWorkspaceStagedEvaluationWorkflow:
    """Build the standard staged workspace evaluation workflow."""

    return RunWorkspaceStagedEvaluationWorkflow(
        example_reader=LocalDatasetExampleReader(),
        behavior_suite=WorkspaceStagedEvaluationSuite(
            pass_threshold=pass_threshold,
            rubric_version=rubric_version,
        ),
        tool_profile_summarizer=LocalDatasetToolProfileSummarizer(),
        evaluation_writer=LocalEvaluationResultWriter(),
    )


def build_workspace_staged_review_workflow() -> RunWorkspaceStagedReviewWorkflow:
    """Build the standard staged workspace review queue workflow."""

    return RunWorkspaceStagedReviewWorkflow(
        example_reader=LocalDatasetExampleReader(),
        evaluation_reader=LocalEvaluationResultReader(),
        review_builder=LocalWorkspaceStagedReviewBuilder(),
        review_writer=LocalWorkspaceStagedReviewQueueWriter(),
    )


def build_evaluation_comparison_workflow() -> RunEvaluationComparisonWorkflow:
    """Build the standard evaluation comparison workflow."""

    return RunEvaluationComparisonWorkflow(
        evaluation_reader=LocalEvaluationResultReader(),
        comparison_writer=LocalEvaluationComparisonReportWriter(),
    )


def build_promotion_gate_workflow() -> RunPromotionGateWorkflow:
    """Build the standard promotion gate workflow."""

    store = LocalPromotionGateStore()
    return RunPromotionGateWorkflow(
        artifact_reader=store,
        evaluation_reader=store,
        result_writer=store,
        policy_factory=MinimumScorePromotionPolicy,
    )


def build_promotion_record_workflow() -> RunPromotionRecordWorkflow:
    """Build the standard promotion registry record workflow."""

    store = LocalPromotionGateStore()
    return RunPromotionRecordWorkflow(
        artifact_reader=store,
        registry_writer=store,
    )


def build_promotion_list_workflow() -> RunPromotionListWorkflow:
    """Build the standard promotion registry listing workflow."""

    return RunPromotionListWorkflow(registry_reader=LocalPromotionGateStore())


def build_promotion_select_workflow() -> RunPromotionSelectWorkflow:
    """Build the standard repository model selection workflow."""

    return RunPromotionSelectWorkflow(
        registry_reader=LocalPromotionGateStore(),
        configuration_writer=LocalRepositoryModelConfigurationStore(),
    )


def build_promotion_package_ollama_workflow() -> RunPromotionPackageOllamaWorkflow:
    """Build the standard Ollama packaging workflow for promoted adapters."""

    return RunPromotionPackageOllamaWorkflow(
        registry_reader=LocalPromotionGateStore(),
        packager=LocalOllamaAdapterPackager(),
    )


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
