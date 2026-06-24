"""Command-line interface entrypoint.

Typer turns these Python functions into terminal commands. The CLI mostly wires
domain/application services together, prints a short result, and exits with a
nonzero status when a command fails.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Never, cast

import typer

from micro_model_agent.domain.contracts import EvaluationResult, WorkflowStatus
from micro_model_agent.domain.datasets import (
    DatasetExample,
    DatasetExampleKind,
    FailureMode,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.domain.training import TrainingConfig
from micro_model_agent.infrastructure.dataset_curation import (
    DeduplicateBy,
    merge_datasets,
    relabel_examples,
)
from micro_model_agent.infrastructure.dataset_metadata import (
    dataset_file_sha256,
    summarize_tool_profiles,
)
from micro_model_agent.infrastructure.dataset_store import (
    JsonlDatasetExampleStore,
    load_dataset_examples,
)
from micro_model_agent.infrastructure.dataset_validation import (
    LocalDatasetValidator,
    export_sft_jsonl,
)
from micro_model_agent.infrastructure.evaluation_comparison import (
    compare_evaluation_results,
)
from micro_model_agent.infrastructure.local_index import LocalLexicalIndexWriter
from micro_model_agent.infrastructure.ollama_packaging import (
    package_promoted_adapter_for_ollama,
)
from micro_model_agent.infrastructure.repository_metadata import (
    initialize_repository,
    load_repository_config,
    update_model_configuration,
)
from micro_model_agent.infrastructure.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.synthetic_evaluation import (
    SyntheticBehaviorEvaluationSuite,
    TraceBehaviorEvaluationSuite,
)
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS
from micro_model_agent.infrastructure.trace_export import (
    TraceDatasetExporter,
    validate_trace_export_examples,
)
from micro_model_agent.infrastructure.trace_review import (
    JsonlTraceReviewStore,
    TraceReview,
)
from micro_model_agent.infrastructure.trace_store import JsonlTraceStore
from micro_model_agent.infrastructure.training_artifacts import (
    FakeTrainingRunner,
    JsonTrainingArtifactStore,
    LocalFineTuningRunner,
    MinimumScorePromotionPolicy,
    SyntheticEvaluationSuite,
    load_artifact_from_training_run,
    load_evaluation_result,
    load_promotion_registry,
    record_promoted_artifact,
    write_evaluation_result,
    write_promotion_gate_result,
)
from micro_model_agent.infrastructure.workspace_staged_evaluation import (
    WorkspaceStagedEvaluationSuite,
    build_workspace_staged_review_records,
)

app = typer.Typer(help="MicroModelAgent CLI.")
dataset_app = typer.Typer(help="Dataset generation, validation, and export commands.")
train_app = typer.Typer(help="Local training commands.")
eval_app = typer.Typer(help="Evaluation commands.")
promote_app = typer.Typer(help="Promotion gate commands.")

# Sub-apps create command groups such as `micro-agent dataset validate`.
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(promote_app, name="promote")


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


def _base_model_from_adapter(adapter_path: Path | None) -> str:
    """Read the base model name from a PEFT adapter config file."""

    if adapter_path is None:
        _fail("--base-model is required when no --adapter-path is provided")

    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        _fail(f"adapter config does not exist: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        _fail(f"adapter config does not contain base_model_name_or_path: {config_path}")
    return base_model


def _resolve_loop_model_options(
    *,
    repository_root: Path,
    model: str | None,
    base_model: str | None,
    adapter_path: Path | None,
    use_adapter: bool = True,
) -> dict[str, str | Path | None]:
    """Resolve CLI loop model settings from args, env vars, and local config."""

    config = load_repository_config(repository_root) or {}
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        model_config = {}

    resolved_adapter_path = None
    if use_adapter:
        resolved_adapter_path = (
            adapter_path
            or _path_env("MICRO_MODEL_AGENT_ADAPTER_PATH")
            or _path_config_value(model_config, "adapter_path")
        )
    resolved_base_model = (
        base_model
        or os.environ.get("MICRO_MODEL_AGENT_BASE_MODEL")
        or _string_config_value(model_config, "base_model")
    )
    resolved_ollama_model = (
        model
        or os.environ.get("MICRO_MODEL_AGENT_DEFAULT_MODEL")
        or _string_config_value(model_config, "default_model")
    )
    return {
        "adapter_path": resolved_adapter_path,
        "base_model": resolved_base_model,
        "model": resolved_ollama_model,
    }


def _path_env(name: str) -> Path | None:
    """Read one path environment variable."""

    value = os.environ.get(name)
    return Path(value) if value else None


def _path_config_value(config: dict[str, Any], key: str) -> Path | None:
    """Read one path value from repository config."""

    value = _string_config_value(config, key)
    return Path(value) if value else None


def _string_config_value(config: dict[str, Any], key: str) -> str | None:
    """Read one string value from repository config."""

    value = config.get(key)
    return value if isinstance(value, str) and value else None


def _format_behavioral_eval_failures(details: dict[str, Any]) -> list[str]:
    """Return short CLI lines for failing behavioral eval examples."""

    lines: list[str] = []
    examples = details.get("examples")
    if not isinstance(examples, list):
        return lines

    for raw_example in examples:
        if not isinstance(raw_example, dict):
            continue
        score = raw_example.get("score")
        if not isinstance(score, int | float) or score >= 1.0:
            continue

        example_id = raw_example.get("example_id", "unknown")
        category = raw_example.get("category") or "uncategorized"
        errors = raw_example.get("errors")
        error_text = ""
        if isinstance(errors, list) and errors:
            error_text = f": {errors[0]}"
        lines.append(f"- {category}/{example_id} scored {score:.2f}{error_text}")
    return lines


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


def _with_evaluation_metadata(
    result: EvaluationResult,
    *,
    run_id: str,
    dataset: Path,
    examples: list[DatasetExample],
    provider_kind: str,
    model: str | None = None,
    base_model: str | None = None,
    adapter_path: Path | None = None,
) -> EvaluationResult:
    """Add report-level metadata without changing evaluator scoring."""

    return EvaluationResult(
        passed=result.passed,
        summary=result.summary,
        score=result.score,
        details={
            **result.details,
            "evaluation_metadata": {
                "run_id": run_id,
                "dataset_path": str(dataset),
                "provider": provider_kind,
                "model": model,
                "base_model": base_model,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "tool_profile": summarize_tool_profiles(
                    examples,
                    default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
                ),
            },
        },
    )


@app.command()
def init(
    repository_root: Path = typer.Option(Path("."), help="Repository root to initialize."),
    default_model: str | None = typer.Option(
        None,
        "--default-model",
        help="Optional default Ollama model name to record in local metadata.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Optional Transformers base model to record in local metadata.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Optional local PEFT adapter path to record in local metadata.",
    ),
) -> None:
    """Initialize MicroModelAgent metadata for the current repository."""

    result = initialize_repository(
        repository_root,
        default_model=default_model,
        base_model=base_model,
        adapter_path=adapter_path,
    )
    if not result.ok:
        _fail(result.error or "MicroModelAgent repository initialization failed")

    action = "Already initialized" if result.already_initialized else "Initialized"
    typer.echo(f"{action} MicroModelAgent metadata at {result.metadata_dir}")
    typer.echo(f"Config: {result.config_path}")


@app.command()
def index(
    repository_root: Path = typer.Option(Path("."), help="Repository root to index."),
    max_file_bytes: int = typer.Option(
        1_000_000,
        help="Maximum file size to index. Larger files are skipped.",
    ),
) -> None:
    """Index the current repository."""

    result = LocalLexicalIndexWriter(
        repository_root,
        max_file_bytes=max_file_bytes,
    ).write()
    typer.echo(f"Indexed {result.indexed_file_count} files into {result.index_path}")
    typer.echo(
        "Terms: "
        f"{result.unique_term_count}; bytes: {result.total_bytes}; "
        f"skipped large: {result.skipped_large_count}; "
        f"skipped binary: {result.skipped_binary_count}"
    )
    typer.echo(f"Source types: {_format_count_distribution(result.source_type_counts)}")
    typer.echo(
        "Code metadata: "
        f"symbols={result.symbol_count}; imports={result.import_count}; "
        f"test_files={result.test_file_count}"
    )


@app.command()
def task(
    prompt: str,
    patch: str = typer.Option("", help="Unified diff to use with the static local model provider."),
    repository_root: Path = typer.Option(Path("."), help="Repository root to operate on."),
    dry_run: bool = typer.Option(True, help="Validate the patch without applying it."),
    require_approval: bool = typer.Option(True, help="Require explicit approval before applying."),
    expected_changed_file: list[str] | None = typer.Option(
        None,
        "--expected-changed-file",
        help="File that the generated patch is expected to change.",
    ),
    verification_command: str | None = typer.Option(
        None, help="Allowed test command name to run after applying changes."
    ),
    test_command: list[str] | None = typer.Option(
        None,
        "--test-command",
        help="Shell-free command tokens for the verification command name.",
    ),
    label_outcome: OutcomeLabel | None = typer.Option(None, "--label"),
    label_quality: QualityLabel | None = typer.Option(None, "--quality"),
    failure_mode: list[FailureMode] | None = typer.Option(None, "--failure-mode"),
    reviewer_notes: str | None = typer.Option(None, "--reviewer-notes"),
    dataset_output: Path | None = typer.Option(
        None, help="Optional JSONL path for storing a trace-derived labeled example."
    ),
) -> None:
    """Run a coding-agent task with local fake dependencies and trace capture."""

    # Imports stay inside the command so lightweight commands start quickly and
    # optional dependencies are only loaded by commands that need them.
    from micro_model_agent.agents.coding_agent import CodingAgent, CodingAgentTask
    from micro_model_agent.application.workflows import RunAgentWorkflow, label_from_workflow_result
    from micro_model_agent.infrastructure.fake_model_provider import StaticModelProvider
    from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor
    from micro_model_agent.infrastructure.trace_store import JsonlTraceStore

    allowed_commands = {}
    if verification_command and test_command:
        # test.run selects this command by name; it never receives shell text.
        allowed_commands[verification_command] = test_command
    elif verification_command:
        _fail("--test-command is required when --verification-command is set")

    trace_store = JsonlTraceStore(
        repository_root / ".micro_model_agent" / "traces" / "workflows.jsonl"
    )
    # StaticModelProvider lets this command exercise the workflow using a patch
    # supplied on the command line instead of calling a real model.
    executor = BuiltinToolExecutor(repository_root, allowed_commands)
    agent = CodingAgent(
        model_provider=StaticModelProvider(patch),
        tool_executor=executor,
        trace_store=trace_store,
    )
    dataset_store = JsonlDatasetExampleStore(dataset_output) if dataset_output else None
    workflow = RunAgentWorkflow(agent=agent, dataset_store=dataset_store)
    task_input = CodingAgentTask(
        goal=prompt,
        dry_run=dry_run,
        require_approval=require_approval,
        expected_changed_files=expected_changed_file or [],
        verification_command_name=verification_command,
    )
    result = _run(workflow.run(task_input))

    if label_outcome or label_quality or failure_mode or reviewer_notes:
        # Labels are only useful if the caller also asked to store the resulting
        # trace-derived dataset example.
        if dataset_store is None:
            _fail("--dataset-output is required when labels are provided")
        assert dataset_store is not None
        label = label_from_workflow_result(
            result,
            outcome=label_outcome,
            quality=label_quality,
            failure_modes=tuple(failure_mode) if failure_mode else None,
            reviewer_notes=reviewer_notes,
        )
        _run(dataset_store.save(workflow.dataset_builder.build_example(result.trace, label)))

    typer.echo(result.summary)
    typer.echo(f"Trace: {result.trace_id}")
    if result.changed_files:
        typer.echo("Changed files: " + ", ".join(result.changed_files))
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def loop(
    prompt: str,
    repository_root: Path = typer.Option(Path("."), help="Repository root to operate on."),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Ollama model name. Defaults to MICRO_MODEL_AGENT_DEFAULT_MODEL.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Transformers base model for direct PEFT adapter inference.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Local PEFT adapter path for direct Transformers inference.",
    ),
    use_adapter: bool = typer.Option(
        True,
        "--adapter/--no-adapter",
        help="Load the configured PEFT adapter. Disable for base-model trace collection.",
    ),
    ollama_base_url: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama host URL. Defaults to MICRO_MODEL_AGENT_OLLAMA_BASE_URL.",
    ),
    max_new_tokens: int = typer.Option(
        384,
        min=1,
        max=4096,
        help="Maximum generated tokens per model turn.",
    ),
    max_tool_result_prompt_chars: int = typer.Option(
        12_000,
        min=1,
        help="Maximum serialized tool-result characters fed back to the model.",
    ),
    max_tool_calls: int | None = typer.Option(
        None,
        min=1,
        help="Maximum tool calls before forcing a final-response-only prompt.",
    ),
    scripted_response: list[str] | None = typer.Option(
        None,
        "--scripted-response",
        help="Scripted JSON model response. Can be passed more than once.",
    ),
    scripted_response_file: Path | None = typer.Option(
        None,
        "--scripted-response-file",
        help="JSONL file containing scripted model responses for local smoke tests.",
    ),
    available_tool: list[str] | None = typer.Option(
        None,
        "--available-tool",
        help="Allowed tool name. Defaults to all built-in tools.",
    ),
    required_tool: list[str] | None = typer.Option(
        None,
        "--required-tool",
        help="Tool that must run before the model can return a final response.",
    ),
    max_turns: int = typer.Option(8, min=1, help="Maximum model turns before failing."),
    context: str = typer.Option(
        "",
        "--context",
        help="Extra model-facing task context.",
    ),
    schema_prompt: bool = typer.Option(
        True,
        "--schema-prompt/--no-schema-prompt",
        help="Include built-in tool argument schemas in the model prompt.",
    ),
    capture_prompts: bool = typer.Option(
        False,
        "--capture-prompts",
        help="Store exact model prompts in the workflow trace for data collection review.",
    ),
    allow_no_tool_final: bool = typer.Option(
        False,
        "--allow-no-tool-final",
        help="Allow a final response before any tool call has run.",
    ),
    verification_command: str | None = typer.Option(
        None, help="Allowed test command name that test.run can select."
    ),
    test_command: list[str] | None = typer.Option(
        None,
        "--test-command",
        help="Shell-free command tokens for the verification command name.",
    ),
) -> None:
    """Run a model-driven agent loop with typed tool calls."""

    from micro_model_agent.agents.tool_loop_agent import (
        DEFAULT_TOOL_NAMES,
        ToolLoopAgent,
        ToolLoopAgentTask,
    )
    from micro_model_agent.application.ports import ModelProvider
    from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
    from micro_model_agent.infrastructure.ollama_model_provider import OllamaModelProvider
    from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor
    from micro_model_agent.infrastructure.tools.catalog import builtin_tool_prompt_schemas
    from micro_model_agent.infrastructure.trace_store import JsonlTraceStore
    from micro_model_agent.infrastructure.transformers_model_provider import (
        TransformersPeftModelProvider,
    )

    _load_dotenv()

    responses = list(scripted_response or [])
    if scripted_response_file:
        if not scripted_response_file.exists():
            _fail(f"scripted response file does not exist: {scripted_response_file}")
        responses.extend(
            line
            for line in scripted_response_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    model_options = _resolve_loop_model_options(
        repository_root=repository_root,
        model=model,
        base_model=base_model,
        adapter_path=adapter_path,
        use_adapter=use_adapter,
    )
    model_provider: ModelProvider
    if responses:
        # Scripted responses make local smoke tests deterministic.
        model_provider = ScriptedModelProvider(responses)
    elif model_options["adapter_path"] or model_options["base_model"]:
        # Direct Transformers inference is used when a base model or adapter is supplied.
        resolved_adapter_path = cast(Path | None, model_options["adapter_path"])
        resolved_base_model = cast(str | None, model_options["base_model"])
        resolved_base_model = resolved_base_model or _base_model_from_adapter(
            resolved_adapter_path
        )
        model_provider = TransformersPeftModelProvider(
            base_model=resolved_base_model,
            adapter_path=resolved_adapter_path,
            max_new_tokens=max_new_tokens,
        )
    else:
        # Otherwise, use Ollama as the local model server.
        model_name = cast(str | None, model_options["model"])
        if not model_name:
            _fail(
                "--model, MICRO_MODEL_AGENT_DEFAULT_MODEL, or configured model default is "
                "required without scripted responses"
            )
        model_provider = OllamaModelProvider(
            model_name=model_name,
            base_url=ollama_base_url or os.environ.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
            options={"num_predict": max_new_tokens},
        )

    allowed_commands = {}
    if verification_command and test_command:
        allowed_commands[verification_command] = test_command
    elif verification_command:
        _fail("--test-command is required when --verification-command is set")

    trace_store = JsonlTraceStore(
        repository_root / ".micro_model_agent" / "traces" / "workflows.jsonl"
    )
    executor = BuiltinToolExecutor(repository_root, allowed_commands)
    agent = ToolLoopAgent(
        model_provider=model_provider,
        tool_executor=executor,
        trace_store=trace_store,
    )
    result = _run(
        # The model sees the tool schemas by default, which helps it produce
        # valid JSON arguments for the selected tools.
        agent.run(
            ToolLoopAgentTask(
                goal=prompt,
                available_tools=tuple(available_tool or DEFAULT_TOOL_NAMES),
                required_tools=tuple(required_tool or ()),
                max_turns=max_turns,
                context=context,
                tool_schemas=(
                    builtin_tool_prompt_schemas(available_tool or DEFAULT_TOOL_NAMES)
                    if schema_prompt
                    else {}
                ),
                require_tool_call=not allow_no_tool_final,
                max_tool_result_prompt_chars=max_tool_result_prompt_chars,
                max_tool_calls=max_tool_calls,
                capture_prompts=capture_prompts,
                run_metadata={
                    "interface": "cli.loop",
                    "schema_prompt": schema_prompt,
                    "capture_prompts": capture_prompts,
                    "use_adapter": use_adapter,
                    "available_tools": list(available_tool or DEFAULT_TOOL_NAMES),
                    "required_tools": list(required_tool or ()),
                    "model": {
                        "model": str(model_options["model"])
                        if model_options["model"]
                        else None,
                        "base_model": model_options["base_model"],
                        "adapter_path": str(model_options["adapter_path"])
                        if model_options["adapter_path"]
                        else None,
                    },
                },
            )
        )
    )

    typer.echo(result.response)
    typer.echo(f"Trace: {result.trace_id}")
    typer.echo(f"Tool calls: {result.tool_calls_made}")
    if not result.ok:
        raise typer.Exit(1)


@app.command("serve-mcp")
def serve_mcp(
    transport: str = typer.Option(
        "stdio",
        help="MCP transport: stdio, sse, or streamable-http.",
    ),
    repository_root: Path = typer.Option(
        Path("."),
        help="Repository root used to decide whether the MCP init tool is needed.",
    ),
) -> None:
    """Serve MicroModelAgent over MCP."""

    from micro_model_agent.interfaces.mcp_server import serve

    serve(transport=transport, repository_root=repository_root)


@dataset_app.command()
def synthesize(
    count: int = typer.Option(100, min=1, help="Number of synthetic examples to generate."),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="Output JSONL path for generated synthetic examples.",
    ),
    template_dir: Path = typer.Option(
        Path("examples/synthetic-data"),
        help="Directory containing committed *.seed.jsonl templates.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Deterministic generation seed for reproducible IDs and variants.",
    ),
    balance_categories: bool = typer.Option(
        True,
        "--balance-categories/--no-balance-categories",
        help="Cycle categories evenly instead of cycling raw templates.",
    ),
    vary_scenarios: bool = typer.Option(
        True,
        "--vary-scenarios/--no-vary-scenarios",
        help="Create deterministic prompt variants while preserving validated targets.",
    ),
    include_category: list[str] | None = typer.Option(
        None,
        "--include-category",
        help="Only synthesize templates with this metadata category. Can be repeated.",
    ),
    exclude_category: list[str] | None = typer.Option(
        None,
        "--exclude-category",
        help="Skip templates with this metadata category. Can be repeated.",
    ),
) -> None:
    """Generate synthetic tool-use and workflow examples."""

    generator = SyntheticTemplateGenerator(template_dir)
    store = JsonlDatasetExampleStore(output)
    examples = _run(
        generator.generate(
            count,
            seed=seed,
            balance_categories=balance_categories,
            vary_scenarios=vary_scenarios,
            include_categories=tuple(include_category or ()),
            exclude_categories=tuple(exclude_category or ()),
        )
    )
    validation = _run(LocalDatasetValidator().validate(examples))
    if not validation.passed:
        typer.echo(validation.summary, err=True)
        for error in validation.details.get("errors", [])[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)
    _run(store.save_many(examples))
    typer.echo(f"Wrote {len(examples)} synthetic examples to {output}")
    typer.echo(
        "Categories: " + _format_count_distribution(validation.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(validation.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(validation.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(validation.details.get("tool_profile")))


@dataset_app.command()
def validate(
    path: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="JSONL dataset path to validate.",
    ),
) -> None:
    """Validate dataset records before training."""

    examples = load_dataset_examples(path)
    result = _run(LocalDatasetValidator().validate(examples))
    typer.echo(result.summary)
    typer.echo(
        "Categories: " + _format_count_distribution(result.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(result.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(result.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(result.details.get("tool_profile")))
    for error in result.details.get("errors", [])[:10]:
        typer.echo(f"- {error}")
    if not result.passed:
        raise typer.Exit(1)


@dataset_app.command("export")
def export_dataset(
    path: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="Validated dataset path to export.",
    ),
    output_format: str = typer.Option("sft-jsonl", "--format", help="Dataset export format."),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.sft.jsonl"),
        help="Output path for exported dataset.",
    ),
) -> None:
    """Export dataset records for a training backend."""

    if output_format != "sft-jsonl":
        _fail(f"Unsupported dataset export format: {output_format}")

    examples = load_dataset_examples(path)
    result = _run(LocalDatasetValidator().validate(examples))
    if not result.passed:
        typer.echo(result.summary, err=True)
        raise typer.Exit(1)

    export_sft_jsonl(output, examples)
    typer.echo(f"Exported {len(examples)} examples to {output}")


@dataset_app.command("export-traces")
def export_traces(
    trace_path: Path = typer.Option(
        Path(".micro_model_agent/traces/workflows.jsonl"),
        help="Stored workflow trace JSONL path.",
    ),
    review_path: Path = typer.Option(
        Path(".micro_model_agent/traces/reviews.jsonl"),
        help="Stored human trace review JSONL path for --label-mode reviewed.",
    ),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/trace_examples.jsonl"),
        help="Output JSONL path for trace-derived examples.",
    ),
    label_mode: str = typer.Option(
        "review",
        "--label-mode",
        help="Label mode: review or evaluation.",
    ),
    kind: DatasetExampleKind = typer.Option(
        DatasetExampleKind.REPAIR,
        "--kind",
        help="Dataset example kind to export.",
    ),
    outcome: OutcomeLabel | None = typer.Option(
        None,
        "--outcome",
        help="Only export examples with this outcome after label assignment.",
    ),
    quality: QualityLabel | None = typer.Option(
        None,
        "--quality",
        help="Only export examples with this quality after label assignment.",
    ),
    workflow_status: WorkflowStatus | None = typer.Option(
        None,
        "--workflow-status",
        help="Only export traces with this workflow status.",
    ),
    require_tool_call: bool = typer.Option(
        False,
        "--require-tool-call",
        help="Only export traces that include at least one tool call.",
    ),
    max_examples: int | None = typer.Option(
        None,
        "--max-examples",
        min=1,
        help="Optional maximum number of examples to export.",
    ),
) -> None:
    """Export stored workflow traces as redacted dataset examples."""

    store = JsonlTraceStore(trace_path)
    traces = _run(store.list())
    reviews_by_trace_id = _run(JsonlTraceReviewStore(review_path).latest_by_trace_id())
    exporter = TraceDatasetExporter()
    try:
        examples = exporter.export(
            traces,
            kind=kind,
            label_mode=label_mode,
            reviews_by_trace_id=reviews_by_trace_id,
            outcome=outcome,
            quality=quality,
            workflow_status=workflow_status,
            require_tool_call=require_tool_call,
            max_examples=max_examples,
        )
    except ValueError as exc:
        _fail(str(exc))

    errors = validate_trace_export_examples(examples)
    if errors:
        typer.echo(f"trace export found {len(errors)} error(s)", err=True)
        for error in errors[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)

    _run(JsonlDatasetExampleStore(output).save_many(examples))
    redacted_count = sum(1 for example in examples if example.metadata.get("redacted"))
    typer.echo(f"Exported {len(examples)} trace-derived examples to {output}")
    typer.echo(f"Traces read: {len(traces)}")
    typer.echo(f"Reviews read: {len(reviews_by_trace_id)}")
    typer.echo(f"Redacted examples: {redacted_count}")
    typer.echo(
        "Outcomes: "
        + _format_count_distribution(
            {
                outcome.value: sum(1 for example in examples if example.label.outcome is outcome)
                for outcome in OutcomeLabel
            }
        )
    )


@dataset_app.command("review-trace")
def review_trace(
    trace_id: str = typer.Option(..., "--trace-id", help="Stored workflow trace id to review."),
    outcome: OutcomeLabel = typer.Option(..., "--outcome", help="Human outcome label."),
    quality: QualityLabel = typer.Option(..., "--quality", help="Human quality label."),
    failure_mode: list[FailureMode] | None = typer.Option(
        None,
        "--failure-mode",
        help="Failure mode label. Can be passed more than once.",
    ),
    reviewer_notes: str | None = typer.Option(
        None,
        "--reviewer-notes",
        help="Human review notes for this trace.",
    ),
    corrected_target_json: str | None = typer.Option(
        None,
        "--corrected-target-json",
        help="Optional corrected dataset target JSON object for this trace.",
    ),
    corrected_target_file: Path | None = typer.Option(
        None,
        "--corrected-target-file",
        help="Optional file containing a corrected dataset target JSON object.",
    ),
    output: Path = typer.Option(
        Path(".micro_model_agent/traces/reviews.jsonl"),
        "--output",
        help="Append-only human trace review JSONL path.",
    ),
) -> None:
    """Record a human review label for one stored workflow trace."""

    if corrected_target_json and corrected_target_file:
        _fail("pass only one of --corrected-target-json or --corrected-target-file")

    corrected_target: dict[str, Any] | None = None
    raw_corrected_target: str | None = corrected_target_json
    if corrected_target_file:
        if not corrected_target_file.exists():
            _fail(f"corrected target file does not exist: {corrected_target_file}")
        raw_corrected_target = corrected_target_file.read_text(encoding="utf-8")
    if raw_corrected_target:
        try:
            parsed = json.loads(raw_corrected_target)
        except json.JSONDecodeError as exc:
            _fail(f"corrected target must be valid JSON: {exc}")
        if not isinstance(parsed, dict):
            _fail("corrected target must be a JSON object")
        corrected_target = parsed

    review = TraceReview(
        trace_id=trace_id,
        label=DatasetLabel(
            outcome=outcome,
            quality=quality,
            failure_modes=tuple(failure_mode or ()),
            reviewer_notes=reviewer_notes,
        ),
        corrected_target=corrected_target,
    )
    _run(JsonlTraceReviewStore(output).save(review))
    typer.echo(f"Recorded {quality.value}/{outcome.value} review for trace {trace_id}")
    typer.echo(f"Review: {review.id}")


@dataset_app.command("relabel")
def relabel_dataset(
    path: Path = typer.Option(..., help="Input JSONL dataset path to relabel."),
    output: Path = typer.Option(..., help="Output JSONL path for relabeled examples."),
    trace_id: str | None = typer.Option(
        None,
        "--trace-id",
        help="Only relabel the example with this metadata.trace_id.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Only relabel examples with this exact source.",
    ),
    input_outcome: OutcomeLabel | None = typer.Option(
        None,
        "--input-outcome",
        help="Only relabel examples currently carrying this outcome.",
    ),
    input_quality: QualityLabel | None = typer.Option(
        None,
        "--input-quality",
        help="Only relabel examples currently carrying this quality.",
    ),
    outcome: OutcomeLabel | None = typer.Option(
        None,
        "--outcome",
        help="New outcome label for matched examples.",
    ),
    quality: QualityLabel | None = typer.Option(
        None,
        "--quality",
        help="New quality label for matched examples.",
    ),
    failure_mode: list[FailureMode] | None = typer.Option(
        None,
        "--failure-mode",
        help="Replacement failure mode label. Can be passed more than once.",
    ),
    reviewer_notes: str | None = typer.Option(
        None,
        "--reviewer-notes",
        help="Replacement reviewer notes for matched examples.",
    ),
) -> None:
    """Relabel reviewed dataset examples and write a curated JSONL file."""

    if not any([outcome, quality, failure_mode, reviewer_notes is not None]):
        _fail("At least one label update is required")

    examples = load_dataset_examples(path)
    relabeled, changed = relabel_examples(
        examples,
        trace_id=trace_id,
        source=source,
        input_outcome=input_outcome,
        input_quality=input_quality,
        outcome=outcome,
        quality=quality,
        failure_modes=tuple(failure_mode) if failure_mode is not None else None,
        reviewer_notes=reviewer_notes,
    )
    _run(JsonlDatasetExampleStore(output).save_many(relabeled))
    validation = _run(LocalDatasetValidator().validate(relabeled))
    typer.echo(f"Relabeled {changed} of {len(examples)} examples to {output}")
    typer.echo(validation.summary)
    typer.echo(
        "Categories: " + _format_count_distribution(validation.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(validation.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(validation.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(validation.details.get("tool_profile")))
    for error in validation.details.get("errors", [])[:10]:
        typer.echo(f"- {error}")


@dataset_app.command("merge")
def merge_dataset(
    input_path: list[Path] = typer.Option(
        ...,
        "--input",
        help="Input JSONL dataset path. Pass more than once.",
    ),
    output: Path = typer.Option(..., help="Output JSONL path for the merged dataset."),
    deduplicate_by: str = typer.Option(
        "source",
        "--deduplicate-by",
        help="Dedupe key: source or id.",
    ),
    validate_training_ready: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate the merged dataset before reporting success.",
    ),
) -> None:
    """Merge dataset JSONL files with simple deduplication."""

    if deduplicate_by not in {"source", "id"}:
        _fail(f"Unsupported dedupe key: {deduplicate_by}")

    datasets = [load_dataset_examples(path) for path in input_path]
    merged, skipped = merge_datasets(
        datasets,
        deduplicate_by=cast(DeduplicateBy, deduplicate_by),
    )
    validation = _run(LocalDatasetValidator().validate(merged))
    if validate_training_ready and not validation.passed:
        typer.echo(validation.summary, err=True)
        for error in validation.details.get("errors", [])[:10]:
            typer.echo(f"- {error}", err=True)
        raise typer.Exit(1)

    _run(JsonlDatasetExampleStore(output).save_many(merged))
    typer.echo(f"Merged {len(merged)} examples to {output}")
    typer.echo(f"Skipped duplicates: {skipped}")
    typer.echo(validation.summary)
    typer.echo(
        "Categories: " + _format_count_distribution(validation.details.get("category_counts"))
    )
    typer.echo("Kinds: " + _format_count_distribution(validation.details.get("kind_counts")))
    typer.echo("Outcomes: " + _format_count_distribution(validation.details.get("outcome_counts")))
    typer.echo("Tool profile: " + _format_tool_profile(validation.details.get("tool_profile")))


@train_app.command("synthetic")
def train_synthetic(
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Local or Hugging Face base model id.",
    ),
    dataset: Path = typer.Option(
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
        help="Validated synthetic dataset path.",
    ),
    output_dir: Path = typer.Option(
        Path(".micro_model_agent/training/runs/latest"),
        help="Training run output directory.",
    ),
    dry_run: bool = typer.Option(True, help="Validate config and write a dry-run artifact."),
    max_steps: int = typer.Option(20, min=1, help="Maximum optimizer steps for real training."),
    batch_size: int = typer.Option(1, min=1, help="Per-device train batch size."),
    gradient_accumulation_steps: int = typer.Option(
        4,
        min=1,
        help="Gradient accumulation steps for real training.",
    ),
    learning_rate: float = typer.Option(2e-4, min=0.0, help="Learning rate for real training."),
    max_seq_length: int = typer.Option(1024, min=128, help="Maximum tokenized sequence length."),
    lora_r: int = typer.Option(16, min=1, help="LoRA rank for real training."),
    lora_alpha: int = typer.Option(32, min=1, help="LoRA alpha for real training."),
    lora_dropout: float = typer.Option(0.05, min=0.0, max=1.0, help="LoRA dropout."),
) -> None:
    """Run or dry-run local synthetic-data fine-tuning."""

    _load_dotenv()

    examples = load_dataset_examples(dataset)
    validation = _run(LocalDatasetValidator().validate(examples))
    if not validation.passed:
        typer.echo(validation.summary, err=True)
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_dataset = output_dir / "synthetic.sft.jsonl"
    # Training backends consume SFT JSONL, so export a run-local copy first.
    export_sft_jsonl(training_dataset, examples)
    source_dataset_sha256 = dataset_file_sha256(dataset)
    training_dataset_sha256 = dataset_file_sha256(training_dataset)

    config = TrainingConfig(
        base_model=base_model,
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dry_run=dry_run,
        parameters={
            "dataset_path": str(training_dataset),
            "source_dataset_path": str(dataset),
            "source_dataset_sha256": source_dataset_sha256,
            "training_dataset_sha256": training_dataset_sha256,
            "example_count": len(examples),
            "dataset_tool_profile": summarize_tool_profiles(examples),
            "max_seq_length": max_seq_length,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    )
    # dry_run uses the fake runner; real training uses the hardware-dependent runner.
    runner = FakeTrainingRunner() if dry_run else LocalFineTuningRunner()
    run = _run(runner.run(config))
    artifact_store = JsonTrainingArtifactStore(Path(".micro_model_agent/training"))
    for artifact in run.artifacts:
        _run(artifact_store.save(artifact))

    typer.echo(
        f"Training run {run.id} completed with status {run.status.value}; "
        f"metadata written to {output_dir}"
    )
    if run.status.value == "failed":
        raise typer.Exit(1)


@eval_app.command("synthetic")
def eval_synthetic(
    run_id: str = typer.Option("latest", help="Training run id or alias to evaluate."),
    dataset: Path = typer.Option(
        Path("examples/synthetic-data/held-out.behavior.jsonl"),
        help="Held-out synthetic JSONL dataset to evaluate against.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Ollama model name to evaluate.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Transformers base model for direct PEFT adapter evaluation.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Local PEFT adapter path for direct Transformers evaluation.",
    ),
    ollama_base_url: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama host URL. Defaults to MICRO_MODEL_AGENT_OLLAMA_BASE_URL.",
    ),
    max_new_tokens: int = typer.Option(
        384,
        min=1,
        max=4096,
        help="Maximum generated tokens per evaluation example.",
    ),
    max_examples: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on evaluated examples.",
    ),
    pass_threshold: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum average behavioral score required to pass.",
    ),
    scripted_response: list[str] | None = typer.Option(
        None,
        "--scripted-response",
        help="Scripted JSON model response. Can be passed more than once.",
    ),
    scripted_response_file: Path | None = typer.Option(
        None,
        "--scripted-response-file",
        help="JSONL file containing scripted model responses for evaluation tests.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Evaluation report path. Defaults to <run>/evaluation.json.",
    ),
) -> None:
    """Evaluate a model or training run against synthetic behavior examples."""

    from micro_model_agent.application.ports import ModelProvider
    from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
    from micro_model_agent.infrastructure.ollama_model_provider import OllamaModelProvider
    from micro_model_agent.infrastructure.transformers_model_provider import (
        TransformersPeftModelProvider,
    )

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    _load_dotenv()
    responses = list(scripted_response or [])
    if scripted_response_file:
        if not scripted_response_file.exists():
            _fail(f"scripted response file does not exist: {scripted_response_file}")
        responses.extend(
            line
            for line in scripted_response_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    provider: ModelProvider | None = None
    provider_kind = "metadata"
    resolved_base_model: str | None = base_model
    resolved_adapter_path: Path | None = adapter_path
    if responses:
        provider = ScriptedModelProvider(responses)
        provider_kind = "scripted"
    elif adapter_path or base_model:
        resolved_base_model = base_model or _base_model_from_adapter(adapter_path)
        provider = TransformersPeftModelProvider(
            base_model=resolved_base_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
        provider_kind = "transformers_peft"
    elif model:
        provider = OllamaModelProvider(
            model_name=model,
            base_url=ollama_base_url or os.environ.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
            options={"num_predict": max_new_tokens},
        )
        provider_kind = "ollama"
    else:
        artifact = load_artifact_from_training_run(run_dir)
        artifact_path = Path(artifact.path)
        if artifact_path.exists():
            provider = TransformersPeftModelProvider(
                base_model=artifact.base_model,
                adapter_path=artifact_path,
                max_new_tokens=max_new_tokens,
            )
            provider_kind = "training_artifact"
            resolved_base_model = artifact.base_model
            resolved_adapter_path = artifact_path

    if provider is not None:
        examples = load_dataset_examples(dataset)
        if max_examples is not None:
            examples = examples[:max_examples]
        result = _run(
            SyntheticBehaviorEvaluationSuite(pass_threshold=pass_threshold).evaluate_model(
                provider,
                examples,
            )
        )
        result = _with_evaluation_metadata(
            result,
            run_id=run_id,
            dataset=dataset,
            examples=examples,
            provider_kind=provider_kind,
            model=model,
            base_model=resolved_base_model,
            adapter_path=resolved_adapter_path,
        )
        report_path = write_evaluation_result(run_dir, result, output)
        typer.echo(f"{result.summary}; report written to {report_path}")
        if not result.passed:
            failure_lines = _format_behavioral_eval_failures(result.details)
            if failure_lines:
                typer.echo("Failures:", err=True)
                for line in failure_lines:
                    typer.echo(line, err=True)
            raise typer.Exit(1)
        return

    # Dry-run training artifacts do not contain runnable model weights. Keep the
    # old metadata smoke gate for those cases, but mark it clearly in the report.
    artifact = load_artifact_from_training_run(run_dir)
    result = _run(SyntheticEvaluationSuite().evaluate_artifact(artifact))
    dataset_examples = load_dataset_examples(dataset)
    if max_examples is not None:
        dataset_examples = dataset_examples[:max_examples]
    result = _with_evaluation_metadata(
        result,
        run_id=run_id,
        dataset=dataset,
        examples=dataset_examples,
        provider_kind=provider_kind,
        model=model,
        base_model=artifact.base_model,
        adapter_path=Path(artifact.path),
    )
    report_path = write_evaluation_result(run_dir, result, output)
    typer.echo(f"{result.summary}; report written to {report_path}")
    if not result.passed:
        raise typer.Exit(1)


@eval_app.command("traces")
def eval_traces(
    run_id: str = typer.Option("latest", help="Training run id or alias to evaluate."),
    dataset: Path = typer.Option(
        Path("examples/trace-data/held-out.trace.jsonl"),
        help="Held-out trace-derived JSONL dataset to evaluate against.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Ollama model name to evaluate.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Transformers base model for direct PEFT adapter evaluation.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Local PEFT adapter path for direct Transformers evaluation.",
    ),
    ollama_base_url: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama host URL. Defaults to MICRO_MODEL_AGENT_OLLAMA_BASE_URL.",
    ),
    max_new_tokens: int = typer.Option(
        512,
        min=1,
        max=4096,
        help="Maximum generated tokens per evaluation example.",
    ),
    max_examples: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on evaluated examples.",
    ),
    pass_threshold: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum average trace behavior score required to pass.",
    ),
    scripted_response: list[str] | None = typer.Option(
        None,
        "--scripted-response",
        help="Scripted JSON model response. Can be passed more than once.",
    ),
    scripted_response_file: Path | None = typer.Option(
        None,
        "--scripted-response-file",
        help="JSONL file containing scripted model responses for evaluation tests.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Evaluation report path. Defaults to <run>/evaluation.json.",
    ),
) -> None:
    """Evaluate a model or training run against held-out trace-derived examples."""

    from micro_model_agent.application.ports import ModelProvider
    from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
    from micro_model_agent.infrastructure.ollama_model_provider import OllamaModelProvider
    from micro_model_agent.infrastructure.transformers_model_provider import (
        TransformersPeftModelProvider,
    )

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    _load_dotenv()
    responses = list(scripted_response or [])
    if scripted_response_file:
        if not scripted_response_file.exists():
            _fail(f"scripted response file does not exist: {scripted_response_file}")
        responses.extend(
            line
            for line in scripted_response_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    provider: ModelProvider | None = None
    provider_kind = "metadata"
    resolved_base_model: str | None = base_model
    resolved_adapter_path: Path | None = adapter_path
    if responses:
        provider = ScriptedModelProvider(responses)
        provider_kind = "scripted"
    elif adapter_path or base_model:
        resolved_base_model = base_model or _base_model_from_adapter(adapter_path)
        provider = TransformersPeftModelProvider(
            base_model=resolved_base_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
        provider_kind = "transformers_peft"
    elif model:
        provider = OllamaModelProvider(
            model_name=model,
            base_url=ollama_base_url or os.environ.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
            options={"num_predict": max_new_tokens},
        )
        provider_kind = "ollama"
    else:
        artifact = load_artifact_from_training_run(run_dir)
        artifact_path = Path(artifact.path)
        if artifact_path.exists():
            provider = TransformersPeftModelProvider(
                base_model=artifact.base_model,
                adapter_path=artifact_path,
                max_new_tokens=max_new_tokens,
            )
            provider_kind = "training_artifact"
            resolved_base_model = artifact.base_model
            resolved_adapter_path = artifact_path

    if provider is None:
        _fail("trace eval requires a runnable model, adapter, or scripted response")

    examples = load_dataset_examples(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]
    result = _run(
        TraceBehaviorEvaluationSuite(pass_threshold=pass_threshold).evaluate_model(
            provider,
            examples,
        )
    )
    result = _with_evaluation_metadata(
        result,
        run_id=run_id,
        dataset=dataset,
        examples=examples,
        provider_kind=provider_kind,
        model=model,
        base_model=resolved_base_model,
        adapter_path=resolved_adapter_path,
    )
    report_path = write_evaluation_result(run_dir, result, output)
    typer.echo(f"{result.summary}; report written to {report_path}")
    if not result.passed:
        raise typer.Exit(1)


@eval_app.command("workspace-staged")
def eval_workspace_staged(
    run_id: str = typer.Option("latest", help="Training run id or alias to evaluate."),
    dataset: Path = typer.Option(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl"),
        help="Held-out staged workspace JSONL dataset to evaluate against.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Ollama model name to evaluate.",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Transformers base model for direct PEFT adapter evaluation.",
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Local PEFT adapter path for direct Transformers evaluation.",
    ),
    ollama_base_url: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama host URL. Defaults to MICRO_MODEL_AGENT_OLLAMA_BASE_URL.",
    ),
    max_new_tokens: int = typer.Option(
        1024,
        min=1,
        max=4096,
        help="Maximum generated tokens per evaluation example.",
    ),
    max_examples: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on evaluated examples.",
    ),
    pass_threshold: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum average staged workspace score required to pass.",
    ),
    rubric_version: str = typer.Option(
        "legacy",
        "--rubric-version",
        help="Staged rubric version: legacy, v2, or auto.",
    ),
    scripted_response: list[str] | None = typer.Option(
        None,
        "--scripted-response",
        help="Scripted JSON model response. Can be passed more than once.",
    ),
    scripted_response_file: Path | None = typer.Option(
        None,
        "--scripted-response-file",
        help="JSONL file containing scripted model responses for evaluation tests.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Evaluation report path. Defaults to <run>/evaluation.json.",
    ),
) -> None:
    """Evaluate staged workspace reasoning without applying patches."""

    from micro_model_agent.application.ports import ModelProvider
    from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
    from micro_model_agent.infrastructure.ollama_model_provider import OllamaModelProvider
    from micro_model_agent.infrastructure.transformers_model_provider import (
        TransformersPeftModelProvider,
    )

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    _load_dotenv()
    responses = list(scripted_response or [])
    if scripted_response_file:
        if not scripted_response_file.exists():
            _fail(f"scripted response file does not exist: {scripted_response_file}")
        responses.extend(
            line
            for line in scripted_response_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    provider: ModelProvider | None = None
    provider_kind = "metadata"
    resolved_base_model: str | None = base_model
    resolved_adapter_path: Path | None = adapter_path
    if responses:
        provider = ScriptedModelProvider(responses)
        provider_kind = "scripted"
    elif adapter_path or base_model:
        resolved_base_model = base_model or _base_model_from_adapter(adapter_path)
        provider = TransformersPeftModelProvider(
            base_model=resolved_base_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
        provider_kind = "transformers_peft"
    elif model:
        provider = OllamaModelProvider(
            model_name=model,
            base_url=ollama_base_url or os.environ.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
            options={"num_predict": max_new_tokens},
        )
        provider_kind = "ollama"
    else:
        artifact = load_artifact_from_training_run(run_dir)
        artifact_path = Path(artifact.path)
        if artifact_path.exists():
            provider = TransformersPeftModelProvider(
                base_model=artifact.base_model,
                adapter_path=artifact_path,
                max_new_tokens=max_new_tokens,
            )
            provider_kind = "training_artifact"
            resolved_base_model = artifact.base_model
            resolved_adapter_path = artifact_path

    if provider is None:
        _fail("workspace-staged eval requires a runnable model, adapter, or scripted response")
    if rubric_version not in {"legacy", "v2", "auto"}:
        _fail(f"Unsupported workspace-staged rubric version: {rubric_version}")

    examples = load_dataset_examples(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]
    result = _run(
        WorkspaceStagedEvaluationSuite(
            pass_threshold=pass_threshold,
            rubric_version=rubric_version,
        ).evaluate_model(provider, examples)
    )
    result = _with_evaluation_metadata(
        result,
        run_id=run_id,
        dataset=dataset,
        examples=examples,
        provider_kind=provider_kind,
        model=model,
        base_model=resolved_base_model,
        adapter_path=resolved_adapter_path,
    )
    report_path = write_evaluation_result(run_dir, result, output)
    typer.echo(f"{result.summary}; report written to {report_path}")
    if not result.passed:
        raise typer.Exit(1)


@eval_app.command("review-workspace-staged")
def review_workspace_staged(
    dataset: Path = typer.Option(
        Path("examples/workspace-eval/held-out.workspace-staged.jsonl"),
        help="Staged workspace scenario JSONL dataset.",
    ),
    report: list[Path] | None = typer.Option(
        None,
        "--report",
        help="Staged workspace evaluation JSON report. Can be passed more than once.",
    ),
    output: Path = typer.Option(
        Path(".micro_model_agent/datasets/workspace_staged_review_queue.jsonl"),
        "--output",
        help="Output JSONL review queue path.",
    ),
    simple_failure_threshold: float = typer.Option(
        0.4,
        min=0.0,
        max=1.0,
        help="Auto-reject examples whose best model score is at or below this value.",
    ),
    auto_accept_threshold: float = typer.Option(
        0.95,
        min=0.0,
        max=1.0,
        help="Mark examples as auto-accept candidates at or above this best score.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Prompt for a human decision and notes for each review record.",
    ),
) -> None:
    """Build or complete a staged workspace review queue from eval reports."""

    report_paths = list(report or [])
    if not report_paths:
        _fail("at least one --report is required")

    examples = load_dataset_examples(dataset)
    reports = [(path, load_evaluation_result(path)) for path in report_paths]
    records = build_workspace_staged_review_records(
        examples=examples,
        reports=reports,
        simple_failure_threshold=simple_failure_threshold,
        auto_accept_threshold=auto_accept_threshold,
    )

    if interactive:
        for index, record in enumerate(records, start=1):
            typer.echo("")
            typer.echo(f"[{index}/{len(records)}] {record['category']}: {record['goal']}")
            typer.echo(f"Auto triage: {record['auto_triage']['decision']}")
            typer.echo(f"Reason: {record['auto_triage']['reason']}")
            for result_record in record["model_results"]:
                score = result_record["score"]
                score_text = "none" if score is None else f"{score:.2f}"
                typer.echo(
                    f"- {result_record['run_id'] or result_record['report_path']}: "
                    f"score={score_text} provider={result_record['provider']}"
                )
            decision = typer.prompt(
                "Decision [accepted/rejected/needs_review/skip]",
                default="needs_review",
            )
            if decision not in {"accepted", "rejected", "needs_review", "skip"}:
                _fail(f"unsupported review decision: {decision}")
            notes = typer.prompt("Notes", default="")
            record["review"] = {
                "decision": decision,
                "notes": notes or None,
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")

    counts: dict[str, int] = {}
    for record in records:
        decision = str(record["auto_triage"]["decision"])
        counts[decision] = counts.get(decision, 0) + 1
    typer.echo(f"Wrote {len(records)} staged workspace review record(s) to {output}")
    typer.echo("Auto triage: " + _format_count_distribution(counts))


@eval_app.command("compare")
def eval_compare(
    baseline_report: Path = typer.Option(
        ...,
        "--baseline-report",
        help="Persisted evaluation JSON report for the base model.",
    ),
    adapter_report: Path = typer.Option(
        ...,
        "--adapter-report",
        help="Persisted evaluation JSON report for the trained adapter.",
    ),
    minimum_score_delta: float = typer.Option(
        0.0,
        min=0.0,
        help="Minimum adapter score improvement over baseline.",
    ),
    minimum_metric_delta: list[str] | None = typer.Option(
        None,
        "--minimum-metric-delta",
        help="Required metric improvement as metric_name=delta. Can be passed more than once.",
    ),
    require_adapter_passed: bool = typer.Option(
        True,
        "--require-adapter-passed/--allow-failing-adapter",
        help="Require the adapter report itself to pass before comparing improvements.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Comparison report path. Defaults to <adapter-report>.comparison.json.",
    ),
) -> None:
    """Compare a baseline evaluation report with a trained adapter report."""

    thresholds = _parse_metric_delta_options(minimum_metric_delta or [])
    baseline = load_evaluation_result(baseline_report)
    adapter = load_evaluation_result(adapter_report)
    comparison = compare_evaluation_results(
        baseline,
        adapter,
        minimum_score_delta=minimum_score_delta,
        minimum_metric_deltas=thresholds,
        require_adapter_passed=require_adapter_passed,
    )

    report_path = output or adapter_report.with_suffix(".comparison.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(comparison.as_record(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    typer.echo(f"{comparison.summary}; report written to {report_path}")
    if not comparison.passed:
        for error in comparison.errors:
            typer.echo(error, err=True)
        raise typer.Exit(1)


def _parse_metric_delta_options(values: list[str]) -> dict[str, float]:
    """Parse metric threshold CLI values."""

    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            _fail(f"metric delta must use metric_name=delta: {value}")
        name, raw_delta = value.split("=", 1)
        metric_name = name.strip()
        if not metric_name:
            _fail(f"metric delta must include a metric name: {value}")
        try:
            delta = float(raw_delta)
        except ValueError:
            _fail(f"metric delta must include a numeric threshold: {value}")
        if delta < 0.0:
            _fail(f"metric delta must be non-negative: {value}")
        thresholds[metric_name] = delta
    return thresholds


@promote_app.command("gate")
def promote_gate(
    run_id: str = typer.Option("latest", help="Training run id or direct run directory path."),
    evaluation_report: list[Path] | None = typer.Option(
        None,
        "--evaluation-report",
        help="Evaluation JSON report to require. Can be passed more than once.",
    ),
    minimum_score: float = typer.Option(
        0.8,
        min=0.0,
        max=1.0,
        help="Minimum score each evaluation report must meet.",
    ),
) -> None:
    """Gate promotion on one or more persisted evaluation reports."""

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    artifact = load_artifact_from_training_run(run_dir)
    report_paths = list(evaluation_report or [run_dir / "evaluation.json"])
    if not report_paths:
        _fail("at least one evaluation report is required")

    policy = MinimumScorePromotionPolicy(minimum_score=minimum_score)
    evaluated_reports = []
    for report_path in report_paths:
        evaluation = load_evaluation_result(report_path)
        can_promote = _run(policy.can_promote(artifact, evaluation))
        evaluated_reports.append((report_path, evaluation, can_promote))

    promoted = all(can_promote for _, _, can_promote in evaluated_reports)
    output = write_promotion_gate_result(
        run_dir,
        promoted=promoted,
        minimum_score=minimum_score,
        evaluation_reports=evaluated_reports,
    )

    if promoted:
        typer.echo(f"Promotion gate passed for {artifact.name}; report written to {output}")
        return

    typer.echo(f"Promotion gate blocked for {artifact.name}; report written to {output}", err=True)
    for report_path, evaluation, can_promote in evaluated_reports:
        if not can_promote:
            score = "none" if evaluation.score is None else f"{evaluation.score:.2f}"
            typer.echo(
                f"{report_path}: passed={evaluation.passed} score={score} "
                f"summary={evaluation.summary}",
                err=True,
            )
    raise typer.Exit(1)


@promote_app.command("record")
def promote_record(
    run_id: str = typer.Option("latest", help="Training run id or direct run directory path."),
    promotion_report: Path | None = typer.Option(
        None,
        "--promotion-report",
        help="Passing promotion report path. Defaults to <run>/promotion.json.",
    ),
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
    reviewer_notes: str | None = typer.Option(
        None,
        "--reviewer-notes",
        help="Human review note to store with the registry entry.",
    ),
    approved_by: str | None = typer.Option(
        None,
        "--approved-by",
        help="Reviewer or process that approved this artifact.",
    ),
) -> None:
    """Record a gate-passing artifact in the local promotion registry."""

    run_dir = Path(run_id)
    if not run_dir.exists():
        run_dir = Path(".micro_model_agent/training/runs") / run_id

    artifact = load_artifact_from_training_run(run_dir)
    report_path = promotion_report or run_dir / "promotion.json"
    entry = record_promoted_artifact(
        registry,
        artifact=artifact,
        run_dir=run_dir,
        promotion_report_path=report_path,
        reviewer_notes=reviewer_notes,
        approved_by=approved_by,
    )
    typer.echo(
        f"Recorded promoted artifact {entry.artifact_name} "
        f"({entry.artifact_id}) in {registry}"
    )


@promote_app.command("list")
def promote_list(
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
) -> None:
    """List locally recorded promoted artifacts."""

    entries = load_promotion_registry(registry)
    if not entries:
        typer.echo(f"No promoted artifacts recorded in {registry}")
        return

    for entry in entries:
        typer.echo(
            f"{entry.artifact_name} {entry.artifact_id} "
            f"score>={entry.minimum_score:.2f} path={entry.artifact_path}"
        )


@promote_app.command("select")
def promote_select(
    artifact_id: str = typer.Option(
        ...,
        "--artifact-id",
        help="Promoted artifact id from the local registry.",
    ),
    repository_root: Path = typer.Option(
        Path("."),
        "--repository-root",
        help="Repository root whose local model defaults should be updated.",
    ),
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Required explicit confirmation before changing local defaults.",
    ),
) -> None:
    """Select a recorded promoted artifact as the repository-local default adapter."""

    if not confirm:
        _fail("Selecting a promoted adapter changes local defaults; rerun with --confirm")

    entries = load_promotion_registry(registry)
    entry = next((item for item in entries if str(item.artifact_id) == artifact_id), None)
    if entry is None:
        _fail(f"promoted artifact id not found in {registry}: {artifact_id}")

    result = update_model_configuration(
        repository_root,
        base_model=entry.base_model,
        adapter_path=entry.artifact_path,
        selected_promotion={
            "artifact_id": str(entry.artifact_id),
            "artifact_name": entry.artifact_name,
            "promotion_report_path": entry.promotion_report_path,
            "registry_path": str(registry),
            "minimum_score": entry.minimum_score,
            "approved_by": entry.approved_by,
            "created_at": entry.created_at.isoformat(),
        },
    )
    if not result.ok:
        _fail(result.error or "failed to update MicroModelAgent config")

    typer.echo(
        f"Selected promoted adapter {entry.artifact_name} "
        f"({entry.artifact_id}) in {result.config_path}"
    )


@promote_app.command("package-ollama")
def promote_package_ollama(
    artifact_id: str = typer.Option(
        ...,
        "--artifact-id",
        help="Promoted artifact id from the local registry.",
    ),
    model_name: str = typer.Option(
        ...,
        "--model-name",
        help="Ollama model name to create, such as micro-agent-proof:qwen.",
    ),
    registry: Path = typer.Option(
        Path(".micro_model_agent/training/promoted_models.jsonl"),
        "--registry",
        help="Local JSONL registry path for approved artifacts.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for the generated Modelfile and package manifest.",
    ),
    ollama_base_model: str | None = typer.Option(
        None,
        "--ollama-base-model",
        help="Ollama FROM value. Defaults to the artifact training base model.",
    ),
    create: bool = typer.Option(
        False,
        "--create",
        help="Run `ollama create` after writing the Modelfile.",
    ),
) -> None:
    """Package a recorded promoted adapter for Ollama with a generated Modelfile."""

    entries = load_promotion_registry(registry)
    entry = next((item for item in entries if str(item.artifact_id) == artifact_id), None)
    if entry is None:
        _fail(f"promoted artifact id not found in {registry}: {artifact_id}")

    package_dir = output_dir or Path(".micro_model_agent/training/ollama") / _path_safe_name(
        model_name
    )
    result = package_promoted_adapter_for_ollama(
        entry=entry,
        model_name=model_name,
        output_dir=package_dir,
        ollama_base_model=ollama_base_model,
        create=create,
    )

    typer.echo(f"Wrote Ollama Modelfile to {result.modelfile_path}")
    typer.echo(f"Wrote package manifest to {result.manifest_path}")
    typer.echo("Command: " + " ".join(result.command))
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if create:
        if result.created:
            typer.echo(f"Created Ollama model {result.model_name}")
            return
        typer.echo(f"ollama create failed with exit code {result.return_code}", err=True)
        if result.stderr:
            typer.echo(result.stderr, err=True)
        raise typer.Exit(1)


def _path_safe_name(value: str) -> str:
    """Return a conservative directory name for a model tag."""

    safe = "".join(character if character.isalnum() else "-" for character in value.lower())
    return "-".join(part for part in safe.split("-") if part) or "ollama-model"


if __name__ == "__main__":
    app()
