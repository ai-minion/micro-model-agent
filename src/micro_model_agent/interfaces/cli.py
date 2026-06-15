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
from typing import Any, Never

import typer

from micro_model_agent.domain.datasets import FailureMode, OutcomeLabel, QualityLabel
from micro_model_agent.domain.training import TrainingConfig
from micro_model_agent.infrastructure.dataset_store import (
    JsonlDatasetExampleStore,
    load_dataset_examples,
)
from micro_model_agent.infrastructure.dataset_validation import (
    LocalDatasetValidator,
    export_sft_jsonl,
)
from micro_model_agent.infrastructure.repository_metadata import initialize_repository
from micro_model_agent.infrastructure.synthetic_data import SyntheticTemplateGenerator
from micro_model_agent.infrastructure.synthetic_evaluation import SyntheticBehaviorEvaluationSuite
from micro_model_agent.infrastructure.training_artifacts import (
    FakeTrainingRunner,
    JsonTrainingArtifactStore,
    LocalFineTuningRunner,
    SyntheticEvaluationSuite,
    load_artifact_from_training_run,
    write_evaluation_result,
)

app = typer.Typer(help="MicroModelAgent CLI.")
dataset_app = typer.Typer(help="Dataset generation, validation, and export commands.")
train_app = typer.Typer(help="Local training commands.")
eval_app = typer.Typer(help="Evaluation commands.")

# Sub-apps create command groups such as `micro-agent dataset validate`.
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")


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
def index() -> None:
    """Index the current repository."""
    typer.echo("Repository indexing is not implemented yet.")


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

    model_provider: ModelProvider
    if responses:
        # Scripted responses make local smoke tests deterministic.
        model_provider = ScriptedModelProvider(responses)
    elif adapter_path or base_model:
        # Direct Transformers inference is used when a base model or adapter is supplied.
        resolved_base_model = base_model or _base_model_from_adapter(adapter_path)
        model_provider = TransformersPeftModelProvider(
            base_model=resolved_base_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
    else:
        # Otherwise, use Ollama as the local model server.
        model_name = model or os.environ.get("MICRO_MODEL_AGENT_DEFAULT_MODEL")
        if not model_name:
            _fail(
                "--model or MICRO_MODEL_AGENT_DEFAULT_MODEL is required without scripted responses"
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
) -> None:
    """Generate synthetic tool-use and workflow examples."""

    generator = SyntheticTemplateGenerator(template_dir)
    store = JsonlDatasetExampleStore(output)
    examples = _run(generator.generate(count))
    _run(store.save_many(examples))
    typer.echo(f"Wrote {len(examples)} synthetic examples to {output}")


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
            "example_count": len(examples),
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
        Path(".micro_model_agent/datasets/synthetic_seed.jsonl"),
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
    if responses:
        provider = ScriptedModelProvider(responses)
    elif adapter_path or base_model:
        resolved_base_model = base_model or _base_model_from_adapter(adapter_path)
        provider = TransformersPeftModelProvider(
            base_model=resolved_base_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
    elif model:
        provider = OllamaModelProvider(
            model_name=model,
            base_url=ollama_base_url or os.environ.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
            options={"num_predict": max_new_tokens},
        )
    else:
        artifact = load_artifact_from_training_run(run_dir)
        artifact_path = Path(artifact.path)
        if artifact_path.exists():
            provider = TransformersPeftModelProvider(
                base_model=artifact.base_model,
                adapter_path=artifact_path,
                max_new_tokens=max_new_tokens,
            )

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
        output = write_evaluation_result(run_dir, result)
        typer.echo(f"{result.summary}; report written to {output}")
        if not result.passed:
            raise typer.Exit(1)
        return

    # Dry-run training artifacts do not contain runnable model weights. Keep the
    # old metadata smoke gate for those cases, but mark it clearly in the report.
    artifact = load_artifact_from_training_run(run_dir)
    result = _run(SyntheticEvaluationSuite().evaluate_artifact(artifact))
    output = write_evaluation_result(run_dir, result)
    typer.echo(f"{result.summary}; report written to {output}")
    if not result.passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
