"""Model-driven loop CLI command."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import typer

from micro_model_agent.infrastructure.composition import (
    RuntimeModelOptions,
    build_builtin_tool_executor,
    build_model_provider,
    workflow_trace_store,
)
from micro_model_agent.interfaces.cli.common import (
    _fail,
    _load_dotenv,
    _read_scripted_responses,
    _resolve_loop_model_options,
    _run,
)


def register_loop_command(app: typer.Typer) -> None:
    """Register the model-driven loop command."""

    app.command()(loop)


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

    from micro_model_agent.agents.tool_loop_agent import ToolLoopAgent
    from micro_model_agent.application.tool_loop import (
        DEFAULT_TOOL_NAMES,
        RunToolLoopRequest,
        RunToolLoopWorkflow,
    )
    from micro_model_agent.infrastructure.tools.catalog import builtin_tool_prompt_schemas

    _load_dotenv()

    responses = _read_scripted_responses(scripted_response, scripted_response_file)

    model_options = _resolve_loop_model_options(
        repository_root=repository_root,
        model=model,
        base_model=base_model,
        adapter_path=adapter_path,
        use_adapter=use_adapter,
    )
    if not responses and not (
        model_options["adapter_path"] or model_options["base_model"] or model_options["model"]
    ):
        _fail(
            "--model, MICRO_MODEL_AGENT_DEFAULT_MODEL, or configured model default is "
            "required without scripted responses"
        )
    try:
        model_provider = build_model_provider(
            options=RuntimeModelOptions(
                model=cast(str | None, model_options["model"]),
                base_model=cast(str | None, model_options["base_model"]),
                adapter_path=cast(Path | None, model_options["adapter_path"]),
            ),
            max_new_tokens=max_new_tokens,
            scripted_responses=responses,
            ollama_base_url=ollama_base_url or os.environ.get("MICRO_MODEL_AGENT_OLLAMA_BASE_URL"),
        )
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    assert model_provider is not None

    allowed_commands = {}
    if verification_command and test_command:
        allowed_commands[verification_command] = test_command
    elif verification_command:
        _fail("--test-command is required when --verification-command is set")

    trace_store = workflow_trace_store(repository_root)
    executor = build_builtin_tool_executor(repository_root, allowed_commands)
    agent = ToolLoopAgent(
        model_provider=model_provider,
        tool_executor=executor,
        trace_store=trace_store,
    )
    workflow = RunToolLoopWorkflow(agent)
    result = _run(
        # The model sees the tool schemas by default, which helps it produce
        # valid JSON arguments for the selected tools.
        workflow.run(
            RunToolLoopRequest(
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
