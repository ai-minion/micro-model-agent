"""Tests for shared runtime composition helpers."""

from __future__ import annotations

import json
from pathlib import Path

from micro_model_agent.infrastructure.composition import (
    allowed_test_commands,
    build_builtin_tool_executor,
    comparison_trace_store,
    resolve_model_options,
    select_evaluation_model,
    workflow_trace_store,
    workspace_registry,
)
from micro_model_agent.infrastructure.fake_model_provider import ScriptedModelProvider
from micro_model_agent.infrastructure.ollama_model_provider import OllamaModelProvider
from micro_model_agent.infrastructure.repository_metadata import (
    initialize_repository,
    update_model_configuration,
)
from micro_model_agent.infrastructure.tool_executor import BuiltinToolExecutor
from micro_model_agent.infrastructure.transformers_model_provider import (
    TransformersPeftModelProvider,
)


def test_resolve_model_options_uses_selected_repository_config(tmp_path: Path) -> None:
    update_model_configuration(
        tmp_path,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=tmp_path / "training" / "runs" / "proof" / "adapter",
        selected_promotion={"artifact_id": "00000000-0000-4000-8000-000000000001"},
    )

    options = resolve_model_options(
        repository_root=tmp_path,
        env={},
    )

    assert options.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert options.adapter_path == tmp_path / "training" / "runs" / "proof" / "adapter"
    assert options.selected_promotion_artifact_id == "00000000-0000-4000-8000-000000000001"


def test_resolve_model_options_prefer_args_over_env_and_config(tmp_path: Path) -> None:
    initialize_repository(
        tmp_path,
        default_model="configured-ollama",
        base_model="configured-base",
        adapter_path="configured-adapter",
    )

    options = resolve_model_options(
        repository_root=tmp_path,
        model="explicit-ollama",
        base_model="explicit-base",
        adapter_path=Path("explicit-adapter"),
        env={
            "MICRO_MODEL_AGENT_DEFAULT_MODEL": "env-ollama",
            "MICRO_MODEL_AGENT_BASE_MODEL": "env-base",
            "MICRO_MODEL_AGENT_ADAPTER_PATH": "env-adapter",
        },
    )

    assert options.model == "explicit-ollama"
    assert options.base_model == "explicit-base"
    assert options.adapter_path == Path("explicit-adapter")


def test_resolve_model_options_can_disable_adapter(tmp_path: Path) -> None:
    update_model_configuration(
        tmp_path,
        base_model="configured-base",
        adapter_path="configured-adapter",
    )

    options = resolve_model_options(
        repository_root=tmp_path,
        use_adapter=False,
        env={},
    )

    assert options.base_model == "configured-base"
    assert options.adapter_path is None


def test_allowed_test_commands_adds_default_pytest() -> None:
    commands = allowed_test_commands(None, None, use_default_pytest=True)

    assert list(commands) == ["pytest"]
    assert commands["pytest"].args == ("python3", "-m", "pytest", "-q")


def test_runtime_factories_use_standard_repository_paths(tmp_path: Path) -> None:
    executor = build_builtin_tool_executor(
        tmp_path,
        allowed_test_commands("unit", ["python", "-m", "pytest"]),
    )

    assert isinstance(executor, BuiltinToolExecutor)
    assert workflow_trace_store(tmp_path).path == tmp_path / ".traces" / "workflows.jsonl"
    assert comparison_trace_store(tmp_path).path == (
        tmp_path / ".traces" / "comparison_sessions.jsonl"
    )
    assert workspace_registry(tmp_path).path == (
        tmp_path / ".micro_model_agent" / "workspaces.jsonl"
    )


def test_select_evaluation_model_prefers_scripted_responses(tmp_path: Path) -> None:
    selection = select_evaluation_model(
        run_dir=tmp_path,
        model="ignored-ollama",
        base_model="metadata-base",
        adapter_path=tmp_path / "missing-adapter",
        scripted_responses=['{"ok": true}'],
        max_new_tokens=32,
    )

    assert selection.provider_kind == "scripted"
    assert isinstance(selection.provider, ScriptedModelProvider)
    assert selection.model == "ignored-ollama"
    assert selection.base_model == "metadata-base"
    assert selection.adapter_path == tmp_path / "missing-adapter"


def test_select_evaluation_model_uses_explicit_adapter(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "configured-base"}',
        encoding="utf-8",
    )

    selection = select_evaluation_model(
        run_dir=tmp_path,
        adapter_path=adapter_path,
        scripted_responses=[],
        max_new_tokens=48,
    )

    assert selection.provider_kind == "transformers_peft"
    assert isinstance(selection.provider, TransformersPeftModelProvider)
    assert selection.base_model == "configured-base"
    assert selection.adapter_path == adapter_path
    assert selection.provider.max_new_tokens == 48


def test_select_evaluation_model_uses_ollama_options(tmp_path: Path) -> None:
    selection = select_evaluation_model(
        run_dir=tmp_path,
        model="qwen:test",
        scripted_responses=[],
        max_new_tokens=64,
        env={"MICRO_MODEL_AGENT_OLLAMA_BASE_URL": "http://localhost:11434"},
    )

    assert selection.provider_kind == "ollama"
    assert isinstance(selection.provider, OllamaModelProvider)
    assert selection.provider.model_name == "qwen:test"
    assert selection.provider.options == {"num_predict": 64}


def test_select_evaluation_model_uses_runnable_training_artifact(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-4000-8000-000000000001",
                "name": "trained-adapter",
                "kind": "adapter",
                "path": str(adapter_path),
                "base_model": "artifact-base",
                "metrics": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    selection = select_evaluation_model(
        run_dir=run_dir,
        scripted_responses=[],
        max_new_tokens=96,
    )

    assert selection.provider_kind == "training_artifact"
    assert isinstance(selection.provider, TransformersPeftModelProvider)
    assert selection.base_model == "artifact-base"
    assert selection.adapter_path == adapter_path
    assert selection.artifact is not None
    assert selection.artifact.name == "trained-adapter"


def test_select_evaluation_model_returns_metadata_fallback_for_dry_run_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact_path = tmp_path / "missing-adapter"
    (run_dir / "artifact.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-4000-8000-000000000001",
                "name": "dry-run-adapter",
                "kind": "adapter",
                "path": str(artifact_path),
                "base_model": "artifact-base",
                "metrics": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    selection = select_evaluation_model(
        run_dir=run_dir,
        scripted_responses=[],
        max_new_tokens=96,
    )

    assert selection.provider is None
    assert selection.provider_kind == "metadata"
    assert selection.artifact is not None
    assert selection.artifact.path == str(artifact_path)
