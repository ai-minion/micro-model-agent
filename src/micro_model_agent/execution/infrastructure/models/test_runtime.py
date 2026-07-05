"""Tests for runtime model and loop assembly helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from micro_model_agent.execution.infrastructure.models.fake import ScriptedModelProvider
from micro_model_agent.execution.infrastructure.models.runtime import (
    RuntimeModelOptions,
    base_model_from_adapter,
    build_model_provider,
    resolve_model_options,
    run_cli_tool_loop,
    run_mcp_agent_loop,
    runtime_model_metadata,
)
from micro_model_agent.repository_ops.infrastructure.metadata import (
    initialize_repository,
    update_model_configuration,
)


def test_resolve_model_options_prefers_args_over_env_and_config(tmp_path: Path) -> None:
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

    assert runtime_model_metadata(options) == {
        "model": "explicit-ollama",
        "base_model": "explicit-base",
        "adapter_path": "explicit-adapter",
        "selected_promotion_artifact_id": None,
    }


def test_resolve_model_options_uses_selected_repository_config(tmp_path: Path) -> None:
    update_model_configuration(
        tmp_path,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=tmp_path / "training" / "runs" / "proof" / "adapter",
        selected_promotion={"artifact_id": "00000000-0000-4000-8000-000000000001"},
    )

    options = resolve_model_options(repository_root=tmp_path, env={})

    assert options.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert options.adapter_path == tmp_path / "training" / "runs" / "proof" / "adapter"
    assert options.selected_promotion_artifact_id == "00000000-0000-4000-8000-000000000001"


def test_resolve_model_options_does_not_use_default_adapter_fallback(
    tmp_path: Path,
) -> None:
    options = resolve_model_options(
        repository_root=tmp_path,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=None,
        env={},
    )

    assert options.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert options.adapter_path is None


def test_build_model_provider_uses_scripted_responses_without_real_model() -> None:
    provider = build_model_provider(
        options=RuntimeModelOptions(model=None, base_model=None, adapter_path=None),
        max_new_tokens=64,
        scripted_responses=["first", "second"],
    )

    assert isinstance(provider, ScriptedModelProvider)
    assert asyncio.run(provider.complete([{"role": "user", "content": "prompt"}])) == "first"


def test_base_model_from_adapter_reads_peft_config(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"}),
        encoding="utf-8",
    )

    assert base_model_from_adapter(adapter_path) == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_mcp_loop_runtime_returns_public_response_shape(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nstatus: runtime\n", encoding="utf-8")

    result = asyncio.run(
        run_mcp_agent_loop(
            goal="Read README.md and summarize status.",
            repository_root=tmp_path,
            available_tools=["read_file"],
            max_tool_calls=1,
            scripted_responses=[
                json.dumps(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "README.md"}]},
                    }
                ),
                json.dumps({"final_response": "status is runtime", "ok": True}),
            ],
            default_available_tools=("repo.read",),
            tool_aliases={"read_file": ("repo.read",)},
            required_tool_aliases={},
        )
    )

    assert result["ok"] is True
    assert result["response"] == "status is runtime"
    assert result["steps"][0]["tool_name"] == "repo.read"
    assert result["model"] == {
        "model": None,
        "base_model": None,
        "adapter_path": None,
        "selected_promotion_artifact_id": None,
    }


def test_cli_loop_runtime_returns_configured_result(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nstatus: cli runtime\n", encoding="utf-8")

    configured = asyncio.run(
        run_cli_tool_loop(
            goal="Read README.md and summarize status.",
            repository_root=tmp_path,
            max_new_tokens=64,
            scripted_responses=[
                json.dumps(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "README.md"}]},
                    }
                ),
                json.dumps({"final_response": "status is cli runtime", "ok": True}),
            ],
            available_tools=("repo.read",),
            required_tools=(),
            max_turns=4,
            max_tool_calls=1,
            max_tool_result_prompt_chars=2500,
        )
    )

    assert configured.result.ok is True
    assert configured.result.response == "status is cli runtime"
    assert configured.result.trace.final_output["run_metadata"]["interface"] == "cli.loop"
