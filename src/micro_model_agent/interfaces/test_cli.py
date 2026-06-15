"""CLI smoke tests for the local synthetic pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from micro_model_agent.infrastructure.dataset_store import load_dataset_examples
from micro_model_agent.interfaces.cli import _load_dotenv, app


def test_load_dotenv_sets_values_without_overriding_existing_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local secrets",
                "HF_TOKEN='from-file'",
                "MICRO_MODEL_AGENT_DEFAULT_MODEL=from-file",
                "export MICRO_MODEL_AGENT_TRACE_DIR=.micro_model_agent/traces",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MICRO_MODEL_AGENT_TRACE_DIR", raising=False)
    monkeypatch.setenv("MICRO_MODEL_AGENT_DEFAULT_MODEL", "already-set")

    _load_dotenv(env_file)

    assert os.environ["HF_TOKEN"] == "from-file"
    assert os.environ["MICRO_MODEL_AGENT_DEFAULT_MODEL"] == "already-set"
    assert os.environ["MICRO_MODEL_AGENT_TRACE_DIR"] == ".micro_model_agent/traces"


def test_cli_init_creates_idempotent_repository_metadata(tmp_path: Path) -> None:
    runner = CliRunner()

    first = runner.invoke(
        app,
        [
            "init",
            "--repository-root",
            str(tmp_path),
            "--default-model",
            "qwen",
        ],
    )
    second = runner.invoke(app, ["init", "--repository-root", str(tmp_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "Initialized MicroModelAgent metadata" in first.output
    assert "Already initialized MicroModelAgent metadata" in second.output
    config = json.loads((tmp_path / ".micro_model_agent" / "config.json").read_text())
    assert config["model"]["default_model"] == "qwen"


def test_cli_synthetic_dataset_train_and_eval_loop(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = tmp_path / "synthetic.jsonl"
    export_path = tmp_path / "synthetic.sft.jsonl"
    run_dir = tmp_path / "training" / "runs" / "latest"

    synthesize = runner.invoke(
        app,
        [
            "dataset",
            "synthesize",
            "--count",
            "3",
            "--output",
            str(dataset_path),
            "--template-dir",
            "examples/synthetic-data",
            "--seed",
            "23",
        ],
    )
    assert synthesize.exit_code == 0, synthesize.output
    assert dataset_path.exists()
    assert "Categories:" in synthesize.output

    validate = runner.invoke(app, ["dataset", "validate", "--path", str(dataset_path)])
    assert validate.exit_code == 0, validate.output
    assert "validated 3 examples" in validate.output
    assert "Categories:" in validate.output
    assert "Outcomes:" in validate.output

    export = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "--path",
            str(dataset_path),
            "--output",
            str(export_path),
        ],
    )
    assert export.exit_code == 0, export.output
    assert export_path.exists()

    train = runner.invoke(
        app,
        [
            "train",
            "synthetic",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(run_dir),
        ],
    )
    assert train.exit_code == 0, train.output
    assert (run_dir / "run.json").exists()
    assert (run_dir / "artifact.json").exists()

    evaluate = runner.invoke(app, ["eval", "synthetic", "--run-id", str(run_dir)])
    assert evaluate.exit_code == 0, evaluate.output
    assert (run_dir / "evaluation.json").exists()


def test_cli_synthetic_eval_failure_names_category_and_example(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = Path("examples/synthetic-data/held-out.behavior.jsonl")
    examples = load_dataset_examples(dataset_path)
    response_file = tmp_path / "responses.jsonl"
    responses = [example.target for example in examples]
    responses[0] = {
        "tool_name": "repo.read",
        "arguments": {"files": [{"path": "README.md"}]},
    }
    response_file.write_text(
        "\n".join(json.dumps(response, sort_keys=True) for response in responses),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "synthetic",
            "--run-id",
            str(tmp_path / "training" / "runs" / "latest"),
            "--dataset",
            str(dataset_path),
            "--scripted-response-file",
            str(response_file),
            "--pass-threshold",
            "1.0",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "Failures:" in result.output
    assert "valid_tool_call/00000000-0000-4000-8000-000000000001" in result.output


def test_cli_loop_runs_scripted_tool_call_and_final_response(tmp_path: Path) -> None:
    runner = CliRunner()
    (tmp_path / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    response_file = tmp_path / "responses.jsonl"
    response_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "tool_name": "repo.read",
                        "arguments": {"files": [{"path": "app.py"}]},
                    }
                ),
                json.dumps({"final_response": "app.py value() returns 1.", "ok": True}),
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "loop",
            "Read app.py and tell me what value() returns.",
            "--repository-root",
            str(tmp_path),
            "--scripted-response-file",
            str(response_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "app.py value() returns 1." in result.output
    assert "Tool calls: 1" in result.output
    assert (tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl").exists()
