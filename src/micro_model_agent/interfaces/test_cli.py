"""CLI smoke tests for the local synthetic pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from micro_model_agent.dataset.domain.value_objects import (
    DatasetExample,
    DatasetExampleKind,
    DatasetLabel,
    OutcomeLabel,
    QualityLabel,
)
from micro_model_agent.dataset.infrastructure.dataset_store import (
    load_dataset_examples,
    write_dataset_examples,
)
from micro_model_agent.interfaces.cli import app
from micro_model_agent.interfaces.cli.common import _load_dotenv
from micro_model_agent.interfaces.composition import resolve_model_options
from micro_model_agent.repository_ops.infrastructure.metadata import (
    initialize_repository,
    update_model_configuration,
)


def _registered_command_names(typer_app: typer.Typer) -> set[str]:
    names: set[str] = set()
    for command in typer_app.registered_commands:
        name = command.name
        if name is None and command.callback is not None:
            name = command.callback.__name__.replace("_", "-")
        if name is not None:
            names.add(name)
    return names


def _registered_group(typer_app: typer.Typer, name: str) -> typer.Typer:
    for group in typer_app.registered_groups:
        if group.name == name:
            assert group.typer_instance is not None
            return group.typer_instance
    raise AssertionError(f"missing Typer group {name!r}")


def test_cli_command_surface_is_stable() -> None:
    assert _registered_command_names(app) == {
        "init",
        "index",
        "task",
        "loop",
        "serve-mcp",
    }
    assert {group.name for group in app.registered_groups} == {
        "dataset",
        "train",
        "eval",
        "promote",
    }
    assert _registered_command_names(_registered_group(app, "dataset")) == {
        "synthesize",
        "validate",
        "export",
        "export-traces",
        "review-trace",
        "relabel",
        "merge",
    }
    assert _registered_command_names(_registered_group(app, "train")) == {"synthetic"}
    assert _registered_command_names(_registered_group(app, "eval")) == {
        "synthetic",
        "traces",
        "workspace-staged",
        "review-workspace-staged",
        "compare",
    }
    assert _registered_command_names(_registered_group(app, "promote")) == {
        "gate",
        "record",
        "list",
        "select",
        "package-ollama",
    }


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
                "export MICRO_MODEL_AGENT_TRACE_DIR=.traces",
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
    assert os.environ["MICRO_MODEL_AGENT_TRACE_DIR"] == ".traces"


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


def test_cli_loop_model_options_use_selected_repository_config(tmp_path: Path) -> None:
    update_model_configuration(
        tmp_path,
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=tmp_path / "training" / "runs" / "proof" / "adapter",
        selected_promotion={"artifact_id": "00000000-0000-4000-8000-000000000001"},
    )

    options = resolve_model_options(
        repository_root=tmp_path,
        model=None,
        base_model=None,
        adapter_path=None,
    )

    assert options.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert options.adapter_path == tmp_path / "training" / "runs" / "proof" / "adapter"


def test_cli_loop_model_options_prefer_args_over_repository_config(tmp_path: Path) -> None:
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
    )

    assert options.model == "explicit-ollama"
    assert options.base_model == "explicit-base"
    assert options.adapter_path == Path("explicit-adapter")


def test_cli_index_writes_local_lexical_index(tmp_path: Path) -> None:
    runner = CliRunner()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "def run_workflow() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["index", "--repository-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Indexed 1 files" in result.output
    assert "Source types: source_code=1" in result.output
    assert "Code metadata: symbols=1; imports=0; test_files=0" in result.output
    index_path = tmp_path / ".micro_model_agent" / "index" / "lexical-index.json"
    assert str(index_path) in result.output
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["summary"]["indexed_file_count"] == 1
    assert index["files"][0]["path"] == "src/service.py"
    assert "workflow" in index["postings"]


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
    assert "Tool profile:" in validate.output

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
    artifact = json.loads((run_dir / "artifact.json").read_text(encoding="utf-8"))
    assert "repo.search" in artifact["metadata"]["dataset_tool_profile"]["available_tools"]
    assert len(artifact["metadata"]["source_dataset_sha256"]) == 64
    assert len(artifact["metadata"]["training_dataset_sha256"]) == 64
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_record["dataset_version"] == artifact["metadata"]["source_dataset_sha256"]

    evaluate = runner.invoke(app, ["eval", "synthetic", "--run-id", str(run_dir)])
    assert evaluate.exit_code == 0, evaluate.output
    assert (run_dir / "evaluation.json").exists()
    report = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert report["details"]["evaluation_metadata"]["tool_profile"]["example_count"] == 15
    assert "repo.read" in report["details"]["evaluation_metadata"]["tool_profile"][
        "available_tools"
    ]


def test_cli_dataset_synthesize_filters_categories(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        app,
        [
            "dataset",
            "synthesize",
            "--count",
            "6",
            "--output",
            str(dataset_path),
            "--template-dir",
            "examples/synthetic-data",
            "--include-category",
            "trace_patch_training",
            "--include-category",
            "trace_final_response_training",
            "--seed",
            "41",
        ],
    )

    assert result.exit_code == 0, result.output
    examples = load_dataset_examples(dataset_path)
    assert len(examples) == 6
    assert {example.metadata["category"] for example in examples} == {
        "trace_patch_training",
        "trace_final_response_training",
    }


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


def test_cli_trace_eval_scores_held_out_trace_examples(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = Path("examples/trace-data/held-out.trace.jsonl")
    examples = load_dataset_examples(dataset_path)
    response_file = tmp_path / "trace_responses.jsonl"
    responses = [
        {
            "patch": example.target.get("patch"),
            "final_response": example.target["final_response"],
            "tool_history": [
                {"tool_name": item["tool_call"]["tool_name"]}
                for item in example.input["tool_history"]
            ],
        }
        for example in examples
    ]
    response_file.write_text(
        "\n".join(json.dumps(response, sort_keys=True) for response in responses),
        encoding="utf-8",
    )
    run_dir = tmp_path / "training" / "runs" / "latest"
    report_path = run_dir / "trace-evaluation.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "traces",
            "--run-id",
            str(run_dir),
            "--dataset",
            str(dataset_path),
            "--scripted-response-file",
            str(response_file),
            "--pass-threshold",
            "1.0",
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "trace behavior eval scored 1.00" in result.output
    assert f"report written to {report_path}" in result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["details"]["metrics"]["tool_history_match_rate"] == 1.0
    assert report["details"]["evaluation_metadata"]["provider"] == "scripted"
    assert "repo.write_patch" in report["details"]["evaluation_metadata"]["tool_profile"][
        "available_tools"
    ]


def test_cli_workspace_staged_eval_scores_held_out_examples(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = Path("examples/workspace-eval/held-out.workspace-staged.jsonl")
    examples = load_dataset_examples(dataset_path)
    responses = []
    for example in examples:
        stages = example.target["stages"]
        responses.append(
            {
                "read_search": {
                    "files": stages["read_search"]["required_files"],
                    "queries": stages["read_search"]["required_queries"],
                },
                "diagnosis": {
                    "root_cause": " ".join(stages["diagnosis"]["required_terms"]),
                    "plan": ["read", "diagnose", "dry-run patch"],
                },
                "patch_proposal": {
                    "changed_files": stages["patch_proposal"]["required_changed_files"],
                    "patch": " ".join(stages["patch_proposal"]["patch_contains"]),
                },
                "test_selection": {
                    "commands": stages["test_selection"]["required_commands"],
                },
                "final_summary": {
                    "summary": "dry-run "
                    + " ".join(stages["final_summary"]["required_summary_terms"]),
                },
            }
        )
    response_file = tmp_path / "workspace_staged_responses.jsonl"
    response_file.write_text(
        "\n".join(json.dumps(response, sort_keys=True) for response in responses),
        encoding="utf-8",
    )
    run_dir = tmp_path / "training" / "runs" / "latest"
    report_path = run_dir / "workspace-staged-evaluation.json"

    result = runner.invoke(
        app,
        [
            "eval",
            "workspace-staged",
            "--run-id",
            str(run_dir),
            "--dataset",
            str(dataset_path),
            "--scripted-response-file",
            str(response_file),
            "--pass-threshold",
            "1.0",
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "staged workspace eval scored 1.00" in result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["details"]["metrics"]["patch_proposal_score"] == 1.0
    assert report["details"]["metrics"]["test_selection_score"] == 1.0
    assert report["details"]["evaluation_metadata"]["provider"] == "scripted"
    assert "repo.write_patch" in report["details"]["evaluation_metadata"]["tool_profile"][
        "available_tools"
    ]


def test_cli_review_workspace_staged_writes_auto_triage_queue(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = Path("examples/workspace-eval/held-out.workspace-staged.jsonl")
    examples = load_dataset_examples(dataset_path)
    report_path = tmp_path / "workspace-staged-evaluation.json"
    output_path = tmp_path / "workspace-review.jsonl"
    report_path.write_text(
        json.dumps(
            {
                "passed": False,
                "summary": "staged weak",
                "score": 0.3,
                "details": {
                    "evaluation_metadata": {"run_id": "base", "provider": "scripted"},
                    "examples": [
                        {
                            "example_id": str(examples[0].id),
                            "score": 0.25,
                            "parse_success": True,
                            "stages": [
                                {"name": "read_search", "score": 0.25, "errors": ["missing"]}
                            ],
                            "raw_response": "{}",
                            "parsed_response": {},
                        },
                        {
                            "example_id": str(examples[1].id),
                            "score": 0.98,
                            "parse_success": True,
                            "stages": [
                                {"name": "read_search", "score": 1.0, "errors": []}
                            ],
                            "raw_response": "{}",
                            "parsed_response": {},
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "review-workspace-staged",
            "--dataset",
            str(dataset_path),
            "--report",
            str(report_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "auto_reject_simple_failure=1" in result.output
    assert "auto_accept_candidate=1" in result.output
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["auto_triage"]["decision"] == "auto_reject_simple_failure"
    assert records[1]["auto_triage"]["decision"] == "auto_accept_candidate"
    assert "workspace_files" in records[0]
    assert "gold_response" in records[0]
    assert records[0]["model_results"][0]["run_id"] == "base"


def test_cli_promotion_gate_requires_all_evaluation_reports(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = tmp_path / "synthetic.jsonl"
    run_dir = tmp_path / "training" / "runs" / "latest"

    synthesize = runner.invoke(
        app,
        [
            "dataset",
            "synthesize",
            "--count",
            "1",
            "--output",
            str(dataset_path),
            "--template-dir",
            "examples/synthetic-data",
            "--seed",
            "23",
        ],
    )
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
    passing_report = run_dir / "synthetic-evaluation.json"
    failing_report = run_dir / "trace-evaluation.json"
    passing_report.write_text(
        json.dumps({"passed": True, "summary": "synthetic ok", "score": 0.95, "details": {}}),
        encoding="utf-8",
    )
    failing_report.write_text(
        json.dumps({"passed": True, "summary": "trace weak", "score": 0.7, "details": {}}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "promote",
            "gate",
            "--run-id",
            str(run_dir),
            "--evaluation-report",
            str(passing_report),
            "--evaluation-report",
            str(failing_report),
            "--minimum-score",
            "0.8",
        ],
    )

    assert synthesize.exit_code == 0, synthesize.output
    assert train.exit_code == 0, train.output
    assert result.exit_code == 1, result.output
    assert "Promotion gate blocked" in result.output
    report = json.loads((run_dir / "promotion.json").read_text(encoding="utf-8"))
    assert report["promoted"] is False
    assert [evaluation["can_promote"] for evaluation in report["evaluations"]] == [True, False]


def test_cli_eval_compare_writes_comparison_report(tmp_path: Path) -> None:
    runner = CliRunner()
    baseline_report = tmp_path / "base-synthetic-evaluation.json"
    adapter_report = tmp_path / "adapter-synthetic-evaluation.json"
    output = tmp_path / "comparison.json"
    baseline_report.write_text(
        json.dumps(
            {
                "passed": False,
                "summary": "base",
                "score": 0.70,
                "details": {
                    "metrics": {
                        "correct_tool_rate": 0.60,
                        "valid_argument_rate": 0.75,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    adapter_report.write_text(
        json.dumps(
            {
                "passed": True,
                "summary": "adapter",
                "score": 0.84,
                "details": {
                    "metrics": {
                        "correct_tool_rate": 0.74,
                        "valid_argument_rate": 0.90,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline-report",
            str(baseline_report),
            "--adapter-report",
            str(adapter_report),
            "--minimum-score-delta",
            "0.10",
            "--minimum-metric-delta",
            "correct_tool_rate=0.10",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "evaluation comparison passed" in result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["score_delta"] == pytest.approx(0.14)
    metric_deltas = {delta["name"]: delta for delta in report["metric_deltas"]}
    assert metric_deltas["correct_tool_rate"]["passed"] is True


def test_cli_eval_compare_fails_when_adapter_misses_threshold(tmp_path: Path) -> None:
    runner = CliRunner()
    baseline_report = tmp_path / "base-trace-evaluation.json"
    adapter_report = tmp_path / "adapter-trace-evaluation.json"
    baseline_report.write_text(
        json.dumps(
            {
                "passed": True,
                "summary": "base",
                "score": 0.80,
                "details": {"metrics": {"tool_history_match_rate": 0.80}},
            }
        ),
        encoding="utf-8",
    )
    adapter_report.write_text(
        json.dumps(
            {
                "passed": True,
                "summary": "adapter",
                "score": 0.82,
                "details": {"metrics": {"tool_history_match_rate": 0.84}},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline-report",
            str(baseline_report),
            "--adapter-report",
            str(adapter_report),
            "--minimum-score-delta",
            "0.05",
            "--minimum-metric-delta",
            "tool_history_match_rate=0.10",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "evaluation comparison failed" in result.output
    assert "score delta 0.0200 is below minimum 0.0500" in result.output
    assert "metric 'tool_history_match_rate' delta 0.0400 is below minimum 0.1000" in result.output


def test_cli_promotion_record_writes_local_registry(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = tmp_path / "synthetic.jsonl"
    run_dir = tmp_path / "training" / "runs" / "latest"
    registry_path = tmp_path / "training" / "promoted_models.jsonl"

    synthesize = runner.invoke(
        app,
        [
            "dataset",
            "synthesize",
            "--count",
            "1",
            "--output",
            str(dataset_path),
            "--template-dir",
            "examples/synthetic-data",
            "--seed",
            "24",
        ],
    )
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
    eval_report = run_dir / "synthetic-evaluation.json"
    eval_report.write_text(
        json.dumps({"passed": True, "summary": "synthetic ok", "score": 0.92, "details": {}}),
        encoding="utf-8",
    )
    gate = runner.invoke(
        app,
        [
            "promote",
            "gate",
            "--run-id",
            str(run_dir),
            "--evaluation-report",
            str(eval_report),
            "--minimum-score",
            "0.9",
        ],
    )
    record = runner.invoke(
        app,
        [
            "promote",
            "record",
            "--run-id",
            str(run_dir),
            "--registry",
            str(registry_path),
            "--reviewer-notes",
            "Reviewed smoke metrics.",
            "--approved-by",
            "tests",
        ],
    )
    listed = runner.invoke(
        app,
        ["promote", "list", "--registry", str(registry_path)],
    )

    assert synthesize.exit_code == 0, synthesize.output
    assert train.exit_code == 0, train.output
    assert gate.exit_code == 0, gate.output
    assert record.exit_code == 0, record.output
    assert listed.exit_code == 0, listed.output
    registry_records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert registry_records[0]["artifact_name"] == "synthetic-dry-run-adapter"
    assert registry_records[0]["minimum_score"] == 0.9
    assert registry_records[0]["reviewer_notes"] == "Reviewed smoke metrics."
    assert "synthetic-dry-run-adapter" in listed.output


def test_cli_promote_select_requires_confirmation(tmp_path: Path) -> None:
    runner = CliRunner()
    registry_path = tmp_path / "training" / "promoted_models.jsonl"

    result = runner.invoke(
        app,
        [
            "promote",
            "select",
            "--artifact-id",
            "00000000-0000-4000-8000-000000000001",
            "--registry",
            str(registry_path),
            "--repository-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "rerun with --confirm" in result.output
    assert not (tmp_path / ".micro_model_agent" / "config.json").exists()


def test_cli_promote_select_updates_repository_model_config(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = tmp_path / "synthetic.jsonl"
    run_dir = tmp_path / "training" / "runs" / "latest"
    registry_path = tmp_path / "training" / "promoted_models.jsonl"

    synthesize = runner.invoke(
        app,
        [
            "dataset",
            "synthesize",
            "--count",
            "1",
            "--output",
            str(dataset_path),
            "--template-dir",
            "examples/synthetic-data",
            "--seed",
            "25",
        ],
    )
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
    eval_report = run_dir / "synthetic-evaluation.json"
    eval_report.write_text(
        json.dumps({"passed": True, "summary": "synthetic ok", "score": 0.95, "details": {}}),
        encoding="utf-8",
    )
    gate = runner.invoke(
        app,
        [
            "promote",
            "gate",
            "--run-id",
            str(run_dir),
            "--evaluation-report",
            str(eval_report),
            "--minimum-score",
            "0.9",
        ],
    )
    record = runner.invoke(
        app,
        [
            "promote",
            "record",
            "--run-id",
            str(run_dir),
            "--registry",
            str(registry_path),
            "--approved-by",
            "tests",
        ],
    )
    registry_record = json.loads(registry_path.read_text(encoding="utf-8").splitlines()[0])

    select = runner.invoke(
        app,
        [
            "promote",
            "select",
            "--artifact-id",
            registry_record["artifact_id"],
            "--registry",
            str(registry_path),
            "--repository-root",
            str(tmp_path),
            "--confirm",
        ],
    )

    assert synthesize.exit_code == 0, synthesize.output
    assert train.exit_code == 0, train.output
    assert gate.exit_code == 0, gate.output
    assert record.exit_code == 0, record.output
    assert select.exit_code == 0, select.output
    config = json.loads((tmp_path / ".micro_model_agent" / "config.json").read_text())
    assert config["model"]["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config["model"]["adapter_path"] == registry_record["artifact_path"]
    assert config["model"]["selected_promotion"]["artifact_id"] == registry_record["artifact_id"]
    assert config["model"]["selected_promotion"]["approved_by"] == "tests"


def test_cli_promote_package_ollama_writes_modelfile(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_path = tmp_path / "synthetic.jsonl"
    run_dir = tmp_path / "training" / "runs" / "latest"
    registry_path = tmp_path / "training" / "promoted_models.jsonl"
    package_dir = tmp_path / "ollama-package"

    synthesize = runner.invoke(
        app,
        [
            "dataset",
            "synthesize",
            "--count",
            "1",
            "--output",
            str(dataset_path),
            "--template-dir",
            "examples/synthetic-data",
            "--seed",
            "26",
        ],
    )
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
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    eval_report = run_dir / "synthetic-evaluation.json"
    eval_report.write_text(
        json.dumps({"passed": True, "summary": "synthetic ok", "score": 0.95, "details": {}}),
        encoding="utf-8",
    )
    gate = runner.invoke(
        app,
        [
            "promote",
            "gate",
            "--run-id",
            str(run_dir),
            "--evaluation-report",
            str(eval_report),
            "--minimum-score",
            "0.9",
        ],
    )
    record = runner.invoke(
        app,
        [
            "promote",
            "record",
            "--run-id",
            str(run_dir),
            "--registry",
            str(registry_path),
        ],
    )
    registry_record = json.loads(registry_path.read_text(encoding="utf-8").splitlines()[0])

    package = runner.invoke(
        app,
        [
            "promote",
            "package-ollama",
            "--artifact-id",
            registry_record["artifact_id"],
            "--registry",
            str(registry_path),
            "--model-name",
            "micro-agent-proof:qwen",
            "--ollama-base-model",
            "qwen2.5-coder:7b",
            "--output-dir",
            str(package_dir),
        ],
    )

    assert synthesize.exit_code == 0, synthesize.output
    assert train.exit_code == 0, train.output
    assert gate.exit_code == 0, gate.output
    assert record.exit_code == 0, record.output
    assert package.exit_code == 0, package.output
    modelfile = (package_dir / "Modelfile").read_text(encoding="utf-8")
    manifest = json.loads((package_dir / "ollama-package.json").read_text(encoding="utf-8"))
    assert "FROM qwen2.5-coder:7b" in modelfile
    assert f"ADAPTER {adapter_dir}" in modelfile
    assert manifest["model_name"] == "micro-agent-proof:qwen"
    assert manifest["created"] is False
    assert "ollama create micro-agent-proof:qwen" in package.output


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
    assert (
        tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"
    ).exists()


def test_cli_dataset_export_traces_writes_review_examples(tmp_path: Path) -> None:
    runner = CliRunner()
    (tmp_path / "app.py").write_text("TOKEN = 'sk-testtoken1234567890'\n", encoding="utf-8")
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
                json.dumps(
                    {
                        "final_response": "Found sk-testtoken1234567890.",
                        "ok": True,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    trace_path = tmp_path / ".micro_model_agent" / "traces" / "workflows.jsonl"
    dataset_path = tmp_path / ".micro_model_agent" / "datasets" / "trace_examples.jsonl"

    loop = runner.invoke(
        app,
        [
            "loop",
            "Read app.py.",
            "--repository-root",
            str(tmp_path),
            "--scripted-response-file",
            str(response_file),
        ],
    )
    export = runner.invoke(
        app,
        [
            "dataset",
            "export-traces",
            "--trace-path",
            str(trace_path),
            "--output",
            str(dataset_path),
        ],
    )

    assert loop.exit_code == 0, loop.output
    assert export.exit_code == 0, export.output
    assert "Exported 1 trace-derived examples" in export.output
    examples = load_dataset_examples(dataset_path)
    assert examples[0].label.outcome.value == "needs_review"
    assert examples[0].metadata["redacted"] is True
    assert "[REDACTED]" in examples[0].target["summary"]


def test_cli_dataset_relabel_and_merge_trace_examples(tmp_path: Path) -> None:
    runner = CliRunner()
    trace_example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "Summarize app.py"},
        target={"final_response": "app.py was summarized."},
        source="trace:abc",
        label=DatasetLabel(
            outcome=OutcomeLabel.NEEDS_REVIEW,
            quality=QualityLabel.UNKNOWN,
        ),
        metadata={"trace_id": "abc", "review_required": True},
    )
    synthetic_example = DatasetExample(
        kind=DatasetExampleKind.REPAIR,
        input={"goal": "Synthetic task"},
        target={"patch": "diff --git a/app.py b/app.py\n"},
        source="synthetic:one",
        label=DatasetLabel(outcome=OutcomeLabel.ACCEPTED, quality=QualityLabel.GOOD),
    )
    trace_path = tmp_path / "trace_examples.jsonl"
    synthetic_path = tmp_path / "synthetic.jsonl"
    curated_path = tmp_path / "curated.jsonl"
    merged_path = tmp_path / "merged.jsonl"
    write_dataset_examples(trace_path, [trace_example])
    write_dataset_examples(synthetic_path, [synthetic_example])

    relabel = runner.invoke(
        app,
        [
            "dataset",
            "relabel",
            "--path",
            str(trace_path),
            "--output",
            str(curated_path),
            "--trace-id",
            "abc",
            "--outcome",
            "accepted",
            "--quality",
            "good",
            "--reviewer-notes",
            "Reviewed.",
        ],
    )
    merge = runner.invoke(
        app,
        [
            "dataset",
            "merge",
            "--input",
            str(synthetic_path),
            "--input",
            str(curated_path),
            "--output",
            str(merged_path),
        ],
    )

    assert relabel.exit_code == 0, relabel.output
    assert "Relabeled 1 of 1 examples" in relabel.output
    assert merge.exit_code == 0, merge.output
    assert "Merged 2 examples" in merge.output
    merged = load_dataset_examples(merged_path)
    assert [example.source for example in merged] == ["synthetic:one", "trace:abc"]
    assert merged[1].label.quality is QualityLabel.GOOD
