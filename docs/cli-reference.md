# Micro-Agent CLI Reference

This document lists the current `micro-agent` commands and options. Run commands
through the WSL uv environment used by this project:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv run micro-agent --help
```

## Global Options

```text
micro-agent [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
| --- | --- |
| `--install-completion` | Install completion for the current shell. |
| `--show-completion` | Show completion for the current shell. |
| `--help` | Show help and exit. |

## Top-Level Commands

| Command | Description |
| --- | --- |
| `micro-agent init` | Initialize MicroModelAgent metadata for the current repository. |
| `micro-agent index` | Build a local lexical repository index under `.micro_model_agent/index/`. |
| `micro-agent task` | Run a fixed coding-agent task with local fake dependencies and trace capture. |
| `micro-agent loop` | Run a model-driven agent loop with typed tool calls. |
| `micro-agent serve-mcp` | Serve MicroModelAgent over MCP. |
| `micro-agent dataset` | Dataset generation, validation, and export commands. |
| `micro-agent train` | Local training commands. |
| `micro-agent eval` | Evaluation commands. |
| `micro-agent promote` | Promotion gate and local registry commands. |

## `micro-agent init`

```text
micro-agent init [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--repository-root PATH` | `.` | Repository root to initialize. |
| `--default-model TEXT` | None | Optional default Ollama model name to record in local metadata. |
| `--base-model TEXT` | None | Optional Transformers base model to record in local metadata. |
| `--adapter-path PATH` | None | Optional local PEFT adapter path to record in local metadata. |

## `micro-agent index`

```text
micro-agent index [OPTIONS]
```

Writes `.micro_model_agent/index/lexical-index.json` with indexed file metadata,
SHA-256 hashes, source types, Python symbols/imports when parseable, token-ish
term counts, and lexical postings for future retrieval. Generated directories
such as `.git`, `.venv`, `.venv-wsl`, `.micro_model_agent`, and caches are
skipped.

`repo.semantic_search` uses this index for default all-files searches when the
index exists, and falls back to direct repository scanning when it does not.
Indexed search results include `metadata.index_status` with `is_stale` plus
changed, missing, and extra file counts so callers can decide when to rerun
`micro-agent index`.
The command prints indexed file counts, source-type distribution, skipped file
counts, and code metadata coverage.

| Option | Default | Description |
| --- | --- | --- |
| `--repository-root PATH` | `.` | Repository root to index. |
| `--max-file-bytes INTEGER` | `1000000` | Maximum file size to index. Larger files are skipped. |

## `micro-agent task`

```text
micro-agent task PROMPT [OPTIONS]
```

| Argument | Description |
| --- | --- |
| `PROMPT` | Coding task prompt. |

| Option | Default | Description |
| --- | --- | --- |
| `--patch TEXT` | Empty string | Unified diff to use with the static local model provider. |
| `--repository-root PATH` | `.` | Repository root to operate on. |
| `--dry-run / --no-dry-run` | `--dry-run` | Validate the patch without applying it. |
| `--require-approval / --no-require-approval` | `--require-approval` | Require explicit approval before applying. |
| `--expected-changed-file TEXT` | None | File that the generated patch is expected to change. Can be passed more than once. |
| `--verification-command TEXT` | None | Allowed test command name to run after applying changes. |
| `--test-command TEXT` | None | Shell-free command tokens for the verification command name. Can be passed more than once. |
| `--label accepted\|rejected\|needs_review\|partial\|errored` | None | Outcome label for a stored dataset example. |
| `--quality good\|bad\|mixed\|unknown` | None | Quality label for a stored dataset example. |
| `--failure-mode VALUE` | None | Failure mode label. Can be passed more than once. |
| `--reviewer-notes TEXT` | None | Reviewer notes for the stored dataset example. |
| `--dataset-output PATH` | None | Optional JSONL path for storing a trace-derived labeled example. Required when labels are supplied. |

Failure mode values:

```text
invalid_tool_schema, wrong_tool_selected, unsafe_path_requested,
hallucinated_file, retrieval_missed_context, ignored_retrieved_context,
bad_patch, patch_failed_to_apply, test_failed, verification_skipped,
too_large_change, architecture_violation, unclear_user_goal, provider_error
```

## `micro-agent loop`

```text
micro-agent loop PROMPT [OPTIONS]
```

| Argument | Description |
| --- | --- |
| `PROMPT` | User task prompt for the model-driven tool loop. |

| Option | Default | Description |
| --- | --- | --- |
| `--repository-root PATH` | `.` | Repository root to operate on. |
| `--model TEXT` | None | Ollama model name. Defaults to `MICRO_MODEL_AGENT_DEFAULT_MODEL` or `.micro_model_agent/config.json` `model.default_model`. |
| `--base-model TEXT` | None | Transformers base model for direct PEFT adapter inference. Defaults to `MICRO_MODEL_AGENT_BASE_MODEL` or selected local config. |
| `--adapter-path PATH` | None | Local PEFT adapter path for direct Transformers inference. Defaults to `MICRO_MODEL_AGENT_ADAPTER_PATH` or selected local config. |
| `--adapter / --no-adapter` | `--adapter` | Load the configured PEFT adapter. Use `--no-adapter` for base-model trace collection. |
| `--ollama-base-url TEXT` | None | Ollama host URL. Defaults to `MICRO_MODEL_AGENT_OLLAMA_BASE_URL`. |
| `--max-new-tokens INTEGER` | `384` | Maximum generated tokens per model turn. Range: 1 to 4096. |
| `--max-tool-result-prompt-chars INTEGER` | `12000` | Maximum serialized tool-result characters fed back to the model. |
| `--max-tool-calls INTEGER` | None | Maximum tool calls before forcing a final-response-only prompt. |
| `--scripted-response TEXT` | None | Scripted JSON model response. Can be passed more than once. |
| `--scripted-response-file PATH` | None | JSONL file containing scripted model responses for local smoke tests. |
| `--available-tool TEXT` | All built-in tools | Allowed tool name. Can be passed more than once. |
| `--required-tool TEXT` | None | Tool that must run before the model can return a final response. Can be passed more than once. |
| `--max-turns INTEGER` | `8` | Maximum model turns before failing. |
| `--context TEXT` | Empty string | Extra model-facing task context. |
| `--schema-prompt / --no-schema-prompt` | `--schema-prompt` | Include built-in tool argument schemas in the model prompt. |
| `--capture-prompts` | Disabled | Store exact model prompts in the workflow trace for data collection review. |
| `--allow-no-tool-final` | Disabled | Allow a final response before any tool call has run. |
| `--verification-command TEXT` | None | Allowed test command name that `test.run` can select. |
| `--test-command TEXT` | None | Shell-free command tokens for the verification command name. Can be passed more than once. |

Built-in tool names:

```text
repo.search, repo.read, repo.semantic_search, repo.write_patch, test.run, git.diff
```

## `micro-agent serve-mcp`

```text
micro-agent serve-mcp [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--transport TEXT` | `stdio` | MCP transport: `stdio`, `sse`, or `streamable-http`. |
| `--repository-root PATH` | `.` | Repository root used to decide whether the MCP init tool is needed. |

## `micro-agent dataset synthesize`

```text
micro-agent dataset synthesize [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--count INTEGER` | `100` | Number of synthetic examples to generate. Minimum: 1. |
| `--output PATH` | `.micro_model_agent/datasets/synthetic_seed.jsonl` | Output JSONL path for generated synthetic examples. |
| `--template-dir PATH` | `examples/synthetic-data` | Directory containing committed `*.seed.jsonl` templates. |
| `--seed INTEGER` | None | Deterministic generation seed for reproducible IDs and variants. |
| `--balance-categories / --no-balance-categories` | `--balance-categories` | Cycle categories evenly instead of cycling raw templates. |
| `--vary-scenarios / --no-vary-scenarios` | `--vary-scenarios` | Create deterministic prompt variants while preserving validated targets. |

## `micro-agent dataset validate`

```text
micro-agent dataset validate [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--path PATH` | `.micro_model_agent/datasets/synthetic_seed.jsonl` | JSONL dataset path to validate. |

Validation prints error count plus category, kind, outcome, and tool-profile
metadata distributions.

## `micro-agent dataset export`

```text
micro-agent dataset export [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--path PATH` | `.micro_model_agent/datasets/synthetic_seed.jsonl` | Validated dataset path to export. |
| `--format TEXT` | `sft-jsonl` | Dataset export format. Currently only `sft-jsonl` is supported. |
| `--output PATH` | `.micro_model_agent/datasets/synthetic_seed.sft.jsonl` | Output path for exported dataset. |

## `micro-agent dataset export-traces`

```text
micro-agent dataset export-traces [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--trace-path PATH` | `.micro_model_agent/traces/workflows.jsonl` | Stored workflow trace JSONL path. |
| `--review-path PATH` | `.micro_model_agent/traces/reviews.jsonl` | Human trace review JSONL path used by `--label-mode reviewed`. |
| `--output PATH` | `.micro_model_agent/datasets/trace_examples.jsonl` | Output JSONL path for trace-derived examples. |
| `--label-mode TEXT` | `review` | Label mode: `review`, `reviewed`, or `evaluation`. `reviewed` exports only traces with a human review record. |
| `--kind VALUE` | `repair` | Dataset example kind to export. |
| `--outcome accepted\|rejected\|needs_review\|partial\|errored` | None | Only export examples with this outcome after label assignment. |
| `--quality good\|bad\|mixed\|unknown` | None | Only export examples with this quality after label assignment. |
| `--workflow-status pending\|running\|succeeded\|failed` | None | Only export traces with this workflow status. |
| `--require-tool-call` | Disabled | Only export traces that include at least one tool call. |
| `--max-examples INTEGER` | None | Optional maximum number of examples to export. |

Trace export redacts common secret-looking keys and token values before writing
examples. The default `review` mode is intentionally not training-ready; curate
and relabel examples before SFT export. For real collection runs, prefer
`--label-mode reviewed --outcome accepted --quality good` so raw rejected traces
do not enter SFT data by accident.

## `micro-agent dataset review-trace`

```text
micro-agent dataset review-trace [OPTIONS]
```

Records a human review decision in `.micro_model_agent/traces/reviews.jsonl`
without modifying the raw workflow trace.

| Option | Default | Description |
| --- | --- | --- |
| `--trace-id TEXT` | Required | Stored workflow trace id to review. |
| `--outcome accepted\|rejected\|needs_review\|partial\|errored` | Required | Human outcome label. |
| `--quality good\|bad\|mixed\|unknown` | Required | Human quality label. |
| `--failure-mode VALUE` | None | Failure mode label. Can be passed more than once. |
| `--reviewer-notes TEXT` | None | Human review notes for this trace. |
| `--corrected-target-json TEXT` | None | Optional corrected dataset target JSON object for this trace. |
| `--corrected-target-file PATH` | None | Optional file containing a corrected dataset target JSON object. |
| `--output PATH` | `.micro_model_agent/traces/reviews.jsonl` | Append-only human trace review JSONL path. |

## `micro-agent dataset relabel`

```text
micro-agent dataset relabel [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--path PATH` | Required | Input JSONL dataset path to relabel. |
| `--output PATH` | Required | Output JSONL path for relabeled examples. |
| `--trace-id TEXT` | None | Only relabel the example with this `metadata.trace_id`. |
| `--source TEXT` | None | Only relabel examples with this exact source. |
| `--input-outcome accepted\|rejected\|needs_review\|partial\|errored` | None | Only relabel examples currently carrying this outcome. |
| `--input-quality good\|bad\|mixed\|unknown` | None | Only relabel examples currently carrying this quality. |
| `--outcome accepted\|rejected\|needs_review\|partial\|errored` | None | New outcome label for matched examples. |
| `--quality good\|bad\|mixed\|unknown` | None | New quality label for matched examples. |
| `--failure-mode VALUE` | None | Replacement failure mode label. Can be passed more than once. |
| `--reviewer-notes TEXT` | None | Replacement reviewer notes for matched examples. |

## `micro-agent dataset merge`

```text
micro-agent dataset merge [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--input PATH` | Required | Input JSONL dataset path. Pass more than once. |
| `--output PATH` | Required | Output JSONL path for the merged dataset. |
| `--deduplicate-by source\|id` | `source` | Dedupe key. |
| `--validate / --no-validate` | `--validate` | Validate the merged dataset before reporting success. |

## `micro-agent train synthetic`

```text
micro-agent train synthetic [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--base-model TEXT` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Local or Hugging Face base model id. |
| `--dataset PATH` | `.micro_model_agent/datasets/synthetic_seed.jsonl` | Validated synthetic dataset path. |
| `--output-dir PATH` | `.micro_model_agent/training/runs/latest` | Training run output directory. |
| `--dry-run / --no-dry-run` | `--dry-run` | Validate config and write a dry-run artifact. |
| `--max-steps INTEGER` | `20` | Maximum optimizer steps for real training. Minimum: 1. |
| `--batch-size INTEGER` | `1` | Per-device train batch size. Minimum: 1. |
| `--gradient-accumulation-steps INTEGER` | `4` | Gradient accumulation steps for real training. Minimum: 1. |
| `--learning-rate FLOAT` | `0.0002` | Learning rate for real training. Minimum: 0. |
| `--max-seq-length INTEGER` | `1024` | Maximum tokenized sequence length. Minimum: 128. |
| `--lora-r INTEGER` | `16` | LoRA rank for real training. Minimum: 1. |
| `--lora-alpha INTEGER` | `32` | LoRA alpha for real training. Minimum: 1. |
| `--lora-dropout FLOAT` | `0.05` | LoRA dropout. Range: 0 to 1. |

Training artifacts include the source dataset path, source dataset SHA-256,
exported SFT JSONL SHA-256, and dataset tool-profile summary.

## `micro-agent eval synthetic`

```text
micro-agent eval synthetic [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run-id TEXT` | `latest` | Training run id, run name under `.micro_model_agent/training/runs/`, or direct run directory path. |
| `--dataset PATH` | `examples/synthetic-data/held-out.behavior.jsonl` | Held-out synthetic JSONL dataset to evaluate against. |
| `--model TEXT` | None | Ollama model name to evaluate. |
| `--base-model TEXT` | None | Transformers base model for direct PEFT adapter evaluation. |
| `--adapter-path PATH` | None | Local PEFT adapter path for direct Transformers evaluation. |
| `--ollama-base-url TEXT` | None | Ollama host URL. Defaults to `MICRO_MODEL_AGENT_OLLAMA_BASE_URL`. |
| `--max-new-tokens INTEGER` | `384` | Maximum generated tokens per evaluation example. Range: 1 to 4096. |
| `--max-examples INTEGER` | None | Optional cap on evaluated examples. Minimum: 1. |
| `--pass-threshold FLOAT` | `0.8` | Minimum average behavioral score required to pass. Range: 0 to 1. |
| `--scripted-response TEXT` | None | Scripted JSON model response. Can be passed more than once. |
| `--scripted-response-file PATH` | None | JSONL file containing scripted model responses for evaluation tests. |
| `--output PATH` | `<run>/evaluation.json` | Evaluation report path. |

## `micro-agent eval traces`

```text
micro-agent eval traces [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run-id TEXT` | `latest` | Training run id, run name under `.micro_model_agent/training/runs/`, or direct run directory path. |
| `--dataset PATH` | `examples/trace-data/held-out.trace.jsonl` | Held-out trace-derived JSONL dataset to evaluate against. |
| `--model TEXT` | None | Ollama model name to evaluate. |
| `--base-model TEXT` | None | Transformers base model for direct PEFT adapter evaluation. |
| `--adapter-path PATH` | None | Local PEFT adapter path for direct Transformers evaluation. |
| `--ollama-base-url TEXT` | None | Ollama host URL. Defaults to `MICRO_MODEL_AGENT_OLLAMA_BASE_URL`. |
| `--max-new-tokens INTEGER` | `512` | Maximum generated tokens per evaluation example. Range: 1 to 4096. |
| `--max-examples INTEGER` | None | Optional cap on evaluated examples. Minimum: 1. |
| `--pass-threshold FLOAT` | `0.8` | Minimum average trace behavior score required to pass. Range: 0 to 1. |
| `--scripted-response TEXT` | None | Scripted JSON model response. Can be passed more than once. |
| `--scripted-response-file PATH` | None | JSONL file containing scripted model responses for evaluation tests. |
| `--output PATH` | `<run>/evaluation.json` | Evaluation report path. |

Synthetic and trace evaluation reports include `details.evaluation_metadata`
with the evaluated dataset path, provider type, and tool-profile summary.
Trace evaluation scores final responses, exact patch text, and expected tool
call order when those fields are available. Keep held-out trace fixtures
separate from curated examples that are merged into training data.

## `micro-agent eval workspace-staged`

```text
micro-agent eval workspace-staged [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run-id TEXT` | `latest` | Training run id, run name under `.micro_model_agent/training/runs/`, or direct run directory path. |
| `--dataset PATH` | `examples/workspace-eval/held-out.workspace-staged.jsonl` | Held-out staged workspace JSONL dataset to evaluate against. |
| `--model TEXT` | None | Ollama model name to evaluate. |
| `--base-model TEXT` | None | Transformers base model for direct PEFT adapter evaluation. |
| `--adapter-path PATH` | None | Local PEFT adapter path for direct Transformers evaluation. |
| `--ollama-base-url TEXT` | None | Ollama host URL. Defaults to `MICRO_MODEL_AGENT_OLLAMA_BASE_URL`. |
| `--max-new-tokens INTEGER` | `1024` | Maximum generated tokens per evaluation example. Range: 1 to 4096. |
| `--max-examples INTEGER` | None | Optional cap on evaluated examples. Minimum: 1. |
| `--pass-threshold FLOAT` | `0.8` | Minimum average staged workspace score required to pass. Range: 0 to 1. |
| `--scripted-response TEXT` | None | Scripted JSON model response. Can be passed more than once. |
| `--scripted-response-file PATH` | None | JSONL file containing scripted model responses for evaluation tests. |
| `--output PATH` | `<run>/evaluation.json` | Evaluation report path. |

This dry-run suite asks the model for one JSON object with five staged fields:
`read_search`, `diagnosis`, `patch_proposal`, `test_selection`, and
`final_summary`. Reports include separate metrics for read/search accuracy,
plan-before-patch accuracy, dry-run patch proposal accuracy, focused test
selection, and final summary accuracy. This suite is evidence about workspace
reasoning, not a promotion gate by itself.

Staged scenario records may include `input.workspace_files`, a JSON object whose
keys are repository-relative paths and whose values are the relevant file
contents or excerpts. Process-rich training records should also include
`target.gold_response`, the five-stage JSON answer used for SFT export, plus
`target.stages`, the looser scoring rubric used by evaluation. The evaluator
includes this virtual filesystem in the model prompt and review queue so
scenario batches can be self-contained.

## `micro-agent eval review-workspace-staged`

```text
micro-agent eval review-workspace-staged [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--dataset PATH` | `examples/workspace-eval/held-out.workspace-staged.jsonl` | Staged workspace scenario JSONL dataset. |
| `--report PATH` | Required | Staged workspace evaluation JSON report. Can be passed more than once. |
| `--output PATH` | `.micro_model_agent/datasets/workspace_staged_review_queue.jsonl` | Output JSONL review queue path. |
| `--simple-failure-threshold FLOAT` | `0.4` | Auto-reject examples whose best model score is at or below this value. |
| `--auto-accept-threshold FLOAT` | `0.95` | Mark examples as auto-accept candidates at or above this best score. |
| `--interactive` | Disabled | Prompt for a human decision and notes for each review record. |

This command builds a review queue from one or more staged evaluation reports.
Each JSONL record includes the goal, virtual filesystem, expected stages, model
outputs, stage scores, and an `auto_triage` decision. Use the non-interactive
mode to weed out obvious failures before human review, then rerun with
`--interactive` when you want to record decisions and notes from the terminal.

## `micro-agent eval compare`

```text
micro-agent eval compare [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--baseline-report PATH` | Required | Persisted evaluation JSON report for the base model. |
| `--adapter-report PATH` | Required | Persisted evaluation JSON report for the trained adapter. |
| `--minimum-score-delta FLOAT` | `0.0` | Minimum adapter score improvement over baseline. |
| `--minimum-metric-delta TEXT` | None | Required metric improvement as `metric_name=delta`. Can be passed more than once. |
| `--require-adapter-passed / --allow-failing-adapter` | `--require-adapter-passed` | Require the adapter report itself to pass before comparing improvements. |
| `--output PATH` | `<adapter-report>.comparison.json` | Comparison report path. |

The command compares the top-level score and any shared numeric values under
`details.metrics`, writes a JSON comparison report, and exits nonzero when the
adapter misses the requested improvement thresholds. Example:

```bash
uv run micro-agent eval compare \
  --baseline-report .micro_model_agent/training/runs/base-qwen-tool-profile/synthetic-evaluation.json \
  --adapter-report .micro_model_agent/training/runs/qwen-tool-profile-proof/synthetic-evaluation.json \
  --minimum-score-delta 0.00 \
  --minimum-metric-delta correct_tool_rate=0.10
```

## `micro-agent promote gate`

```text
micro-agent promote gate [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run-id TEXT` | `latest` | Training run id, run name under `.micro_model_agent/training/runs/`, or direct run directory path. |
| `--evaluation-report PATH` | `<run>/evaluation.json` | Evaluation JSON report to require. Can be passed more than once. |
| `--minimum-score FLOAT` | `0.8` | Minimum score each evaluation report must meet. Range: 0 to 1. |

The gate loads the run artifact metadata, applies the minimum-score promotion
policy to every required report, writes `promotion.json`, and exits nonzero
when any report fails or scores below the threshold.

## `micro-agent promote record`

```text
micro-agent promote record [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run-id TEXT` | `latest` | Training run id, run name under `.micro_model_agent/training/runs/`, or direct run directory path. |
| `--promotion-report PATH` | `<run>/promotion.json` | Passing promotion report path to record. |
| `--registry PATH` | `.micro_model_agent/training/promoted_models.jsonl` | Local JSONL registry path for approved artifacts. |
| `--reviewer-notes TEXT` | None | Human review note to store with the registry entry. |
| `--approved-by TEXT` | None | Reviewer or process that approved this artifact. |

This command records a gate-passing artifact in a local registry. It does not
change runtime defaults or package the model.

## `micro-agent promote list`

```text
micro-agent promote list [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--registry PATH` | `.micro_model_agent/training/promoted_models.jsonl` | Local JSONL registry path for approved artifacts. |

## `micro-agent promote select`

```text
micro-agent promote select [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--artifact-id TEXT` | Required | Promoted artifact id from the local registry. |
| `--repository-root PATH` | `.` | Repository root whose local model defaults should be updated. |
| `--registry PATH` | `.micro_model_agent/training/promoted_models.jsonl` | Local JSONL registry path for approved artifacts. |
| `--confirm` | Disabled | Required explicit confirmation before changing local defaults. |

The command selects an already recorded promoted artifact as the repository-local
default adapter by updating `.micro_model_agent/config.json`. It never promotes
an artifact by itself; run `promote gate` and `promote record` first.

## `micro-agent promote package-ollama`

```text
micro-agent promote package-ollama [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--artifact-id TEXT` | Required | Promoted artifact id from the local registry. |
| `--model-name TEXT` | Required | Ollama model name to create, such as `micro-agent-proof:qwen`. |
| `--registry PATH` | `.micro_model_agent/training/promoted_models.jsonl` | Local JSONL registry path for approved artifacts. |
| `--output-dir PATH` | `.micro_model_agent/training/ollama/<model-name>` | Directory for the generated Modelfile and package manifest. |
| `--ollama-base-model TEXT` | Artifact base model | Ollama `FROM` value. Use an Ollama model tag, GGUF file, or supported local model directory compatible with the adapter. |
| `--create` | Disabled | Run `ollama create` after writing the Modelfile. |

By default, this command writes a Modelfile and `ollama-package.json` manifest
without invoking Ollama. Pass `--create` only after verifying the base model and
adapter path are compatible.
