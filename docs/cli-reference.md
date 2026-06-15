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
| `micro-agent index` | Index the current repository. Currently prints that indexing is not implemented. |
| `micro-agent task` | Run a fixed coding-agent task with local fake dependencies and trace capture. |
| `micro-agent loop` | Run a model-driven agent loop with typed tool calls. |
| `micro-agent serve-mcp` | Serve MicroModelAgent over MCP. |
| `micro-agent dataset` | Dataset generation, validation, and export commands. |
| `micro-agent train` | Local training commands. |
| `micro-agent eval` | Evaluation commands. |

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
micro-agent index
```

No command-specific options. This command is a placeholder and currently prints
that repository indexing is not implemented.

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
| `--model TEXT` | None | Ollama model name. Defaults to `MICRO_MODEL_AGENT_DEFAULT_MODEL`. |
| `--base-model TEXT` | None | Transformers base model for direct PEFT adapter inference. |
| `--adapter-path PATH` | None | Local PEFT adapter path for direct Transformers inference. |
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

Validation prints error count plus category, kind, and outcome distributions.

## `micro-agent dataset export`

```text
micro-agent dataset export [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--path PATH` | `.micro_model_agent/datasets/synthetic_seed.jsonl` | Validated dataset path to export. |
| `--format TEXT` | `sft-jsonl` | Dataset export format. Currently only `sft-jsonl` is supported. |
| `--output PATH` | `.micro_model_agent/datasets/synthetic_seed.sft.jsonl` | Output path for exported dataset. |

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

