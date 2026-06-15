# Usage Guide

This guide describes the system that exists today: a local Python package with
safe repository tools, a model-driven tool loop, trace capture, synthetic dataset
generation, local PEFT fine-tuning, and synthetic behavioral evaluation.

Some project docs still describe the broader roadmap. When they differ from this
guide, treat this guide as the current operating procedure.

For a full command and option listing, see
[`docs/cli-reference.md`](cli-reference.md).

## Current Shape

micro-model-agent has two runnable agent paths:

- `micro-agent task`: a fixed coding workflow that uses a static patch supplied
  on the command line. This is useful for exercising trace capture, patch
  validation, labeling, and dataset export without calling a real model.
- `micro-agent loop`: a model-driven tool loop. The model returns one JSON
  decision per turn, either a typed tool call or a final response. The Python
  orchestrator enforces tool availability, schemas, turn limits, trace capture,
  and repository safety.

The built-in tools are:

- `repo.search`: text, glob, or Python symbol search.
- `repo.read`: repository-relative text file reads with optional line ranges.
- `repo.semantic_search`: lexical context retrieval over code and docs.
- `repo.write_patch`: unified diff preview or application, constrained to the
  repository root.
- `test.run`: one allowlisted verification command by name.
- `git.diff`: read-only working tree diff inspection.

The training path is currently:

```text
committed synthetic templates
  -> generated JSONL dataset
  -> dataset validation with category/kind/outcome distribution checks
  -> SFT chat JSONL export
  -> dry-run metadata or local HF/PEFT LoRA adapter training
  -> held-out behavioral synthetic evaluation, with metadata-only fallback for dry runs
```

## Setup

Install Python 3.12 and `uv`, then create the development environment:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
```

Create local configuration:

```bash
cp .env.example .env
```

Useful variables in `.env`:

```text
MICRO_MODEL_AGENT_OLLAMA_BASE_URL=http://localhost:11434
MICRO_MODEL_AGENT_DEFAULT_MODEL=qwen2.5-coder:7b
MICRO_MODEL_AGENT_TRAINING_BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
HF_TOKEN=
```

Initialize repository-local metadata:

```bash
uv run micro-agent init \
  --default-model qwen2.5-coder:7b \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct
```

Generated local state is written under `.micro_model_agent/`. It should remain
outside source control.

## Running The Agent

Use `loop` for the real model-driven tool loop. With Ollama running locally:

```bash
uv run micro-agent loop "Read README.md and summarize the project status." \
  --model qwen2.5-coder:7b \
  --available-tool repo.read \
  --max-tool-calls 1 \
  --max-turns 4
```

The model must answer with JSON shaped like one of these:

```json
{"tool_name":"repo.read","arguments":{"files":[{"path":"README.md"}]},"reason":"Read the requested file."}
```

```json
{"final_response":"Concise answer to the user.","ok":true}
```

For deterministic smoke tests, bypass Ollama with scripted model responses:

```bash
uv run micro-agent loop "Read README.md." \
  --scripted-response '{"tool_name":"repo.read","arguments":{"files":[{"path":"README.md"}]},"reason":"Read the requested file."}' \
  --scripted-response '{"final_response":"README.md was read successfully.","ok":true}' \
  --available-tool repo.read \
  --max-turns 3
```

To allow test execution, register an allowlisted command name:

```bash
uv run micro-agent loop "Run the test suite." \
  --scripted-response '{"tool_name":"test.run","arguments":{"command_name":"pytest"},"reason":"Run the allowlisted tests."}' \
  --scripted-response '{"final_response":"Tests completed.","ok":true}' \
  --available-tool test.run \
  --verification-command pytest \
  --test-command uv \
  --test-command run \
  --test-command pytest
```

`repo.write_patch` accepts unified diffs. In MCP, patch application is forced to
dry-run unless `apply_patches` is explicitly true. In the CLI, use tool
availability and the tool arguments to keep writes deliberate.

## Traces And Labeled Examples

Every `task` and `loop` run writes a workflow trace to:

```text
.micro_model_agent/traces/workflows.jsonl
```

The `task` command can turn a completed workflow trace into a labeled dataset
example:

```bash
uv run micro-agent task "Apply this known patch." \
  --patch "--- a/example.py
+++ b/example.py
@@
-old
+new
" \
  --dataset-output .micro_model_agent/datasets/trace_examples.jsonl \
  --label accepted \
  --quality good
```

This path is intentionally static today: the patch comes from `--patch`, not
from Ollama or Transformers. Use it to collect labels and verify the
trace-to-dataset plumbing.

## Synthetic Dataset Workflow

Use this loop when changing dataset templates, validation, training, or
behavioral evaluation:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv run micro-agent dataset synthesize --count 500 --seed 2026
uv run micro-agent dataset validate
uv run micro-agent dataset export --format sft-jsonl
uv run micro-agent train synthetic \
  --output-dir .micro_model_agent/training/runs/synthetic-smoke
uv run micro-agent eval synthetic --run-id synthetic-smoke
```

`dataset validate` prints the total error count plus category, kind, and outcome
distributions. Use those counts as a quick balance check before training:

```text
validated 500 examples with 0 error(s)
Categories: arbitrary_shell_rejection=83, documentation_grounded=83, ...
Kinds: repair=83, tool_use=417
Outcomes: accepted=334, rejected=166
```

`dataset synthesize` balances categories by default and creates deterministic
scenario-text variants when `--seed` is provided. Use these switches when you
need a specific shape:

```bash
uv run micro-agent dataset synthesize \
  --count 100 \
  --seed 7 \
  --balance-categories \
  --vary-scenarios
```

Use `--no-balance-categories` to preserve raw template order, or
`--no-vary-scenarios` when you need exact copies of the seed prompt text with
fresh IDs.

Default paths:

```text
examples/synthetic-data/*.seed.jsonl
.micro_model_agent/datasets/synthetic_seed.jsonl
.micro_model_agent/datasets/synthetic_seed.sft.jsonl
```

The committed held-out behavioral fixture is separate from the training seed
templates:

```bash
uv run micro-agent dataset validate \
  --path examples/synthetic-data/held-out.behavior.jsonl
```

Use that fixture for stable scoreboard-style evaluation:

```bash
uv run micro-agent eval synthetic \
  --run-id synthetic-smoke \
  --dataset examples/synthetic-data/held-out.behavior.jsonl
```

The current synthetic generator cycles the hand-authored seed templates and
assigns fresh IDs. It does not yet use a model to invent new scenarios.

## Fine-Tuning

Install the optional training dependencies:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --group training
```

Run a fast dry run first. This validates the dataset and writes training
metadata without loading a model:

```bash
uv run micro-agent train synthetic \
  --dry-run \
  --output-dir .micro_model_agent/training/runs/dry-run-smoke
uv run micro-agent eval synthetic --run-id dry-run-smoke
```

Dry-run artifacts do not contain runnable adapter weights, so `eval synthetic`
falls back to a metadata-only smoke gate for those runs. Use a scripted response
file, Ollama model, or PEFT adapter when you want behavioral scoring.

Run local PEFT/LoRA fine-tuning with Transformers:

```bash
uv run micro-agent train synthetic \
  --no-dry-run \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir .micro_model_agent/training/runs/qwen-tool-schema-smoke \
  --max-steps 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 1024 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

Outputs are written under the run directory:

```text
.micro_model_agent/training/runs/<run-name>/
  synthetic.sft.jsonl
  adapter/
  checkpoints/
  artifact.json
  run.json
  evaluation.json
```

The real runner trains a PEFT adapter using Hugging Face Transformers, PEFT, and
the exported SFT JSONL. It saves adapter and tokenizer files to `adapter/`.

Evaluate the run:

```bash
uv run micro-agent eval synthetic --run-id qwen-tool-schema-smoke
```

Evaluation scores model behavior when a runnable model, adapter, or scripted
response source is available. Dry-run artifacts still use a metadata-only smoke
gate because they do not contain runnable model weights.

`--run-id` accepts either a direct path or a name under
`.micro_model_agent/training/runs/`. Prefer named run directories such as
`qwen-tool-schema-smoke`, `dry-run-smoke`, or a date-stamped name when comparing
experiments. The default `latest` path is convenient for quick local checks but
will be overwritten by the next default run.

## Behavioral Evaluation

Behavioral evaluation prompts a provider with held-out examples and expects one
JSON object per example: either a typed tool call or a safe refusal. Reports are
written to the selected run directory as `evaluation.json` and include:

- overall score and pass/fail
- global parse/tool/argument/refusal/repair/final-response metrics
- per-category metrics
- per-example raw responses, parsed responses, and errors

Run a deterministic scripted smoke test with one response per held-out example:

```bash
uv run micro-agent eval synthetic \
  --run-id scripted-held-out-smoke \
  --scripted-response-file .micro_model_agent/eval/scripted-held-out-responses.jsonl
```

Each line in the scripted response file should be a complete model response JSON
object. For example:

```json
{"tool_name":"repo.search","arguments":{"query":"BuiltinToolSpec","kind":"text","limit":25},"reason":"Search first because the relevant file is not explicitly known."}
```

Run against an Ollama model:

```bash
uv run micro-agent eval synthetic \
  --run-id ollama-held-out-smoke \
  --model qwen2.5-coder:7b \
  --max-examples 8
```

Run directly against a PEFT adapter:

```bash
uv run micro-agent eval synthetic \
  --run-id qwen-tool-schema-smoke \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/qwen-tool-schema-smoke/adapter
```

When behavioral evaluation fails, the CLI prints failing examples as
`category/example_id scored <score>`, then writes full details to
`evaluation.json`.

## Running With A Fine-Tuned Adapter

After training, run the tool loop directly through Transformers with the adapter:

```bash
uv run micro-agent loop "Read docs/architecture.md and summarize the dependency direction." \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/qwen-tool-schema-smoke/adapter \
  --available-tool repo.read \
  --max-tool-calls 1 \
  --max-turns 4
```

You can also record the adapter during initialization:

```bash
uv run micro-agent init \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/qwen-tool-schema-smoke/adapter
```

Packaging the adapter into an Ollama model is not implemented in this repository
yet. Today, adapter inference is handled by the direct Transformers provider.

## MCP Usage

Start the MCP server:

```bash
uv run micro-agent serve-mcp
```

The default public MCP tool is:

```text
micro_agent_run_loop
```

If `.micro_model_agent/config.json` is missing, the server also exposes:

```text
micro_agent_init
```

Set these environment variables for local inspection:

```text
MICRO_MODEL_AGENT_MCP_DEBUG_TOOLS=1
MICRO_MODEL_AGENT_MCP_EXPOSE_INIT=1
```

Debug mode exposes direct built-in-tool execution, trace reads, and built-in tool
listing. Keep debug tools local.

## Moving From Synthetic To Real-Life Fine-Tuning

Synthetic fine-tuning teaches tool schema, refusal habits, and workflow shape.
Real-life fine-tuning should teach what actually worked in repositories.

Use this progression:

1. Start with synthetic templates.
   Generate and train on small, explicit examples for tool selection, argument
   schemas, safe refusals, patch dry-runs, and verification repair.

2. Run real workflows and keep traces.
   Use `loop` for model-driven tool use and `task` for static patch workflows.
   Every run creates trace records under `.micro_model_agent/traces/`.

3. Label outcomes.
   Good training data needs labels: `accepted`, `rejected`, `partial`,
   `errored`, plus quality labels such as `good`, `bad`, or `mixed`. Rejected
   examples are useful when the failure mode is clear.

4. Convert high-quality traces into dataset examples.
   The implemented trace-to-dataset builder currently exists behind `task`
   labeling. The next production step is to add a CLI exporter that reads stored
   `loop` traces, redacts sensitive values, filters by labels, and writes
   trace-derived examples.

5. Mix synthetic and real examples.
   Keep synthetic records for invariants: path safety, JSON schema compliance,
   tool choice, and refusal behavior. Add real traces for local code conventions,
   patch size, test-failure repair, documentation compliance, and final response
   quality.

6. Hold out real examples for evaluation.
   Do not train on every accepted trace. Keep a validation/test split that
   includes real tasks and failures. Promotion should require behavioral checks,
   not just metadata.

7. Train small, compare, then expand.
   Run short adapter jobs first, compare against the base model and previous
   adapter, inspect traces, and only then increase data volume or training
   steps.

8. Promote manually.
   Record metrics and artifact metadata. Use a human review before changing the
   default adapter or MCP configuration.

The practical transition point is when real labeled traces outnumber the
synthetic templates for the behaviors you care about. Keep synthetic examples in
the mix as guardrails; let real traces become the main source for coding style,
repair behavior, and repository-specific judgment.

## What Is Not Built Yet

These items are described in roadmap docs but are not complete today:

- `micro-agent index` prints that indexing is not implemented.
- Synthetic generation creates deterministic template variants, but it is not
  model-authored generation.
- The held-out synthetic evaluation suite is still small and hand-authored.
- There is no automatic trace export command for all stored `loop` traces.
- There is no model registry or automatic promotion workflow.
- There is no Ollama packaging step for trained PEFT adapters.
- MCP defaults use direct Transformers adapter inference, not Ollama.
