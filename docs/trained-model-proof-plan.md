# Trained Model Proof Plan

## Goal

The next project priority is to prove that a local model can be trained on a
configurable tool set, project patterns, and workspace context, then run through
the existing agent loop and MCP surface with measurably better behavior than the
base model.

The proof is not "a training command completed." The proof is a comparison:

```text
base local model
  vs.
trained local adapter
  on held-out tool-use, repair, and workspace tasks
```

The trained model should produce valid JSON tool decisions more often, choose
the right tool more often, obey repository safety rules, use workspace context,
and recover from common failures better than the base model.

## Current Position

The platform mechanics are mostly in place:

- `micro-agent loop` runs a model-driven tool loop.
- `micro-agent serve-mcp` exposes the system over MCP.
- built-in tools are typed, constrained, and configurable per run.
- traces are captured for loop and task workflows.
- synthetic datasets can be generated, validated, exported, trained, evaluated,
  and gated for manual promotion.
- local lexical workspace indexing exists through `micro-agent index`.

The unproven part is the learning loop:

- enough curated examples to teach the model the configured tools and workspace
  patterns.
- a real adapter that beats the base model on held-out examples.
- a promotion path that records a proven adapter and makes it selectable for
  agent-loop and MCP runs.

## Proof Scope

The first proof should stay narrow and observable.

### Tool Set

Use the initial coding-agent tools:

- `repo.search`
- `repo.read`
- `repo.semantic_search`
- `repo.write_patch`
- `test.run`
- `git.diff`

The dataset format should keep tool availability explicit so later runs can
train and evaluate different tool profiles.

### Behaviors

The first adapter should improve:

- tool-call JSON validity.
- correct tool selection.
- valid tool arguments.
- refusal of unsafe paths and arbitrary shell requests.
- use of retrieved workspace context.
- patch dry-run and verification flow.
- final response concision after tool use.
- recovery from invalid arguments, failed patches, and failed tests.

### Workspace Patterns

Training examples should include:

- current project documentation and architecture rules.
- actual tool schemas.
- local repository paths and module names.
- accepted and rejected workflow traces.
- repair examples from failed tool calls.
- held-out trace examples that never enter training.

## Success Gates

A trained adapter is considered proven only when all gates pass.

### Gate 1: Dataset Readiness

- Synthetic training set validates with zero schema errors.
- Curated trace examples have reviewed labels.
- Held-out examples are stored separately and are not merged into training.
- Dataset distribution is reported by category, kind, outcome, and source.

### Gate 2: Base Model Baseline

Run the same held-out suites against the untrained local model and save reports:

- held-out synthetic behavior.
- held-out trace behavior.
- at least one scripted or manual workspace task set.

The baseline creates the comparison target. Without it, adapter evaluation only
proves that the adapter can produce some passing outputs.

### Gate 3: Adapter Training

Run a real local PEFT/LoRA adapter job with recorded metadata:

- base model.
- dataset path and hash or version.
- tool profile.
- training parameters.
- adapter path.
- run metrics.

Dry runs remain useful for plumbing checks, but they do not satisfy this gate.

### Gate 4: Adapter Evaluation

Evaluate the trained adapter against the same held-out suites as the baseline.
Minimum first-pass targets:

- valid JSON decision rate: at least 95%.
- unsafe refusal pass rate: 100%.
- schema-valid tool arguments: at least 90%.
- correct tool choice: better than base model by at least 10 percentage points.
- overall held-out synthetic score: at least 0.80.
- overall held-out trace score: at least 0.80.

If the base model already exceeds a target, the adapter must match it and
improve another meaningful metric such as repair accuracy or final-response
quality.

### Gate 5: Agent Loop Smoke Test

Run the adapter through `micro-agent loop` with real tools enabled:

```bash
uv run micro-agent loop "Read docs/architecture.md and summarize the dependency direction." \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/<run-id>/adapter \
  --available-tool repo.read \
  --max-tool-calls 1 \
  --max-turns 4
```

The trace must show a valid tool call, successful tool result, and concise final
response.

### Gate 6: MCP Smoke Test

Run the same proven adapter through the MCP entrypoint and verify that:

- the MCP server starts cleanly.
- the agent loop can be invoked through MCP.
- configured tools are enforced.
- traces are written.
- write-capable tools remain dry-run or approval-gated by default.

## Recommended Command Flow

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --dev
uv sync --group training

uv run micro-agent index

uv run micro-agent dataset synthesize --count 500 --seed 2026
uv run micro-agent dataset validate
uv run micro-agent dataset export --format sft-jsonl

uv run micro-agent eval synthetic \
  --run-id base-qwen-tool-profile \
  --model qwen2.5-coder:7b \
  --output .micro_model_agent/training/runs/base-qwen-tool-profile/synthetic-evaluation.json

uv run micro-agent eval traces \
  --run-id base-qwen-tool-profile \
  --model qwen2.5-coder:7b \
  --output .micro_model_agent/training/runs/base-qwen-tool-profile/trace-evaluation.json

uv run micro-agent train synthetic \
  --no-dry-run \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir .micro_model_agent/training/runs/qwen-tool-profile-proof

uv run micro-agent eval synthetic \
  --run-id qwen-tool-profile-proof \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/qwen-tool-profile-proof/adapter \
  --output .micro_model_agent/training/runs/qwen-tool-profile-proof/synthetic-evaluation.json

uv run micro-agent eval compare \
  --baseline-report .micro_model_agent/training/runs/base-qwen-tool-profile/synthetic-evaluation.json \
  --adapter-report .micro_model_agent/training/runs/qwen-tool-profile-proof/synthetic-evaluation.json \
  --minimum-score-delta 0.00 \
  --minimum-metric-delta correct_tool_rate=0.10 \
  --minimum-metric-delta valid_argument_rate=0.00 \
  --output .micro_model_agent/training/runs/qwen-tool-profile-proof/synthetic-comparison.json

uv run micro-agent eval traces \
  --run-id qwen-tool-profile-proof \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/qwen-tool-profile-proof/adapter \
  --output .micro_model_agent/training/runs/qwen-tool-profile-proof/trace-evaluation.json

uv run micro-agent eval compare \
  --baseline-report .micro_model_agent/training/runs/base-qwen-tool-profile/trace-evaluation.json \
  --adapter-report .micro_model_agent/training/runs/qwen-tool-profile-proof/trace-evaluation.json \
  --minimum-score-delta 0.00 \
  --minimum-metric-delta tool_history_match_rate=0.00 \
  --output .micro_model_agent/training/runs/qwen-tool-profile-proof/trace-comparison.json

uv run micro-agent promote gate \
  --run-id qwen-tool-profile-proof \
  --evaluation-report .micro_model_agent/training/runs/qwen-tool-profile-proof/synthetic-evaluation.json \
  --evaluation-report .micro_model_agent/training/runs/qwen-tool-profile-proof/trace-evaluation.json
```

## Latest Local Proof Attempt

The existing non-dry-run 7B adapter at
`.micro_model_agent/training/runs/qwen-coder-7b-tool-schema-20260613-205520`
was evaluated against the committed held-out suites and promotion gate.

- Synthetic behavior: failed, score `0.38` over 15 examples. JSON parsing was
  available, but correct tool and argument rates were both `0.20`.
- Trace behavior: failed, score `0.00` over 8 examples. Tool-history match rate
  was `0.125`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

The proof is therefore complete as a negative result: the current adapter is not
promotable. The next training iteration should improve data quality and
supervised target formatting before another non-dry-run adapter run.

## Implementation Priorities

1. Add explicit baseline-vs-adapter comparison reporting. Done for persisted
   evaluation reports with `micro-agent eval compare`.
2. Expand held-out synthetic examples for unsafe requests, schema repair, patch
   repair, and verification loops.
3. Expand held-out trace fixtures with real accepted and rejected workflow
   traces.
4. Add tool-profile metadata to datasets, training runs, and evaluation reports.
   Done for dataset serialization/export metadata, training artifact metadata
   with dataset hashes, and synthetic/trace evaluation report details.
5. Add a command that selects a promoted adapter as the local default without
   bypassing manual approval. Done with `micro-agent promote select --confirm`,
   which only selects artifacts already recorded in the promotion registry.
6. Add MCP smoke documentation and tests for a promoted adapter path. Done for
   selected repository config resolution, MCP scripted smoke coverage, and
   documented promoted-adapter smoke checks.
7. Add Ollama packaging for promoted PEFT adapters after the direct Transformers
   adapter path is proven. Done as `micro-agent promote package-ollama`, which
   writes a Modelfile/package manifest for a recorded promoted artifact and only
   runs `ollama create` when explicitly requested.

## Non-Goals For This Proof

- General coding benchmark leadership.
- Cloud training orchestration.
- Automatic model promotion.
- Training on unreviewed private traces.
- Supporting every possible tool set before the first tool profile is proven.

## Decision Rule

Until the adapter beats the base model on held-out tool-use and trace behavior,
the project should prioritize data quality, evaluation coverage, and training
feedback over new agent features.
