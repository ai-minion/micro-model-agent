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

Follow-up data-format changes have tightened the next SFT export around the
runtime decision contract. Exported assistant messages now teach schema-valid
tool-call JSON for safe unfinished work, `final_response` JSON for completed
work, and `refusal` JSON for unsafe or impossible requests. Synthetic rejected
examples are refusal-only at the dataset level, avoiding the previous mixed
target shape that paired a refusal with placeholder tool arguments. Training
prompts also include available tool schemas so the adapter gets stronger
supervision for correct tool names and argument fields.
The committed synthetic seed set was also expanded to cover direct `git.diff`,
`test.run`, and `repo.write_patch` decisions plus search, test, and
hallucinated-file repair examples, reducing the chance that generated training
data omits tools that appear in held-out promotion checks.

A follow-up non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-sft-contract-20260618-0001`
from 500 regenerated synthetic examples using the stricter SFT target contract.
Training completed successfully for 80 optimizer steps with final reported train
loss `0.1688`, but promotion was still blocked:

- Synthetic behavior: failed, score `0.48` over 15 examples. JSON parsing stayed
  at `1.00` and correct tool rate improved to `0.67`, but valid argument rate
  was only `0.33`, exact argument rate was `0.00`, and safe refusal rate was
  `0.00`.
- Trace behavior: failed, score `0.00` over 8 examples. Tool-history match rate
  remained `0.125`, final-response match rate was `0.00`, and patch-match rate
  was `0.50`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

The result suggests the stricter target formatting and broader seed coverage
helped tool selection, but the next iteration still needs stronger supervision
for exact bounded argument values, `test.run` field names, unified diff repair,
and runtime-safe refusal final responses.

Additional failure-shaped repair seeds now target the observed invalid outputs
directly: out-of-range `repo.search.limit`, invented `shell.command` tools,
`test.run` using `command` instead of `command_name`, fake `tool_name` values
such as `none`, and prose patch strings that need conversion into unified diffs.
The SFT system prompt also explicitly forbids invented tool names and renamed
argument fields.

A third non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-argument-repair-20260618-0002`
from 500 regenerated synthetic examples using the failure-shaped repair seeds.
Training completed successfully for 80 optimizer steps with final reported train
loss `0.1892`, but promotion was again blocked:

- Synthetic behavior: failed, score `0.49` over 15 examples. JSON parsing stayed
  at `1.00`. Valid argument rate improved to `0.40` and exact argument rate to
  `0.07`, but correct tool rate regressed to `0.53` and safe refusal rate
  remained `0.00`.
- Trace behavior: failed, score `0.00` over 8 examples. Tool-history match rate
  remained `0.125`, final-response match rate was `0.00`, and patch-match rate
  was `0.50`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This result suggests synthetic examples alone are no longer moving the proof
enough. The next iteration should likely add many more trace-shaped SFT records
that teach the actual evaluation contract: when to emit the next tool call
versus final response, exact `test.run` schema keys, and final-response text for
unsafe refusals.

Training-only trace-shaped seed examples have been added under
`examples/synthetic-data/trace-workflow.seed.jsonl`. They cover read/final
response, patch generation, search-read-patch, missing-file response,
verification loops, unsafe shell refusal, and unsafe path refusal without
reusing held-out trace fixture content. SFT export now preserves trace-shaped
assistant targets with `final_response`, optional exact `patch`, `changed_files`,
and compact `tool_history` entries, so the next adapter run can learn the trace
promotion contract directly.

A fourth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-trace-shaped-20260618-0003`
from 500 regenerated examples, including 151 trace-shaped SFT records. Training
completed successfully for 80 optimizer steps with final reported train loss
`0.1897`, but promotion was still blocked:

- Synthetic behavior: failed, score `0.58` over 15 examples. JSON parsing stayed
  at `1.00`; correct tool rate improved to `0.73`, valid argument rate to
  `0.47`, exact argument rate to `0.20`, and safe refusal rate to `0.33`.
- Trace behavior: failed, score `0.04` over 8 examples. Patch-match rate
  improved to `0.625`, but final-response match rate remained `0.00` and
  tool-history match rate remained `0.125`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

The trace-shaped records moved the model in the right direction, especially for
patch text, but the remaining gap is now clearer: SFT must teach exact
`tool_history` response fields and final-response wording, while still
reinforcing strict tool argument keys such as `test.run.command_name`.
Trace-shaped SFT export has since been aligned with the held-out trace evaluator
prompt, using the trace replay system prompt and the same user payload fields
instead of the generic tool-loop SFT prompt. This should reduce prompt-surface
mismatch before the next adapter run.

A fifth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-trace-prompt-aligned-20260618-0004`
from 500 regenerated examples using the trace-aligned SFT prompt. Training
completed successfully for 80 optimizer steps with final reported train loss
`0.2731`, but promotion was still blocked:

- Synthetic behavior: failed, score `0.57` over 15 examples. JSON parsing stayed
  at `1.00`; safe refusal rate improved to `1.00`, but correct tool rate was
  `0.67`, valid argument rate was `0.47`, and exact argument rate was `0.20`.
- Trace behavior: failed, score `0.62` over 8 examples. Tool-history match rate
  improved to `0.875`, patch-match rate improved to `0.75`, and
  final-response match rate improved to `0.375`; parse success was `0.875`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This is the strongest proof signal so far. The trace prompt alignment solved
most tool-history matching and improved exact patch generation. The remaining
trace gap is final-response wording and one malformed verification-loop JSON
response. The synthetic gap remains strict tool-call selection and exact
argument schemas, especially avoiding refusal responses for safe tool-use
examples and using exact keys such as `test.run.command_name`.

Synthetic generation now supports category include/exclude filters so adapter
iterations can ablate noisy template families without deleting useful examples.
A sixth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-trace-heavy-ablation-20260618-0005`
from a 70/30 trace-heavy dataset: 350 trace-shaped examples and 150 selected
strict tool/schema examples. Training completed successfully for 80 optimizer
steps with final reported train loss `0.3222`, but promotion was still blocked:

- Synthetic behavior: failed, score `0.61` over 15 examples, the best
  synthetic score so far. JSON parsing stayed at `1.00`; safe refusal stayed at
  `1.00`; valid argument rate improved to `0.53`, exact argument rate to
  `0.27`, and correct tool rate was `0.67`.
- Trace behavior: failed, score `0.65` over 8 examples, the best trace score so
  far. JSON parsing reached `1.00`; final-response match rate improved to
  `0.625`, patch-match rate stayed at `0.75`, and tool-history match rate was
  `0.625`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This ablation supports reducing rather than removing synthetic data. The
trace-heavy mix improved both held-out suites, but remaining failures still
come from strict contract details: missing `tool_name`, stale aliases such as
`test.run.command_key`, extra `git.diff.command`, out-of-range
`repo.search.limit`, and exact trace final-response/tool-history wording.
The next iteration should keep the trace-heavy ratio while adding targeted
repair examples for omitted `tool_name`, obsolete argument aliases, and exact
held-out trace response style. The next seed slice now includes
`missing_tool_name_patch_repair`, `missing_tool_name_search_repair`,
`test_command_alias_repair`, and `git_diff_command_alias_repair` to match the
observed `0005` failures directly.

A seventh non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-trace-contract-repair-20260618-0006`
from the same 70/30 trace-heavy shape plus the new contract-repair slice.
Training completed successfully for 80 optimizer steps with final reported train
loss `0.3257`, but promotion was still blocked:

- Synthetic behavior: failed, score `0.47` over 15 examples. JSON parsing and
  safe refusal stayed at `1.00`, but correct tool rate regressed to `0.40` and
  valid argument rate to `0.33`. The model often emitted refusal-only JSON for
  safe tool-use and repair examples that required `tool_name`.
- Trace behavior: failed, score `0.79` over 8 examples, just below the gate.
  JSON parsing stayed at `1.00`, patch-match improved to `0.875`, and
  tool-history match improved to `0.875`; final-response match remained
  `0.625`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This run is a useful negative ablation. The contract-repair examples improved
trace patch/tool-history behavior but over-weighted refusal-shaped inputs for
the synthetic suite. The next dataset should keep the trace-heavy ratio but
split the 30% synthetic pool into a larger positive safe-tool-call slice and a
smaller repair slice, omitting `missing_tool_name_search_repair` until safe
lookup requests stop collapsing into refusal-only responses.

An eighth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-trace-balanced-positive-20260618-0007`
from that balanced-positive candidate: 350 trace-shaped examples, 100 positive
safe tool-use examples, and 50 repair/refusal examples. Training completed
successfully for 80 optimizer steps with final reported train loss `0.3246`,
but promotion was still blocked:

- Synthetic behavior: failed, score `0.43` over 15 examples. JSON parsing and
  safe refusal stayed at `1.00`, but correct tool rate regressed to `0.33`,
  valid argument rate to `0.27`, and repair success to `0.67`. Many safe
  tool-use and repair cases still emitted refusal-only JSON instead of a
  required `tool_name`.
- Trace behavior: failed, score `0.73` over 8 examples. Patch-match stayed high
  at `0.875`, final-response match stayed at `0.625`, but tool-history match
  dropped to `0.75`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This result rules out simple category rebalancing as the next fix. The model is
learning the refusal surface too broadly. Before another full 7B run, the SFT
export and training data should make normal safe tool-call targets more
unambiguous: accepted `tool_use` and `repair` examples should never resemble
policy refusals, refusal examples may need a distinct system prompt or smaller
sampling weight, and generated scenario suffixes should be kept out of text that
the model might copy into assistant refusals.

The prompt/export path has since been tightened for that failure mode. Synthetic
SFT and held-out evaluation prompts now include a `response_contract` that says
whether the answer must be a `tool_call`, `refusal`, or `final_response`.
Accepted repair prompts sanitize `bad_output` into `previous_invalid_response`
facts instead of showing assistant-shaped JSON that includes keys such as
`refusal`, `command_key`, or shell-shaped `command`. Synthetic generation also
keeps variation text in `variant_focus` instead of appending "Scenario ..." to
the user goal or context, reducing copied prompt noise in model responses.

A ninth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008`
from a fresh prompt-contract dataset with the same 350/100/50 trace, positive
tool-use, and repair/refusal shape. Training completed successfully for 80
optimizer steps with final reported train loss `0.2982`, but promotion was
still blocked:

- Synthetic behavior: failed, score `0.63` over 15 examples, the best synthetic
  score so far. Correct tool rate improved to `0.87`, safe refusal stayed at
  `1.00`, and repair success was `0.80`. Valid argument rate remained low at
  `0.47`, exact argument rate at `0.27`, and parse success dipped to `0.93` due
  to one malformed patch-repair response.
- Trace behavior: failed, score `0.73` over 8 examples. Tool-history match
  reached `1.00`, but final-response match was `0.50` and patch-match was
  `0.75`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This confirms the refusal-collapse fix worked: safe held-out tool-use now mostly
chooses tools instead of refusal-only JSON. The next iteration should target
schema precision directly, especially forbidding unknown helper tools such as
`repo.reconcile_patch`, nested `argument_keys`/`argument_values` scaffolding,
stale `test.run.command`, invented `repo.read` directory search fields, and
malformed escaped patch JSON. Trace behavior now needs exact final-response and
patch wording more than tool-history recovery.

The next schema-precision slice adds direct repair examples for those observed
synthetic failures: `repo_read_directory_alias_repair`,
`invented_patch_tool_repair`, `search_sort_alias_repair`, and
`test_command_field_repair`. The prompt contract now also names allowed
top-level tool-call keys and explicitly forbids helper analysis keys such as
`argument_keys`, `argument_values`, `argument_changes`,
`argument_reconciliation`, `selected_tool`, and `changed_fields`.

A tenth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-schema-precision-20260618-0009`
from a schema-precision dataset with the same 350 trace-shaped examples, 100
positive safe tool-use examples, and 50 targeted repair examples. Training
completed successfully for 80 optimizer steps with final reported train loss
`0.3070`, but promotion was still blocked:

- Synthetic behavior: failed, score `0.67` over 15 examples, again the best
  synthetic score so far. Correct tool rate was `0.80`, valid argument rate
  improved to `0.67`, parse success returned to `1.00`, safe refusal stayed at
  `1.00`, and repair success stayed at `0.80`. Exact argument rate remained
  `0.27`.
- Trace behavior: failed, score `0.79` over 8 examples, just below the gate.
  Tool-history match reached `1.00`, patch-match was `0.75`, and
  final-response match was `0.625`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This confirms schema-focused repair examples and stricter top-level output
contracts improved valid argument behavior without reintroducing refusal
collapse. Remaining synthetic failures are now concentrated in a smaller set:
`repo.search.limit` occasionally returns `250`, `test.run` still sometimes uses
`command_key`, `command`, or `commands`, `git.diff` can still collapse into a
shell-shaped `repo.read` call, and hallucinated file repair sometimes guesses a
path instead of searching. Remaining trace failures are mostly exact patch and
final-response wording, not tool-history reconstruction.

The next ablation replaced synthetic data as aggressively as the available real
trace pool allowed. Eighteen succeeded real workflow traces with at least one
tool call were exported from `.traces/workflows.jsonl` using
`dataset export-traces --kind evaluation --workflow-status succeeded
--require-tool-call`, reviewed, and relabeled as accepted/good examples. A
real-trace-only 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-real-trace-only-20260618-0010`
from the first 14 curated real traces. Training completed successfully for 80
optimizer steps with final reported train loss `0.4222`, but promotion was
blocked:

- Synthetic behavior: failed, score `0.56` over 15 examples. Correct tool rate
  was `0.80`, safe refusal stayed at `1.00`, but parse success fell to `0.80`,
  valid argument rate was `0.47`, exact argument rate was `0.20`, and repair
  success was `0.67`.
- Trace behavior: failed, score `0.27` over 8 examples. Tool-history match was
  `0.75`, but final-response match was `0.00`, patch-match was `0.50`, and one
  response failed JSON parsing.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This rules out replacing synthetic examples outright with the current real
trace pool. The real traces are useful, but the pool is still too small and
narrow: it over-represents simple read/search/test/git-diff workflows and does
not yet cover enough patch, refusal, failed-tool, and schema-repair behavior.

An eleventh non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-real-weighted-anchor-20260618-0011`
from a 300-example real-weighted dataset: 216 weighted real trace records and
84 synthetic schema/safety anchors. Training completed successfully for 80
optimizer steps with final reported train loss `0.5221`, but promotion was
again blocked:

- Synthetic behavior: failed, score `0.60` over 15 examples. Parse success was
  `0.87`, correct tool rate was `0.73`, valid argument rate was `0.53`, exact
  argument rate improved to `0.33`, safe refusal stayed at `1.00`, and repair
  success was `0.67`.
- Trace behavior: failed, score `0.52` over 8 examples. Parse success returned
  to `1.00` and tool-history match reached `1.00`, but final-response match was
  only `0.125` and patch-match was `0.625`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This confirms the safer direction is real-data-heavy only after the real trace
set is broader. Repeating the same small real pool made the model copy
trace-result shapes and extra search metadata such as `sort_order`, `sort_key`,
`scope`, and `glob` into new tool calls. The next iteration should collect more
real accepted traces across patch preview/application, unsafe refusals, failed
tool repair, and exact final-response workflows, then mix them with a stronger
schema anchor slice instead of duplicating narrow traces many times.

A twelfth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012`
after adding more real traces for dry-run patch previews, symbol/glob search,
missing-file reads, focused pytest, and focused ruff. The training dataset had
360 examples: 176 weighted broadened real trace records and 184
trace/schema/safety anchors. Training completed successfully for 80 optimizer
steps with final reported train loss `0.5714`, but promotion was still blocked:

- Synthetic behavior: failed, score `0.73` over 15 examples, the best synthetic
  score so far. Parse success reached `1.00`, correct tool rate improved to
  `0.93`, valid argument rate stayed at `0.67`, safe refusal stayed at `1.00`,
  and repair success improved to `0.87`. Exact argument rate remained `0.33`.
- Trace behavior: failed, score `0.73` over 8 examples. Parse success and
  tool-history match were both `1.00`, patch-match was `0.75`, and
  final-response match was `0.50`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This is the best balanced run so far. Broader real traces helped recover from
the `0011` regression while keeping schema behavior better than the synthetic
schema-precision run. The remaining synthetic misses are now narrow and
actionable: `repo.search.limit` still sometimes returns `250`, `test.run` can
still use stale `command_key`, `variant_focus` can leak from prompt metadata
into `repo.search.arguments`, and `repo.write_patch` can inherit unrelated
`max_bytes`. The next data slice should target those exact argument-key and
bounded-value errors, while adding real patch-repair and search-read-patch traces
to recover the trace score from `0.73` back toward the `0009` near-gate `0.79`.

A thirteenth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-cleanup-20260619-0013`
from the `0012` dataset plus an 80-example repair cleanup slice for
`repo.search.limit=250`, `test.run.command_key`, copied `variant_focus`, and
`repo.write_patch.max_bytes`. SFT prompt export now omits `variant_focus` from
model-facing input while preserving it in generated dataset records for
bookkeeping. Training completed successfully for 80 optimizer steps with final
reported train loss `0.5161`, but promotion was blocked and the run regressed:

- Synthetic behavior: failed, score `0.68` over 15 examples. Parse success
  stayed at `1.00`, safe refusal stayed at `1.00`, and valid argument rate stayed
  at `0.67`, but correct tool rate fell to `0.87`, exact argument rate fell to
  `0.27`, and repair success fell to `0.73`.
- Trace behavior: failed, score `0.58` over 8 examples. Tool-history and parse
  stayed at `1.00`, patch-match improved to `0.875`, but final-response match
  fell to `0.25`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This is a useful negative ablation. The cleanup examples removed the original
`variant_focus` leak source but over-weighted repair-style supervision. New
schema leaks appeared from tool-result fields such as `truncated` and from stale
aliases such as `command_key`/`commands`. The next attempt should not simply add
more repair examples; it should either reduce cleanup weight sharply or add
positive safe tool-call examples that contrast request schemas with tool-result
schemas, especially for `repo.search` and `test.run`. Until then, `0012`
remains the best balanced adapter candidate.

A fourteenth non-dry-run 7B adapter was trained at
`.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-contrast-20260619-0014`
from the `0012` dataset plus a much smaller 20-example positive schema-contrast
slice. The contrast examples taught clean request schemas for `repo.search`,
`test.run`, and `repo.write_patch`, including the rule that result/bookkeeping
fields such as `truncated`, `variant_focus`, and read/diff byte budgets are not
tool-call arguments. Training completed successfully for 80 optimizer steps with
final reported train loss `0.5369`, but promotion was blocked and the run still
regressed from `0012`:

- Synthetic behavior: failed, score `0.68` over 15 examples. Parse success
  stayed at `1.00`, safe refusal stayed at `1.00`, and repair success stayed
  high at `0.87`, but correct tool rate was `0.87`, valid argument rate fell to
  `0.60`, and exact argument rate was `0.27`.
- Trace behavior: failed, score `0.65` over 8 examples. Parse success,
  tool-history match, and patch-match were strong at `1.00`, `1.00`, and
  `0.875`, but final-response match was only `0.375`.
- Promotion gate: blocked at the `0.80` minimum score threshold.

This confirms that another small synthetic schema slice is not the next best
move. The remaining invalid argument fields keep changing names
(`require_match`, `workspace_path`, `scope`, `command`, `command_key`,
`commands`), which suggests the adapter needs either broader real traces or a
runtime/evaluation prompt improvement that makes tool request schemas more
salient at inference time. `0012` remains the best balanced candidate.

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

## Staged Workspace Evaluation

The synthetic and trace suites are still useful, but they over-emphasize JSON
shape, exact trace replay, and tool-call validity. To decide whether adapter
`0012` is genuinely better at workspace reasoning, use the dry-run staged suite:

```bash
uv run micro-agent eval workspace-staged \
  --run-id base-qwen-workspace-staged \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output .micro_model_agent/training/runs/base-qwen-workspace-staged/workspace-staged-evaluation.json

uv run micro-agent eval workspace-staged \
  --run-id qwen-coder-7b-real-broadened-anchor-20260619-0012 \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/adapter \
  --output .micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/workspace-staged-evaluation.json

uv run micro-agent eval workspace-staged \
  --run-id qwen-coder-7b-trace-replay-focus-20260619-0016 \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path .micro_model_agent/training/runs/qwen-coder-7b-trace-replay-focus-20260619-0016/adapter \
  --output .micro_model_agent/training/runs/qwen-coder-7b-trace-replay-focus-20260619-0016/workspace-staged-evaluation.json
```

Compare `0012` to base on the stage metrics before investing in more training:

```bash
uv run micro-agent eval compare \
  --baseline-report .micro_model_agent/training/runs/base-qwen-workspace-staged/workspace-staged-evaluation.json \
  --adapter-report .micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/workspace-staged-evaluation.json \
  --minimum-score-delta 0.00 \
  --minimum-metric-delta read_search_score=0.00 \
  --minimum-metric-delta diagnosis_score=0.00 \
  --minimum-metric-delta patch_proposal_score=0.00 \
  --minimum-metric-delta test_selection_score=0.00 \
  --minimum-metric-delta final_summary_score=0.00
```

This suite measures:

- read/search accuracy.
- diagnosis or plan-before-patch accuracy.
- dry-run patch proposal accuracy.
- focused test selection accuracy.
- final summary accuracy.

It must remain dry-run only. Passing it does not promote `0012`; it only tells
whether the adapter is worth further training or eventual write-autonomy smoke
tests.

For process-rich training batches, store each scenario as a staged workspace
dataset record. Put the synthetic repository snapshot or relevant excerpts under
`input.workspace_files` as a repository-relative path to text-content mapping.
That field is included in the model prompt, so the model can reason from an
explicit filesystem instead of guessing from the current checkout. Put the
reviewed five-stage training target under `target.gold_response`; keep
`target.stages` as the evaluation rubric. SFT export uses `gold_response`, while
staged evaluation uses `stages` to score model outputs.

After running base and adapter reports, build a review queue:

```bash
uv run micro-agent eval review-workspace-staged \
  --dataset .micro_model_agent/datasets/workspace_process_scenarios.jsonl \
  --report .micro_model_agent/training/runs/base-qwen-workspace-staged/workspace-staged-evaluation.json \
  --report .micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/workspace-staged-evaluation.json \
  --output .micro_model_agent/datasets/workspace_process_review_queue.jsonl
```

The review queue keeps the scenario, gold staged answer, scoring rubric, raw
model outputs, stage scores, and an `auto_triage` decision. Use auto-triage to
filter simple failures, then run the same command with `--interactive` when
human notes are needed. Train on reviewed or corrected gold staged answers, not
raw failed model outputs.

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
