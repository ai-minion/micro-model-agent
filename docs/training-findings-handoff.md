# Training Findings Handoff

This document summarizes the local trained-model proof work through run `0014`
and gives the next chat a focused starting prompt.

## Current Best Result

The best balanced adapter remains:

```text
.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012
```

Scores:

- Synthetic held-out: `0.73`
- Trace held-out: `0.7292`
- Promotion gate: blocked at `0.80`

Why it is the current best:

- Parse success reached `1.00` on both suites.
- Synthetic correct-tool rate reached `0.93`.
- Trace tool-history match reached `1.00`.
- Trace patch-match was usable at `0.75`.
- It avoided the heavy repair-data regressions seen in `0013` and `0014`.

## Key Findings

Synthetic examples should not be removed outright. The real-trace-only run
`0010` regressed badly: synthetic `0.56`, trace `0.27`. The current real trace
pool is too narrow to replace synthetic data by itself.

Repeating a small real trace pool too heavily is also harmful. The real-weighted
`0011` run improved over real-only but still regressed: synthetic `0.60`, trace
`0.52`. Repetition made the model copy trace-result shapes and extra metadata
into tool-call arguments.

Broader real traces helped. `0012` added more real traces for dry-run patch
preview, symbol/glob search, missing-file reads, focused pytest, and focused
ruff. It produced the best balanced result so far.

Repair slices are easy to overweight. The `0013` cleanup dataset added 80 repair
examples targeting `repo.search.limit=250`, `test.run.command_key`, copied
`variant_focus`, and `repo.write_patch.max_bytes`. It regressed to synthetic
`0.6767`, trace `0.5833`.

Small positive schema contrast was not enough. `0014` added only 20 positive
examples contrasting request schemas with tool-result fields. It recovered some
trace behavior versus `0013` but still regressed from `0012`: synthetic
`0.6767`, trace `0.6458`.

The `variant_focus` prompt leak is fixed and should stay fixed. Generated
dataset records may retain `variant_focus` for bookkeeping, but SFT/evaluation
prompts now omit it from model-facing input.

The remaining invalid fields keep changing names. Recent failures included
`limit=250`, `command_key`, `commands`, `require_match`, `workspace_path`,
`scope`, `truncated`, `type`, `command`, and `max_bytes` on the wrong tool. This
suggests more synthetic schema nudging is not the best next lever.

## Run Summary

```text
0009 schema precision
  synthetic 0.67
  trace     0.7917
  best trace score, but weaker synthetic than 0012

0010 real trace only
  synthetic 0.56
  trace     0.2708
  proved current real trace pool cannot replace synthetic data alone

0011 real weighted + anchors
  synthetic 0.6033
  trace     0.5208
  repeated narrow real traces too heavily

0012 broadened real + anchors
  synthetic 0.73
  trace     0.7292
  current best balanced run

0013 0012 + 80 repair cleanup examples
  synthetic 0.6767
  trace     0.5833
  negative ablation: too repair-heavy

0014 0012 + 20 positive schema contrast examples
  synthetic 0.6767
  trace     0.6458
  negative ablation: small synthetic schema nudge still worse than 0012
```

## Important Code/Data Changes

Prompt and export:

- `src/micro_model_agent/infrastructure/dataset_prompting.py`
  - New prompt payload helper.
  - Adds `response_contract`.
  - Sanitizes repair `bad_output`.
  - Omits `variant_focus` from model-facing prompt input.
- `src/micro_model_agent/infrastructure/dataset_validation.py`
  - SFT export uses strict assistant JSON targets.
  - Refusal targets export as refusal-only JSON.
  - Trace-shaped evaluation examples use trace-evaluation assistant shape.

Data generation and examples:

- `src/micro_model_agent/infrastructure/synthetic_data.py`
  - Category include/exclude filters.
  - Scenario variation goes into `variant_focus`, not the user goal/context.
- `examples/synthetic-data/tool-use.seed.jsonl`
  - Added schema repair and contrast examples.
- `examples/synthetic-data/trace-workflow.seed.jsonl`
  - Added training-only trace-shaped workflow records.
- `examples/synthetic-data/held-out.behavior.jsonl`
  - Updated held-out targets for stricter refusal/tool-call contracts.

Trace export:

- `src/micro_model_agent/infrastructure/trace_export.py`
  - Can filter by workflow status.
  - Can require at least one tool call.
- `src/micro_model_agent/interfaces/cli.py`
  - `dataset export-traces` supports `--kind`, `--workflow-status`, and
    `--require-tool-call`.
  - `dataset synthesize` supports repeated `--include-category` and
    `--exclude-category`.

Documentation:

- `docs/trained-model-proof-plan.md`
  - Full run history and proof interpretation.
- `docs/training-pipeline.md`
  - Data-format and real-trace guidance.
- `docs/training-session-command-log.md`
  - Commands used and why.

## Verification Status

Last full verification passed:

```bash
uv run pytest
uv run ruff check .
uv run mypy
git diff --check
```

The last pytest count was `143 passed`.

## Recommended Next Move

Do not start by adding more synthetic schema examples.

The best next step is one of these:

1. Collect broader real traces, especially exact trace final-response examples,
   search-read-patch workflows, unsafe refusals, and failed-tool repair flows.
2. Improve the inference/evaluation prompt so tool request schemas are more
   salient at generation time.
3. Add an evaluation diagnostic that separates request-schema errors from
   tool-result-field copying, so the next dataset can be smaller and more
   surgical.

The most promising path is likely:

```text
collect broader real traces
  -> keep 0012-like dataset balance
  -> add little or no synthetic schema nudging
  -> rerun short adapter
```

## Prompt For Next Chat

Copy this into the next chat:

```text
We are working in D:\Projects\code\micro-model-agent.

Use WSL for project commands:

cd /mnt/d/Projects/code/micro-model-agent
export UV_PROJECT_ENVIRONMENT=.venv-wsl

Repo context:
- Worktree is dirty with intentional training-pipeline changes.
- Do not revert unrelated changes.
- Preserve DDD boundaries:
  - domain framework-free
  - application uses ports
  - infrastructure owns providers/tools/storage
  - interfaces own CLI/MCP
- Use rg for search.
- Use apply_patch for manual edits.

Read first:
- README.md
- docs/training-findings-handoff.md
- docs/trained-model-proof-plan.md
- docs/training-session-command-log.md
- docs/training-pipeline.md
- docs/cli-reference.md
- docs/usage.md

Current best adapter run:
- .micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012
- Synthetic held-out score: 0.73
- Trace held-out score: 0.7292
- Promotion gate still blocked at 0.80

Important negative ablations:
- 0013 added an 80-example repair cleanup slice and regressed to synthetic 0.6767, trace 0.5833.
- 0014 added a 20-example positive schema contrast slice and regressed to synthetic 0.6767, trace 0.6458.
- Conclusion: more synthetic schema nudging is not the next best lever.

Keep:
- The variant_focus prompt fix. Generated records may keep variant_focus, but model-facing SFT/eval prompts should omit it.
- The trace export filters: --kind, --workflow-status, --require-tool-call.
- The response_contract and repair bad_output sanitization.

Next priority:
Improve beyond 0012 by collecting broader real traces or improving inference/evaluation prompting, not by adding a large synthetic repair slice.

Suggested starting path:
1. Generate/collect more real scripted loop traces for:
   - exact final-response wording
   - search-read-patch workflows
   - unsafe shell/path refusals
   - failed patch/read repair flows
   - valid test.run and git.diff request schemas
2. Export with:
   uv run micro-agent dataset export-traces \
     --trace-path .traces/workflows.jsonl \
     --output .micro_model_agent/datasets/<new_review_file>.jsonl \
     --kind evaluation \
     --workflow-status succeeded \
     --require-tool-call
3. Relabel reviewed succeeded examples accepted/good.
4. Build a new 0012-like dataset balance with broader real traces and little or no added synthetic schema slice.
5. Train one short 7B adapter run and evaluate:
   uv run micro-agent eval synthetic ...
   uv run micro-agent eval traces ...
   uv run micro-agent promote gate ...
6. Run:
   uv run pytest
   uv run ruff check .
   uv run mypy
   git diff --check

When reporting back, keep it concise:
- files changed
- dataset/run created
- synthetic and trace scores
- promotion result
- whether 0012 is still best
- next recommended move
```
