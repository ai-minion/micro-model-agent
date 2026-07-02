# Phase 2: Move Evaluation Orchestration And Clean Rubrics

## Status

Complete.

Completed:

- Added application-owned synthetic behavior evaluation orchestration in
  `src/micro_model_agent/application/evaluation/workflows.py`.
- Converted `src/micro_model_agent/infrastructure/synthetic_evaluation.py` into
  a compatibility adapter that wires infrastructure scoring/tool contracts.
- Added application-owned trace-derived behavior evaluation orchestration in
  `src/micro_model_agent/application/evaluation/workflows.py`.
- Converted `src/micro_model_agent/infrastructure/trace_evaluation.py` into a
  compatibility adapter plus extracted trace scoring helpers.
- Added application-owned workspace-staged behavior evaluation orchestration in
  `src/micro_model_agent/application/evaluation/workflows.py`.
- Converted `src/micro_model_agent/infrastructure/workspace_staged_evaluation.py`
  into a compatibility adapter that wires the staged workspace rubric.
- Added application tests that exercise synthetic and trace model-call loops
  with fake scorers/providers and no infrastructure imports.
- Added application tests that exercise workspace-staged model-call loops with
  fake scorers/providers and no infrastructure imports.
- Removed concrete infrastructure imports from the pure synthetic, trace, and
  workspace-staged rubric modules.
- Moved pure synthetic, trace-derived, and workspace-staged scoring/rubric
  modules into `src/micro_model_agent/application/evaluation_rubrics/`.
- Kept the old flat application evaluation and rubric imports as compatibility
  facades with explicit public exports.
- Added direct rubric tests that run without model providers or filesystem
  fixtures.
- Added architecture coverage to prevent pure rubric modules from importing
  concrete infrastructure helpers.
- Added architecture coverage that keeps the old infrastructure rubric paths as
  compatibility shims over the application-owned modules.

Verification at the latest update:

```text
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m ruff check src docs'
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m pytest'
```

Both passed; the full test suite reported 313 passing tests.

## Goal

Make evaluation use cases application-owned and keep infrastructure focused on
adapters, external mechanisms, and persistence. Rubric/scoring code should be
testable without model providers or filesystem setup.

## Current Mismatch

`src/micro_model_agent/application/evaluation/workflows.py` coordinates dataset
loading, metadata, and report writing through ports. That is good.

The model-call loops for synthetic, trace-derived, and workspace-staged
behavior now live in application code. Their infrastructure modules remain as
compatibility adapters that wire concrete scoring/rubric helpers and tool
contract defaults.

The rubric ownership gap has been closed:

- `application/evaluation_rubrics/workspace_staged.py` owns staged workspace
  scoring without importing dataset metadata, response parsing, or tool catalog
  adapters.
- `application/evaluation_rubrics/synthetic.py` owns synthetic scoring and
  accepts tool contracts from its adapter instead of importing `tools.catalog`.
- `application/evaluation_rubrics/trace.py` owns trace-derived scoring.
- Old flat application and `infrastructure/evaluation/*_rubric.py` paths remain
  compatibility shims.

## Suggested Shape

One possible package shape after this phase:

```text
domain/
  evaluation.py              # stable score/value objects if they become domain vocabulary

application/
  evaluation.py              # compatibility exports during migration
  evaluation/
    synthetic.py             # model-call orchestration
    traces.py
    workspace_staged.py
    compare.py
    rubrics.py               # app-level pure-ish scoring if not domain-pure

infrastructure/
  evaluation/
    reports.py               # persisted report IO
    response_parsing.py      # JSON/markdown parsing if treated as adapter detail
    artifact_suite.py        # metadata-only artifact evaluation, if adapter-like
```

Do not create the full tree in one move unless it stays small and obvious.
Compatibility re-exports are fine while tests migrate.

## Implementation Slices

1. Introduce application-owned evaluation suite classes.
   - Start with synthetic behavior evaluation because it is the simplest.
   - Use the existing `ModelBehaviorEvaluationSuite` port shape initially.

2. Move prompt construction and model-call loops inward.
   - Keep `ModelProvider` as an application port.
   - Keep concrete providers in infrastructure.

3. Move or wrap scoring into pure functions/classes. **Completed.**
   - If scoring needs only `DatasetExample` and raw response, it can live in
     application or domain.
   - If scoring needs tool-contract catalogs, pass those contracts in rather
     than importing `infrastructure.tools.catalog` directly.
   - **Current status:** completed for synthetic, trace-derived, and
     workspace-staged behavior scoring while preserving compatibility imports.

4. Split report IO from evaluation behavior. **Completed for behavior
   evaluation: report readers/writers remain infrastructure adapters.**
   - Report readers/writers should remain infrastructure adapters.

5. Repeat for trace-derived and workspace-staged evaluation. **Completed.**

## Acceptance Criteria

- Model-call evaluation orchestration lives in application code.
- Infrastructure evaluation modules are adapters, report IO, parsing helpers, or
  compatibility re-exports.
- Rubric tests can run without model providers and without filesystem setup.
- CLI `eval` commands still delegate through application workflows.
- Evaluation result shapes and metrics remain compatible.
