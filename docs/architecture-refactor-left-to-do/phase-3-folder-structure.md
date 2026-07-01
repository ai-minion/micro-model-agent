# Phase 3: Reshape Folder Structure

## Goal

Make directory depth express both architecture layer and mid-level concern. The
top-level folders are correct, but `infrastructure/` and `application/` are
still flatter than the planned tree.

## Current Tree

Current top-level package shape:

```text
micro_model_agent/
  domain/
  application/
  agents/
  infrastructure/
  interfaces/
```

This part is good.

The interface split is also mostly good:

```text
interfaces/
  cli/
    app.py
    common.py
    commands/
  mcp/
    compat.py
    policy/
    prompts/
    tools/
```

The main mismatch is `infrastructure/`, where most modules still sit directly
under the package root.

## Target Direction

The original plan suggested this style:

```text
infrastructure/
  composition.py
  models/
    fake.py
    ollama.py
    transformers.py
  persistence/
    trace_store.py
    dataset_store.py
    training_artifacts.py
    comparison_trace.py
    workspace_registry.py
  repositories/
    local_index.py
    local_retrieval.py
    repository_metadata.py
    repository_paths.py
  tools/
    catalog.py
    command_runner.py
    contracts.py
    executor.py
    git_diff.py
    repo_read.py
    repo_search.py
    repo_semantic_search.py
    repo_write_files.py
    repo_write_patch.py
  training/
    local_finetuning.py
    ollama_packaging.py
  evaluation/
    reports.py
    response_parsing.py
    synthetic_behavior.py
    trace_behavior.py
    workspace_staged.py
  promotion/
    gate.py
```

The exact names can vary. The important thing is that a future contributor can
guess where a new adapter belongs before a file turns into a catch-all.

The tool package shape is now mostly complete: `tool_executor.py` moved to
`infrastructure/tools/executor.py`, and the old flat module remains as a
compatibility import.

The dataset helper shape is also grouped now: `dataset_curation.py`,
`dataset_metadata.py`, `dataset_prompting.py`, `dataset_validation.py`, and
`synthetic_data.py` moved under `infrastructure/datasets/`, with old flat
modules kept as compatibility imports.

Trace review/export helpers are grouped too: `trace_review.py` and
`trace_export.py` moved under `infrastructure/traces/`, with old flat modules
kept as compatibility imports.

## Suggested Moves

Use compatibility re-export modules to keep old imports working while internal
imports migrate.

### Models

Move:

- `fake_model_provider.py` -> `infrastructure/models/fake.py` **done**
- `ollama_model_provider.py` -> `infrastructure/models/ollama.py` **done**
- `transformers_model_provider.py` -> `infrastructure/models/transformers.py` **done**

Old modules remain as re-export shims during migration. Internal production
imports and direct provider tests should prefer the new package paths.

### Persistence

Move:

- `trace_store.py` -> `infrastructure/persistence/trace_store.py` **done**
- `comparison_trace.py` -> `infrastructure/persistence/comparison_trace.py` **done**
- `dataset_store.py` -> `infrastructure/persistence/dataset_store.py` **done**
- `training_records.py` -> `infrastructure/persistence/training_records.py` **done**
- `workspace_registry.py` -> `infrastructure/persistence/workspace_registry.py` **done**
- `training_artifacts.py` **not moved in this slice**

Some of these include more than persistence today. Move only cohesive pieces
when needed. `training_artifacts.py` currently also exposes fake training,
fine-tuning, evaluation-reader, and promotion helpers, so it remains a
follow-up candidate rather than being forced into `persistence/`.

### Repositories

Move:

- `local_index.py` -> `infrastructure/repositories/local_index.py` **done**
- `local_retrieval.py` -> `infrastructure/repositories/local_retrieval.py` **done**
- `repository_metadata.py` -> `infrastructure/repositories/metadata.py` **done**
- `repository_paths.py` -> `infrastructure/repositories/paths.py` **done**

Old modules remain as re-export shims during migration. Internal production
imports and direct repository tests should prefer the new package paths.

### Training And Packaging

Move:

- `local_finetuning.py` -> `infrastructure/training/local_finetuning.py` **done**
- `ollama_packaging.py` -> `infrastructure/training/ollama_packaging.py` **done**

Consider keeping `training_artifacts.py` as a compatibility shim after moving
store and record helpers. It remains a follow-up candidate because it still
mixes artifact storage, fake training, evaluation re-exports, and compatibility
imports.

### Evaluation

Move after Phase 2, not before:

- `evaluation_reports.py` -> `infrastructure/evaluation/reports.py` **done**
- `evaluation_response_parsing.py` ->
  `infrastructure/evaluation/response_parsing.py` **done**
- `evaluation_comparison.py` -> `infrastructure/evaluation/comparison.py`
  **done**
- `artifact_evaluation.py` -> `infrastructure/evaluation/artifact.py` **done**
- `synthetic_evaluation.py` -> `infrastructure/evaluation/synthetic_behavior.py`
  **done**
- `trace_evaluation.py` -> `infrastructure/evaluation/trace_behavior.py`
  **done**
- `workspace_staged_evaluation.py` ->
  `infrastructure/evaluation/workspace_staged.py` **done**
- `workspace_staged_review.py` ->
  `infrastructure/evaluation/workspace_staged_review.py` **done**
- `synthetic_rubrics.py` -> `infrastructure/evaluation/synthetic_rubric.py`
  **done**
- `trace_rubrics.py` -> `infrastructure/evaluation/trace_rubric.py` **done**
- `workspace_staged_rubrics.py` ->
  `infrastructure/evaluation/workspace_staged_rubric.py` **done**

Do not move model-call orchestration deeper into infrastructure if Phase 2 is
going to move it to application.

### Promotion

Move:

- `promotion_gate.py` -> `infrastructure/promotion/gate.py` **done**

If promotion policy becomes pure, move that policy to application or domain
instead of burying it in infrastructure.

## Application Folder Shape

The application layer is also flat today:

```text
application/
  datasets.py
  evaluation.py
  promotion.py
  tool_loop.py
  training.py
  workflows.py
```

This is acceptable while files are moderate, but the target shape is:

```text
application/
  tool_loop/
  datasets/
  training/
  evaluation/
  promotion/
```

Do this only after ownership is correct. Moving flat files into subpackages
before Phase 1 and Phase 2 may just preserve the current misplaced logic in
prettier folders.

## Acceptance Criteria

- Old import paths continue to work through re-export shims or are migrated in
  the same slice.
- `infrastructure/` no longer reads as a flat list of unrelated adapters.
- Heavy optional dependencies remain isolated from basic CLI/MCP usage.
- Tests and docs import the new paths where practical.
- Full ruff and pytest pass after each small move.
