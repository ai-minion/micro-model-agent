# Architecture Refactor Remaining Work

This is the leftover-work checklist for the architecture refactor. Use
`docs/architecture-refactor-followup.md` for the current checkpoint and
verification details.

## Completed Inward Moves

- Full `promote` CLI group:
  - `promote gate`
  - `promote record`
  - `promote list`
  - `promote select`
  - `promote package-ollama`
- Dataset commands:
  - `dataset synthesize`
  - `dataset validate`
  - `dataset export`
  - `dataset merge`
  - `dataset relabel`
  - `dataset export-traces`
  - `dataset review-trace`
- Training commands:
  - `train synthetic`
- Eval commands:
  - `eval compare`
  - `eval synthetic`
  - `eval traces`
  - `eval workspace-staged`
  - `eval review-workspace-staged`

## Remaining CLI Orchestration

All tracked CLI orchestration slices have been moved behind application
workflows. Keep concrete file formats, model providers, local process execution,
and external integrations in infrastructure. Keep Typer option parsing, output
formatting, interactive prompts, and exit behavior in CLI adapters.

## Completed Larger Cleanup

- Enforce boundaries and contributor guidance:
  - Architecture boundary tests are strict for `domain` and `application`.
  - `docs/architecture.md` reflects the completed CLI/application workflow
    moves.
  - `CONTRIBUTING.md` documents contributor-facing boundary guidance.
- Split large infrastructure modules:
  - Evaluation report loading/writing was extracted from `training_artifacts.py`
    into `infrastructure/evaluation_reports.py`, with compatibility re-exports
    left in place.
  - Metadata-only synthetic artifact evaluation was extracted from
    `training_artifacts.py` into `infrastructure/artifact_evaluation.py`, with a
    compatibility re-export left in place.
  - Training and artifact JSON record helpers were extracted from
    `training_artifacts.py` into `infrastructure/training_records.py`, with
    compatibility re-exports left in place.
  - Promotion gate storage, registry serialization, and the minimum-score policy
    were extracted from `training_artifacts.py` into
    `infrastructure/promotion_gate.py`, with compatibility re-exports left in
    place.
  - Local fine-tuning backends, runner logic, and training helper functions were
    extracted from `training_artifacts.py` into
    `infrastructure/local_finetuning.py`, with compatibility re-exports left in
    place.
  - Trace-derived behavior evaluation was extracted from
    `synthetic_evaluation.py` into `infrastructure/trace_evaluation.py`, and
    shared JSON response parsing was extracted into
    `infrastructure/evaluation_response_parsing.py`.
  - Staged workspace review queue building/writing was extracted from
    `workspace_staged_evaluation.py` into
    `infrastructure/workspace_staged_review.py`, with compatibility re-exports
    left in place.

## Larger Remaining Slices

1. Split large infrastructure modules.
   - Break up `training_artifacts.py`.
   - Break up `synthetic_evaluation.py`.
   - Break up `workspace_staged_evaluation.py`.
   - Preserve old import paths with temporary re-export modules if needed.

2. Extract pure evaluation rubrics.
   - Make scoring testable without model providers or filesystem setup.
   - Keep report serialization in infrastructure.

3. Slim `ToolLoopAgent`.
   - Extract prompt rendering.
   - Extract history compaction.
   - Extract decision parsing.
   - Extract portable workflow policy where it is not model-loop mechanics.

## Acceptance For Each Slice

- Public CLI commands and groups stay unchanged.
- Console script import remains `micro_model_agent.interfaces.cli:app`.
- New application tests use fakes and do not import Typer, concrete providers,
  FastMCP, or external process execution.
- Existing characterization tests keep passing.
- `ruff check src docs` and `pytest` pass.
