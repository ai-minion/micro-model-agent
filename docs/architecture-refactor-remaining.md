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

## Remaining CLI Orchestration

Move these command paths into application workflows in small, behavior-preserving
slices. Keep concrete file formats, model providers, local process execution,
and external integrations in infrastructure. Keep Typer option parsing, output
formatting, and exit behavior in CLI adapters.

- `eval compare`
  - Load persisted evaluation reports.
  - Compare score/metric deltas.
  - Write comparison report.
- Remaining eval command orchestration, as needed:
  - `eval synthetic`
  - `eval traces`
  - `eval workspace-staged`
  - `eval review-workspace-staged`

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

4. Enforce boundaries and contributor guidance.
   - Keep architecture tests strict.
   - Update `docs/architecture.md` after each completed structural move.
   - Add contribution guidance once the new shape is stable.

## Acceptance For Each Slice

- Public CLI commands and groups stay unchanged.
- Console script import remains `micro_model_agent.interfaces.cli:app`.
- New application tests use fakes and do not import Typer, concrete providers,
  FastMCP, or external process execution.
- Existing characterization tests keep passing.
- `ruff check src docs` and `pytest` pass.
