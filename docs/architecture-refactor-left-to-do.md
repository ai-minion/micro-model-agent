# Architecture Refactor Left To Do

This is the short handoff for continuing the architecture refactor from the
current tree. The older assessment in `docs/architecture-refactor-plan.md`
contains the full history, but parts of its "current shape" are now stale. Use
this document as the starting point for a new chat.

## Current State

The top-level package names now match the intended Clean/Hexagonal shape:

```text
domain/
application/
agents/
infrastructure/
interfaces/
```

The largest visible moves are complete:

- `interfaces/cli.py` has been replaced by the `interfaces/cli/` package.
- `interfaces/mcp_server.py` is a compatibility shim over `interfaces/mcp/*`.
- `application` and `domain` boundary tests are strict for inward imports.
- Dataset, training, promotion, and evaluation CLI commands mostly delegate to
  application workflows.
- Dataset, evaluation, training, promotion, and root repo CLI commands now use
  shared composition factories for concrete adapter wiring.
- MCP workspace registration and optional init-tool exposure now use shared
  composition helpers for repository metadata and workspace registry access.
- Large agent and infrastructure modules have already been split substantially.
- Run-loop budget/tool policy now lives in `application/tool_loop.py`, with
  shared concrete loop assembly in `infrastructure/composition.py`.
- Synthetic, trace-derived, and workspace-staged behavior evaluation model-call
  loops now live in `application/evaluation.py`; their infrastructure modules
  are compatibility adapters around scoring/rubric wiring.

The remaining mismatch is less about file count and more about ownership:

- Some use-case orchestration still lives in adapters or infrastructure.
- Most `infrastructure/` adapters are grouped by mechanism, with old flat
  compatibility modules still present for external callers.
- Synthetic, trace, and workspace-staged scoring/rubric code is now separated
  from concrete infrastructure helpers, though broader module-shape cleanup
  remains.
- Private helper compatibility exports have been retired from
  `interfaces.cli` and `interfaces.mcp_server`; tests import owner modules
  directly.
- Boundary tests now enforce private shim retirement, grouped infrastructure
  imports, and that interface modules do not import concrete agents or
  low-level concrete model/tool adapters. They also guard top-level
  `infrastructure/` against new non-compatibility adapter modules and keep
  pure evaluation rubrics free of outward project/framework dependencies.

## Recommended Order

Work in small behavior-preserving slices. Keep public CLI commands, MCP tool
names, prompt names, and trace/evaluation schemas stable.

1. Finish run-loop application ownership. **Completed in this slice.**
   - Details: `docs/architecture-refactor-left-to-do/phase-1-run-loop.md`

2. Move evaluation model-call orchestration inward and make rubrics cleaner.
   **Mostly complete: behavior model-call loops moved and pure rubric modules
   no longer import concrete infrastructure helpers. Evaluation module/package
   shape remains a follow-up decision.**
   - Details: `docs/architecture-refactor-left-to-do/phase-2-evaluation.md`

3. Reshape folders, especially `infrastructure/`, behind compatibility imports.
   **Started: concrete model providers now live under
   `infrastructure/models/`, and repository-local adapters now live under
   `infrastructure/repositories/`; JSONL/filesystem persistence helpers now
   live under `infrastructure/persistence/`; training/packaging adapters now
   live under `infrastructure/training/`; promotion gate adapters now live
   under `infrastructure/promotion/`; evaluation report/comparison/artifact
   helpers, behavior adapters, and rubrics now live under
   `infrastructure/evaluation/`; the built-in tool
   executor now lives under `infrastructure/tools/`; dataset generation,
   validation, curation, prompting, and metadata helpers now live under
   `infrastructure/datasets/`; trace review and trace dataset export helpers
   now live under `infrastructure/traces/`; training artifact store/fake-runner
   helpers now live under `infrastructure/training/artifacts.py`. Old flat
   modules are kept as compatibility imports.**
   - Details: `docs/architecture-refactor-left-to-do/phase-3-folder-structure.md`

4. Retire private compatibility shims and strengthen architecture tests.
   **Started: tests no longer import private helpers from
   `interfaces.cli` or `interfaces.mcp_server`; those compatibility modules no
   longer export underscore helpers; CLI task assembly no longer imports
   concrete agents directly; dataset, evaluation, training, promotion, and root
   repo CLI assembly and MCP workspace metadata/registry access now go through
   composition helpers; architecture tests guard those constraints and block
   direct interface imports of concrete model providers, tool executors,
   concrete dataset stores, workspace registries, repository metadata/index
   adapters, concrete training runners, promotion stores, and Ollama packagers;
   top-level infrastructure modules must now be composition or compatibility
   facades.**
   - Details: `docs/architecture-refactor-left-to-do/phase-4-shims-boundaries.md`

## Compatibility Invariants

Do not change these unless the task explicitly asks for a migration:

- Console script import: `micro_model_agent.interfaces.cli:app`.
- Compatibility import: `micro_model_agent.interfaces.mcp_server`.
- CLI root commands: `init`, `index`, `task`, `loop`, `serve-mcp`.
- CLI groups: `dataset`, `train`, `eval`, `promote`.
- Default MCP tools:
  - `micro_agent_init_workspace`
  - `micro_agent_run_loop`
  - `micro_agent_start_trace`
  - `micro_agent_record_trace_event`
  - `micro_agent_stop_trace`
  - `micro_agent_review_trace`
- Conditional MCP tools:
  - `micro_agent_init`
  - `micro_agent_builtin_tool`
  - `micro_agent_read_trace`
  - `micro_agent_list_builtin_tools`
- MCP prompts:
  - `compare_local_model_on_task`
  - `collect_real_trace`
  - `review_comparison_trace`
  - `smoke_test_micro_agent`
- Trace JSON, dataset JSONL, and evaluation report shapes.

## Current Hot Spots

These files are the most important starting points:

- `src/micro_model_agent/application/tool_loop.py`
- `src/micro_model_agent/application/evaluation.py`
- `src/micro_model_agent/infrastructure/workspace_staged_evaluation.py`
- `src/micro_model_agent/infrastructure/workspace_staged_rubrics.py`
- `src/micro_model_agent/infrastructure/synthetic_rubrics.py`
- `src/micro_model_agent/infrastructure/trace_evaluation.py`
- `src/micro_model_agent/infrastructure/models/`
- `src/micro_model_agent/infrastructure/repositories/`
- `src/micro_model_agent/infrastructure/persistence/`
- `src/micro_model_agent/infrastructure/training/`
- `src/micro_model_agent/infrastructure/promotion/`
- `src/micro_model_agent/infrastructure/evaluation/`
- `src/micro_model_agent/infrastructure/tools/`
- `src/micro_model_agent/infrastructure/datasets/`
- `src/micro_model_agent/infrastructure/traces/`
- `src/micro_model_agent/infrastructure/composition.py`
- `src/micro_model_agent/interfaces/mcp_server.py`
- `src/micro_model_agent/interfaces/cli/__init__.py`
- `src/micro_model_agent/test_architecture_boundaries.py`

## Verification

Use the same verification style as the previous refactor slices:

```text
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m ruff check src docs'
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m pytest'
```

If running directly in PowerShell with a Windows virtual environment, use the
equivalent project-local Python command.

## Definition Of Done

The remaining refactor is complete when:

- `domain` and `application` import only allowed inward layers.
- CLI and MCP modules are thin adapters, not use-case owners.
- Run-loop and evaluation use-case orchestration live in `application`.
- Pure policies/rubrics do not depend on infrastructure adapters.
- `infrastructure/` is grouped by adapter mechanism or clearly documented.
- Private compatibility shims are removed or explicitly documented as public.
- Architecture tests catch the important drift modes.
- Full ruff and pytest are green.
