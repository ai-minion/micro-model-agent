# Architecture Refactor Follow-Up

This is the short continuation handoff for finishing the architecture refactor.
Use it instead of carrying the full assessment in
`docs/architecture-refactor-plan.md` unless deeper historical context is needed.

## Current State

Completed through the CLI package conversion, common-helper extraction, and
command-module split:

- Architecture boundary tests exist for `domain` and `application`.
- CLI and MCP public surface characterization tests exist.
- The production `application -> agents` dependency has been removed.
- `interfaces/mcp_server.py` is now a compatibility shim over
  `interfaces/mcp/*`.
- Shared runtime assembly lives in `infrastructure/composition.py`.
- MCP runtime assembly uses composition helpers.
- CLI `loop` uses composition helpers for model, trace, tool executor, and
  model option resolution.
- CLI eval commands use composition helpers for provider selection:
  `eval synthetic`, `eval traces`, and `eval workspace-staged`.
- `application/tool_loop.py` owns the protocol-neutral tool-loop request/result
  contracts and `RunToolLoopWorkflow`.
- `ToolLoopAgent` consumes the application-owned task/result contract and
  remains available through the old `agents.tool_loop_agent` import path.
- MCP `run_agent_loop` and CLI `loop` reuse `RunToolLoopWorkflow`; MCP-specific
  dictionaries and CLI-specific printing/exit behavior stay in their adapters.
- Application run-loop tests use a fake runner and do not import FastMCP, Typer,
  or concrete model providers.
- `interfaces/cli.py` has been replaced by the `interfaces/cli/` package.
- `interfaces/cli/app.py` owns Typer app and command-group construction, then
  registers command modules.
- `interfaces/cli/common.py` owns shared CLI helpers such as `_run`, `_fail`,
  `.env` loading, scripted-response loading, and model option selection.
- `interfaces/cli/__init__.py` re-exports `app`, `_load_dotenv`, and
  `_resolve_loop_model_options` for compatibility with the console script and
  existing tests.
- `interfaces/cli/commands/repo.py` owns root `init`, `index`, and `task`
  command handlers.
- `interfaces/cli/commands/loop.py` owns the root `loop` command handler.
- `interfaces/cli/commands/mcp.py` owns the root `serve-mcp` command handler.
- `interfaces/cli/commands/dataset.py` owns the `dataset` command group.
- `interfaces/cli/commands/train.py` owns the `train synthetic` command.
- `interfaces/cli/commands/eval.py` owns the `eval` command group.
- `interfaces/cli/commands/promote.py` owns the `promote` command group.

Last known verification:

```text
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m ruff check src docs'
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m pytest'
```

Result:

```text
209 passed
```

## Compatibility Invariants

Keep these stable unless a separate migration is explicitly requested:

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
- Trace JSON shapes and evaluation semantics.

## Next Slice: Move CLI Orchestration Inward

Goal: add application services for dataset, training, evaluation, and promotion
workflows that currently coordinate multiple infrastructure adapters directly
inside CLI command handlers.

Suggested implementation:

1. Start with one narrow command path that currently performs orchestration in
   CLI code, such as promotion gating or dataset validation.
2. Introduce an application request/result service that depends on existing
   ports or small new ports.
3. Keep file formats, concrete providers, and external process integration in
   infrastructure.
4. Leave the CLI responsible for Typer options, printing, and exit behavior.

Acceptance:

- Console script import remains `micro_model_agent.interfaces.cli:app`.
- CLI root commands remain `init`, `index`, `task`, `loop`, and `serve-mcp`.
- CLI groups remain `dataset`, `train`, `eval`, and `promote`.
- Existing CLI characterization tests pass after each small move.
- New application service tests exercise orchestration without Typer, concrete
  model providers, or external process execution.
- `ruff check src docs` and `pytest` pass.

## Remaining Slices

Work in small behavior-preserving steps. Keep shims until tests and known
callers have moved.

1. Move dataset, training, evaluation, and promotion orchestration inward.
   - Add application services where CLI currently coordinates multiple
     adapters or policies.
   - Keep file formats and external integrations in infrastructure.

2. Split large infrastructure modules.
   - Break up `training_artifacts.py`.
   - Break up `synthetic_evaluation.py`.
   - Break up `workspace_staged_evaluation.py`.
   - Preserve old import paths with temporary re-export modules if needed.

3. Extract pure evaluation rubrics.
   - Make scoring testable without model providers or filesystem setup.
   - Keep report serialization in infrastructure.

4. Slim `ToolLoopAgent`.
   - Extract prompt rendering.
   - Extract history compaction.
   - Extract decision parsing.
   - Extract portable workflow policy where it is not model-loop mechanics.

5. Enforce boundaries.
   - Keep architecture tests strict.
   - Update `docs/architecture.md` after each completed structural move.
   - Add contribution guidance once the new shape is stable.

## Final Done Definition

- `domain` and `application` import only allowed layers.
- CLI and MCP modules are thin adapters with compatibility shims.
- Use-case orchestration lives in `application`.
- Large modules are split or explicitly documented.
- Public CLI commands and MCP tool/prompt names are unchanged.
- Full ruff and pytest are green.
