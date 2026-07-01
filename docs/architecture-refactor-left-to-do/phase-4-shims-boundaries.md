# Phase 4: Retire Shims And Strengthen Boundaries

## Goal

Prevent the architecture from drifting back after the remaining moves. The
current boundary tests protect `domain` and `application`, but they do not yet
enforce thin adapters, folder shape, or private shim retirement.

## Current State

The strict tests are in:

- `src/micro_model_agent/test_architecture_boundaries.py`

They currently check:

- production `application` modules do not import `agents`, `infrastructure`,
  `interfaces`, Typer, MCP, or Pydantic
- production `domain` modules do not import project modules or heavy adapter
  dependencies
- pure rubric modules do not import concrete infrastructure adapters
- internal production imports use grouped infrastructure packages instead of
  old flat compatibility modules
- `interfaces.cli` and `interfaces.mcp_server` do not re-export private helper
  names
- production `interfaces` modules do not import concrete `agents`

Compatibility shims still exist:

- `src/micro_model_agent/interfaces/mcp_server.py`
  - public compatibility entrypoint
  - no longer re-exports private helpers such as `_run_profile_settings`
- `src/micro_model_agent/interfaces/cli/__init__.py`
  - re-exports `app`
  - no longer re-exports private helpers such as `_resolve_loop_model_options`

Tests that covered private helper behavior now import the owner modules
directly:

- `src/micro_model_agent/interfaces/test_mcp_server.py`
- `src/micro_model_agent/interfaces/test_cli.py`

## Shim Policy

Keep public compatibility imports:

- `micro_model_agent.interfaces.mcp_server:create_mcp_server`
- `micro_model_agent.interfaces.mcp_server:run_agent_loop`
- `micro_model_agent.interfaces.cli:app`

Retire or explicitly bless private compatibility imports:

- `_run_profile_settings`
- `_allowed_tool_names`
- `_required_tool_names`
- `_resolve_model_settings`
- `_model_provider`
- `_tool_prompt_schemas`
- `_workflow_trace_store`
- `_resolve_loop_model_options`
- `_load_dotenv`

Done for the compatibility modules above. If another helper is later made
genuinely public, rename it without a leading underscore and document it.
Otherwise keep tests on the real module that owns the behavior.

## Boundary Tests To Add

Add these only after the relevant phase is complete, so the tests describe the
intended architecture rather than the transition state.

### Thin Interface Checks

Possible checks:

- `interfaces/cli/commands/*.py` should not import `agents`. **Done via a
  package-wide `interfaces` boundary test.**
- `interfaces/mcp/tools/*.py` should not import `agents`. **Done via the same
  package-wide `interfaces` boundary test.**
- interface command modules should avoid direct low-level provider imports,
  except composition modules explicitly allowed by name.

### Infrastructure Shape Checks

Possible checks:

- new concrete model providers live under `infrastructure.models`.
- new persistence adapters live under `infrastructure.persistence`.
- no new top-level `infrastructure/*_evaluation.py` modules once evaluation is
  moved.

Avoid brittle tests that fail every time a module is renamed. Prefer simple
rules that catch architectural direction mistakes.

### Pure Policy Checks

Possible checks:

- `domain` stays free of project imports.
- pure rubric modules do not import model providers, filesystem stores, Typer,
  FastMCP, or concrete infrastructure adapters.

## Documentation Updates

After each phase:

- update `docs/architecture.md` to describe the actual current tree
- update this left-to-do set if the phase is partially completed
- avoid relying on stale "Current Shape" sections in
  `docs/architecture-refactor-plan.md`

## Acceptance Criteria

- Tests no longer depend on private compatibility exports.
- Public compatibility shims remain only where they serve external callers.
- Architecture tests catch the most important future drift:
  - application importing outward
  - domain importing framework/adapters
  - interfaces importing concrete agents
  - private helper exports returning to public compatibility shims
  - new infrastructure modules bypassing the agreed package shape
- Full ruff and pytest pass.
