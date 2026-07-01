# Phase 1: Finish Run-Loop Application Ownership

## Status

Completed.

The run-loop policy and preparation work now lives in
`src/micro_model_agent/application/tool_loop.py`, including run-profile budget
application, tool-name normalization, tool selection, schema selection, and
metadata assembly. Concrete model/tool/trace wiring is shared through
`src/micro_model_agent/infrastructure/composition.py`.

`src/micro_model_agent/interfaces/cli/commands/loop.py` and
`src/micro_model_agent/interfaces/mcp/tools/run_loop.py` now translate CLI/MCP
arguments and format responses while delegating the shared loop assembly. MCP
legacy aliases and patch-application safety remain at the MCP edge.

Verification at completion:

```text
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m ruff check src docs'
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m pytest'
```

Both passed; the full test suite reported 258 passing tests after the follow-up
trace-evaluation slice.

## Goal

Make the model-driven tool loop a real application use case. Today
`RunToolLoopWorkflow` exists, but most run-loop orchestration still happens in
the CLI and MCP adapters.

## Current Mismatch

`src/micro_model_agent/application/tool_loop.py` owns request/result dataclasses
and a `RunToolLoopWorkflow`, but the workflow mostly translates a request into a
`ToolLoopAgentTask` and calls the injected runner.

The adapter still owns too much:

- `src/micro_model_agent/interfaces/mcp/tools/run_loop.py`
  - applies `run_profile` settings
  - normalizes allowed and required tools
  - resolves model settings
  - constructs the model provider
  - constructs the built-in executor
  - wraps the executor with MCP patch policy
  - constructs trace stores
  - chooses tool schemas
  - appends comparison trace events
  - builds the MCP response dictionary
- `src/micro_model_agent/interfaces/cli/commands/loop.py`
  - imports `ToolLoopAgent` directly
  - constructs the model provider, executor, trace store, and workflow
  - selects tool schemas
  - performs loop-specific model option handling

The MCP dictionary response and CLI printing belong in adapters, but the
workflow policy and orchestration should move inward.

## Suggested Shape

Keep application contracts protocol-neutral. One possible direction:

- `application/tool_loop.py` remains the compatibility module for now.
- Add smaller request/config dataclasses if needed:
  - `RunToolLoopRequest`
  - `ToolLoopRuntime`
  - `ToolLoopPolicy`
  - `ToolLoopTraceHooks`
- `RunToolLoopWorkflow` should own:
  - run-profile budget application, once represented in protocol-neutral terms
  - allowed/required tool policy inputs
  - schema selection decision
  - run metadata assembly that is not transport-specific
  - comparison/event hooks through an application port, if retained
- Infrastructure should provide:
  - model provider factories
  - tool executor factories
  - trace stores
- Interfaces should translate:
  - Typer/FastMCP arguments into application requests
  - application results into CLI text or MCP dictionaries

## Implementation Slices

1. Add application-level tests with fake runner/executor/model/trace hooks.
   - The tests should not import Typer, FastMCP, `interfaces`, or concrete
     model providers.

2. Move run-profile budget application out of MCP policy.
   - Either make run profiles an application concept or have MCP translate the
     profile into explicit budget fields before calling application.

3. Move allowed/required tool normalization into application or a pure policy
   module.
   - Keep MCP legacy alias handling at the MCP edge.
   - Keep repository-specific availability checks outside domain.

4. Move common CLI/MCP loop assembly behind a factory or application-facing
   runner builder.
   - The interface may still choose concrete infrastructure for now, but it
     should not hand-roll the loop use case.

5. Keep response formatting at the edge.
   - MCP returns JSON-compatible dictionaries.
   - CLI prints response, trace id, and tool-call count.

## Acceptance Criteria

- `interfaces/mcp/tools/run_loop.py` no longer owns run-loop policy.
- `interfaces/cli/commands/loop.py` no longer imports `ToolLoopAgent`
  directly unless it is clearly limited to composition.
- Application tests can exercise the run-loop workflow with fakes and no MCP or
  Typer imports.
- MCP and CLI behavior remains compatible.
- Existing public MCP tool names and CLI options remain stable.
