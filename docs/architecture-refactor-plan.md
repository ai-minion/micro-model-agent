# Architecture Refactor Assessment and Plan

This document is a handoff plan for bringing `src/micro_model_agent` back toward
domain-driven, hexagonal/clean architecture without doing a risky rewrite.

For continuing the remaining implementation without carrying this full
assessment forward, use `docs/architecture-refactor-followup.md`.

## Summary

The package already has the right top-level names:

- `domain`
- `application`
- `infrastructure`
- `interfaces`
- `agents`

The problem is that the dependency boundaries and module responsibilities have
drifted. The most visible symptom is very large interface modules, especially
`interfaces/cli.py` and `interfaces/mcp_server.py`, but the deeper issue is that
several use cases live in adapters or infrastructure instead of the application
layer. There is also a reverse dependency from `application` to `agents`, which
means the application layer is not purely orchestrating ports and domain
contracts.

The refactor should be done as a sequence of small behavior-preserving moves:
first extract composition and command handlers, then move use-case orchestration
inward, then split infrastructure by adapter concern, and only then tighten
imports with tests.

## Progress Update - 2026-06-30

This section records the implementation work completed so far, so a new chat can
continue from the current tree without replaying the whole plan.

### Completed

- Added architecture boundary coverage in
  `src/micro_model_agent/test_architecture_boundaries.py`.
  - `application` production modules may not import `agents`, `infrastructure`,
    `interfaces`, Typer, MCP, or Pydantic.
  - `domain` production modules may not import project modules or framework /
    heavy adapter dependencies.
- Added CLI command-surface characterization coverage in
  `src/micro_model_agent/interfaces/test_cli.py`.
  - Root commands: `init`, `index`, `task`, `loop`, `serve-mcp`.
  - Groups covered: `dataset`, `train`, `eval`, `promote`.
- Tightened MCP prompt characterization in
  `src/micro_model_agent/interfaces/test_mcp_server.py`.
  - Default prompt names must remain exactly:
    `compare_local_model_on_task`, `collect_real_trace`,
    `review_comparison_trace`, `smoke_test_micro_agent`.
- Removed the production `application -> agents` dependency.
  - Moved `CodingAgentTask`, `CodingAgentResult`, and the
    `CodingWorkflowRunner` port into `application/ports.py`.
  - Updated `RunAgentWorkflow` to depend on `CodingWorkflowRunner`.
  - Kept compatibility by importing/re-exporting the task/result contracts from
    `agents/coding_agent.py`.
- Created the MCP package shell under `interfaces/mcp/`.
- Extracted MCP compatibility constants and type aliases to
  `interfaces/mcp/compat.py`.
- Extracted MCP policy helpers:
  - `PatchPolicyToolExecutor` to `interfaces/mcp/policy/patch_policy.py`.
  - tool-name normalization, allowed test commands, git-diff filtering, and
    run-profile budgets to `interfaces/mcp/policy/tool_names.py`.
  - debug/init exposure decisions to `interfaces/mcp/policy/exposure.py`.
- Extracted MCP prompt registration and prompt text to
  `interfaces/mcp/prompts/registry.py`.
- Extracted MCP workspace helpers to `interfaces/mcp/workspace.py`.
  - `init_workspace`
  - `init_repository`
  - workspace-id resolution
  - user path parsing / Windows-to-WSL path mapping
- Extracted MCP trace helpers to `interfaces/mcp/traces.py`.
  - `read_trace`
  - comparison trace start/record/stop/review helpers
  - workflow/comparison trace store factories
- Extracted MCP debug built-in tool bridge to
  `interfaces/mcp/tools/builtin.py`.
  - `call_builtin_tool`
  - `list_builtin_tools`
- Extracted FastMCP server construction to `interfaces/mcp/server.py`.
  - `create_mcp_server`
  - `serve`
  - tool-list changed capability patching
- Extracted MCP tool registration wrappers to
  `interfaces/mcp/tools/registry.py`.
  - default public tool wrappers
  - conditional init tool registration
  - conditional debug tool registration
- Extracted the MCP run-loop helper and model-setting helpers to
  `interfaces/mcp/tools/run_loop.py`.
  - `run_agent_loop`
  - model provider caching / scripted-provider selection
  - model config resolution
  - tool prompt schema enrichment
- Added shared runtime composition helpers in
  `infrastructure/composition.py`.
  - model option resolution across explicit args, environment, and repository
    config
  - model provider construction for scripted, Transformers/PEFT, and Ollama
  - workflow/comparison trace store factories
  - workspace registry factory
  - built-in tool executor factory
  - allowed test command construction
  - tool schema enrichment
- Added composition coverage in
  `src/micro_model_agent/infrastructure/test_composition.py`.
- Wired MCP run-loop, debug built-in tool execution, trace stores, workspace
  registry, and allowed test command construction through shared composition.
- Wired the CLI `loop` command's model provider, trace store, tool executor,
  and model option resolution through shared composition while preserving the
  private `_resolve_loop_model_options` compatibility helper.
- Wired the CLI evaluation commands through shared composition for model
  provider selection.
  - `eval synthetic`
  - `eval traces`
  - `eval workspace-staged`
  - Scripted, explicit adapter/base, Ollama, runnable training artifact, and
    dry-run metadata fallback selection are now covered in
    `infrastructure/test_composition.py`.
- Kept `interfaces/mcp_server.py` as the compatibility entrypoint for existing
  imports. It still exports the moved public helpers by importing them from the
  new modules.
- Updated `docs/architecture.md` with the new application-owned coding workflow
  port and a link back to this refactor plan.

### Current Verification

Last known green commands:

```text
.venv/bin/python -m ruff check src docs
.venv/bin/python -m pytest
```

Result:

```text
203 passed
```

The commands were run through WSL from the repository root because the checked-in
virtual environment has a POSIX-style layout:

```text
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m pytest'
```

### Current Shape After Completed Work

- `interfaces/mcp_server.py` has been reduced from approximately 1,203 lines to
  116 lines.
- The new MCP modules are:
  - `interfaces/mcp/compat.py`
  - `interfaces/mcp/server.py`
  - `interfaces/mcp/workspace.py`
  - `interfaces/mcp/traces.py`
  - `interfaces/mcp/policy/exposure.py`
  - `interfaces/mcp/policy/patch_policy.py`
  - `interfaces/mcp/policy/tool_names.py`
  - `interfaces/mcp/prompts/registry.py`
  - `interfaces/mcp/tools/builtin.py`
  - `interfaces/mcp/tools/registry.py`
  - `interfaces/mcp/tools/run_loop.py`
- Shared runtime assembly now lives in `infrastructure/composition.py` and is
  used by the MCP helpers and the CLI `loop` command.
- The most important dependency-direction issue from the original assessment is
  fixed: `application/workflows.py` no longer imports `agents.coding_agent`.

### Latest Slice Details

The latest completed slice finished the remaining CLI evaluation part of Phase 2
runtime composition:

- Added `EvaluationModelSelection` and `select_evaluation_model` to
  `src/micro_model_agent/infrastructure/composition.py`.
- Added tests for evaluation provider selection:
  - scripted responses
  - explicit PEFT adapter with base model read from `adapter_config.json`
  - Ollama model selection with environment-provided base URL
  - runnable training artifact selection
  - dry-run training artifact metadata fallback
- Updated the CLI eval commands to use the shared selector:
  - `eval synthetic`
  - `eval traces`
  - `eval workspace-staged`
- Preserved the synthetic dry-run metadata fallback for artifacts without local
  adapter weights.
- Preserved the stricter trace/workspace behavior that requires a runnable
  model, adapter, or scripted response.
- Removed direct low-level evaluation provider imports from the CLI eval
  handlers. The remaining CLI provider import is the unrelated
  `StaticModelProvider` used by the root `task` smoke command.

The broader Phase 2 work also created
`src/micro_model_agent/infrastructure/composition.py` and moved shared runtime
assembly behind composition helpers:
  - `resolve_model_options`
  - `build_model_provider`
  - `base_model_from_adapter`
  - `select_evaluation_model`
  - `allowed_test_commands`
  - `build_builtin_tool_executor`
  - `workflow_trace_store`
  - `comparison_trace_store`
  - `workspace_registry`
  - `tool_prompt_schemas`
- Updated MCP modules to call composition helpers:
  - `interfaces/mcp/tools/run_loop.py` now uses shared model provider,
    executor, model option, and schema helpers.
  - `interfaces/mcp/tools/builtin.py` now uses the shared built-in executor
    factory.
  - `interfaces/mcp/traces.py` now delegates workflow/comparison trace store
    creation and trace directory construction to composition.
  - `interfaces/mcp/workspace.py` now delegates workspace registry creation to
    composition.
  - `interfaces/mcp/policy/tool_names.py` now delegates allowed test command
    construction to composition, while retaining MCP-specific tool filtering and
    compatibility aliases.
- Updated the CLI `loop` command in `interfaces/cli.py` to use composition for
  model provider construction, trace store creation, tool executor creation,
  and model option resolution.
- Preserved `_resolve_loop_model_options` in `interfaces/cli.py` as a
  compatibility helper returning the same dictionary shape used by tests.
- Preserved `interfaces/mcp_server.py` compatibility re-exports for moved MCP
  helpers.

### Compatibility Notes

- Keep `micro_model_agent.interfaces.mcp_server` working until callers and tests
  migrate to the new `interfaces.mcp.*` modules.
- Known private compatibility wrappers still intentionally exist in
  `mcp_server.py`, including `_path_from_user_input`, `_allowed_test_commands`,
  `_allowed_tool_names`, `_required_tool_names`, `_run_profile_settings`, and
  `_workflow_trace_store`. Model-setting helpers are also re-exported for
  compatibility as `_resolve_model_settings`, `_model_provider`,
  `_base_model_from_adapter`, `_string_config_value`, and
  `_tool_prompt_schemas`.
- Public MCP tool and prompt names are unchanged.
- Public CLI command names are unchanged.
- `docs/architecture-refactor-plan.md` was untracked when implementation began;
  treat the document and the implementation changes as part of the same pending
  refactor worktree state.

### Recommended Next Slice

Begin Phase 3: move the run-loop use case into the application layer.

1. Add an application service such as `RunToolLoopWorkflow` in
   `application/tool_loop.py`.
2. Define protocol-neutral request/result dataclasses for the run loop instead
   of returning MCP-shaped dictionaries from the core orchestration.
3. Move orchestration currently behind `interfaces/mcp/tools/run_loop.py` into
   the application service:
   - run-profile setting application
   - allowed/required tool policy inputs
   - tool schema selection
   - trace/comparison event recording hooks
4. Keep MCP response dictionaries in the MCP adapter.
5. Let CLI `loop` use the same application service where practical.
6. Re-run:

   ```text
   .venv/bin/python -m ruff check src docs
   .venv/bin/python -m pytest
   ```

Suggested acceptance for the next chat:

- `interfaces/mcp_server.py` remains a compatibility entrypoint.
- MCP tool and prompt names remain unchanged.
- Runtime assembly logic is shared outside the interface layer where practical.
- CLI eval commands continue to avoid direct low-level model provider imports.
- Application run-loop orchestration can be tested with fake model/tool/trace
  dependencies and no FastMCP imports.
- Full tests and ruff remain green.

## Current Shape

Approximate Python file and line counts from `src/micro_model_agent`:

| Package | Files | Lines | Notes |
| --- | ---: | ---: | --- |
| `domain` | 6 | 409 | Small, mostly framework-independent dataclasses and enums. |
| `application` | 4 | 359 | Thin, but currently imports `agents.coding_agent`. |
| `agents` | 5 | 2,526 | Contains model/tool loop orchestration and tests. |
| `infrastructure` | 58 | 8,875 | Many adapters plus several application-like workflows. |
| `interfaces` | 5 | 4,782 | Very large CLI and MCP modules with assembly and use-case logic. |

Largest production modules:

| Module | Lines | Primary Concern Today |
| --- | ---: | --- |
| `interfaces/cli.py` | 1,887 | Typer declarations, command handlers, output formatting, model config, dataset/training/eval orchestration. |
| `interfaces/mcp_server.py` | 1,203 | FastMCP registration, run-loop orchestration, workspace handling, model provider assembly, MCP-specific tool policy. |
| `agents/tool_loop_agent.py` | 1,165 | Model decision loop, prompt construction, orchestration policies, history compaction, trace finalization. |
| `infrastructure/workspace_staged_evaluation.py` | 786 | Evaluation suite plus rubric scoring details and review-record building. |
| `infrastructure/training_artifacts.py` | 661 | Artifact store, fake runner, Hugging Face backend, local runner, eval IO, promotion registry/policy. |
| `infrastructure/synthetic_evaluation.py` | 544 | Synthetic and trace behavior evaluation suites plus scoring rubrics. |

## Current Public Surface To Preserve

The refactor should keep the public command and MCP surface stable unless a
separate deprecation/migration task is created.

### CLI Commands

Current Typer commands and command groups in `interfaces/cli.py`:

| Group | Command | Current Implementation Location | Refactor Destination |
| --- | --- | --- | --- |
| root | `init` | `interfaces/cli.py::init` | `interfaces/cli/commands/repo.py` |
| root | `index` | `interfaces/cli.py::index` | `interfaces/cli/commands/repo.py` |
| root | `task` | `interfaces/cli.py::task` | `interfaces/cli/commands/task.py` or `repo.py` |
| root | `loop` | `interfaces/cli.py::loop` | `interfaces/cli/commands/loop.py` |
| root | `serve-mcp` | `interfaces/cli.py::serve_mcp` | `interfaces/cli/commands/mcp.py` |
| `dataset` | `synthesize` | `interfaces/cli.py::synthesize` | `interfaces/cli/commands/dataset.py` |
| `dataset` | `validate` | `interfaces/cli.py::validate` | `interfaces/cli/commands/dataset.py` |
| `dataset` | `export` | `interfaces/cli.py::export_dataset` | `interfaces/cli/commands/dataset.py` |
| `dataset` | `export-traces` | `interfaces/cli.py::export_traces` | `interfaces/cli/commands/dataset.py` |
| `dataset` | `review-trace` | `interfaces/cli.py::review_trace` | `interfaces/cli/commands/dataset.py` |
| `dataset` | `relabel` | `interfaces/cli.py::relabel_dataset` | `interfaces/cli/commands/dataset.py` |
| `dataset` | `merge` | `interfaces/cli.py::merge_dataset` | `interfaces/cli/commands/dataset.py` |
| `train` | `synthetic` | `interfaces/cli.py::train_synthetic` | `interfaces/cli/commands/train.py` |
| `eval` | `synthetic` | `interfaces/cli.py::eval_synthetic` | `interfaces/cli/commands/eval.py` |
| `eval` | `traces` | `interfaces/cli.py::eval_traces` | `interfaces/cli/commands/eval.py` |
| `eval` | `workspace-staged` | `interfaces/cli.py::eval_workspace_staged` | `interfaces/cli/commands/eval.py` |
| `eval` | `review-workspace-staged` | `interfaces/cli.py::review_workspace_staged` | `interfaces/cli/commands/eval.py` |
| `eval` | `compare` | `interfaces/cli.py::eval_compare` | `interfaces/cli/commands/eval.py` |
| `promote` | `gate` | `interfaces/cli.py::promote_gate` | `interfaces/cli/commands/promote.py` |
| `promote` | `record` | `interfaces/cli.py::promote_record` | `interfaces/cli/commands/promote.py` |
| `promote` | `list` | `interfaces/cli.py::promote_list` | `interfaces/cli/commands/promote.py` |
| `promote` | `select` | `interfaces/cli.py::promote_select` | `interfaces/cli/commands/promote.py` |
| `promote` | `package-ollama` | `interfaces/cli.py::promote_package_ollama` | `interfaces/cli/commands/promote.py` |

Compatibility note: `pyproject.toml` currently points the console script at
`micro_model_agent.interfaces.cli:app`. Keep that import working until the
script entrypoint is updated and released.

### MCP Tools And Prompts

Current default MCP tools registered by `create_mcp_server`:

- `micro_agent_init_workspace`
- `micro_agent_run_loop`
- `micro_agent_start_trace`
- `micro_agent_record_trace_event`
- `micro_agent_stop_trace`
- `micro_agent_review_trace`

Conditionally registered tools:

- `micro_agent_init`, controlled by repository initialization and exposure flags
- `micro_agent_builtin_tool`, debug-only
- `micro_agent_read_trace`, debug-only
- `micro_agent_list_builtin_tools`, debug-only

Current MCP prompts:

- `compare_local_model_on_task`
- `collect_real_trace`
- `review_comparison_trace`
- `smoke_test_micro_agent`

The next implementation should add characterization tests that assert these
names remain registered under the same default/debug/init conditions.

## Detailed Responsibility Inventory

### Domain

Current role:

- Value objects and enums for workflow traces, tools, retrieval, datasets, and
  training artifacts.
- Mostly clean: no infrastructure, interface framework, filesystem adapter, or
  model-provider imports were found in production domain modules.

Recommended growth:

- Add pure concepts only when they are stable business vocabulary.
- Candidate additions are evaluation score/rubric value objects and pure tool
  workflow policies.
- Avoid moving Pydantic request models into domain; those are adapter-facing
  validation contracts.

### Application

Current role:

- Port protocols in `application/ports.py`.
- `DefaultWorkflowEvaluator`, `TraceDatasetBuilder`, `RunAgentWorkflow`, and
  `label_from_workflow_result` in `application/workflows.py`.

Current drift:

- `application/workflows.py` imports concrete `agents.coding_agent` task,
  result, and runner classes.
- Important use cases live outside application:
  - model tool-loop orchestration lives in `interfaces/mcp_server.py`
  - dataset validation/export/relabel/merge orchestration mostly lives in CLI
    and infrastructure
  - training and promotion orchestration mostly lives in CLI and
    `infrastructure/training_artifacts.py`
  - evaluation suite orchestration mostly lives in infrastructure

Recommended role:

- Own use-case request/result objects.
- Own orchestration over ports.
- Remain free of Typer, FastMCP, concrete model providers, concrete stores,
  Hugging Face, and repository filesystem adapters.

### Agents

Current role:

- `CodingAgent` and `ToolLoopAgent` implement concrete model-driven workflows.
- Agents depend on application ports and domain contracts, which is generally
  acceptable if they are treated as outer application services.

Current drift:

- `ToolLoopAgent` owns enough workflow policy that it is hard to tell which
  rules are agent mechanics and which are business requirements.
- Agent task/result types are imported inward by application.

Recommended role:

- Keep model-specific prompt/decision-loop mechanics here.
- Move generic workflow policies to application or domain policy modules.
- Expose concrete runners that satisfy application ports.

### Infrastructure

Current role:

- Concrete adapters for local files, traces, datasets, repository metadata,
  repository tools, model providers, indexing, training, packaging, and
  evaluation.

Current drift:

- Contains workflows and policies that are not merely adapters.
- `training_artifacts.py` spans too many subdomains.
- Evaluation modules mix prompt payload creation, model calls, scoring rubrics,
  result aggregation, and reporting shapes.
- Some infrastructure modules import other infrastructure modules deeply, which
  is expected for adapters but becomes hard to manage when modules also contain
  use-case orchestration.

Recommended role:

- Provide concrete implementations of ports.
- Keep file formats, external libraries, subprocess-safe tools, indexes, and
  provider clients here.
- Avoid deciding high-level workflow success except when implementing a
  low-level adapter contract.

### Interfaces

Current role:

- CLI and MCP entrypoints.
- Runtime composition.
- Input parsing and output formatting.
- Several use cases and policies.

Current drift:

- `interfaces/cli.py` imports many infrastructure classes directly and
  orchestrates domain workflows.
- `interfaces/mcp_server.py` imports concrete agents, stores, tools, model
  providers, workspace registry, repository metadata, comparison traces, and
  FastMCP in one module.

Recommended role:

- Keep Typer and FastMCP details here.
- Translate user/protocol input into application request objects.
- Translate application results into stdout/stderr, exit codes, or MCP response
  dictionaries.
- Own protocol compatibility names and aliases.

## Architectural Findings

### 1. Interfaces contain use-case orchestration

`interfaces/cli.py` and `interfaces/mcp_server.py` should be thin adapters:
parse inputs, call application use cases, and translate results to CLI/MCP
responses. Today they also resolve model settings, construct runners, choose
tool policies, run datasets/training/evaluation, write reports, and format
domain-specific outcomes.

This makes the interface layer hard to test without Typer/FastMCP context and
encourages new behavior to keep landing at the edge.

### 2. MCP server is a composition root, adapter, and service module at once

`interfaces/mcp_server.py` currently contains:

- FastMCP server creation and tool/prompt registration.
- Public async functions such as `run_agent_loop`, `start_comparison_trace`,
  `record_comparison_event`, and workspace initialization.
- MCP-specific policy via `PatchPolicyToolExecutor`.
- Model-provider resolution and caching.
- Tool-name normalization and compatibility aliases.
- Trace-store/workspace-registry construction.
- Path parsing and environment flag helpers.

These are separable responsibilities. The FastMCP registration should be only
one adapter over application services.

### 3. Application depends on a concrete agent

`application/workflows.py` imports `micro_model_agent.agents.coding_agent`.
That reverses the intended direction. The application layer should depend on a
port such as `CodingWorkflowRunner` or an application-owned task/result contract,
not on a concrete agent implementation.

This is the most important dependency-direction issue because it makes
`agents` effectively part of the inner core while still living outside it.

### 4. Agents contain application policy

`agents/tool_loop_agent.py` does more than adapt a model. It owns policies such
as:

- required tool enforcement
- final-response gating
- verification-after-write enforcement
- duplicate write detection
- prompt/history shaping
- trace persistence and finalization

Some of this belongs in an application use case or policy module. The model loop
can remain as an agent service, but policy that describes what a valid workflow
must do should be separable and testable without a full model loop.

### 5. Infrastructure contains application workflows and domain-ish rubrics

Several infrastructure modules are adapters to files, Hugging Face, local
indexes, or tools. Others are richer workflows:

- `training_artifacts.py` mixes artifact persistence, training backends,
  training runner orchestration, evaluation result IO, promotion registry IO,
  and promotion policy.
- `synthetic_evaluation.py` and `workspace_staged_evaluation.py` contain
  evaluation use cases, scoring rules, and result record shapes.
- `trace_export.py` imports `application.workflows.TraceDatasetBuilder`, which
  is acceptable at first glance, but it also suggests trace-to-dataset export is
  a use case split across application and infrastructure.

The infrastructure layer should keep adapters; application should own the
workflow boundaries and domain/rubric modules should own scoring concepts when
they are framework-independent.

### 6. Tests mirror the current drift

Tests live beside implementation files, which is fine for this repository's
style, but the largest tests exercise large modules. After extraction, tests
should move with the behavior they verify:

- interface tests should verify adapter binding, argument parsing, and response
  translation
- application tests should verify use-case orchestration
- infrastructure tests should verify filesystem, tool, model, and registry
  adapters
- domain tests should verify pure rules and value objects

## Target Architecture

Keep the current top-level packages, but make directory depth express both the
architectural layer and the mid-level feature/protocol area. The goal is not
deep nesting for its own sake; it is to make "where should this code live?"
obvious before a file grows into a catch-all.

Recommended directory depth:

```text
high-level layer     domain / application / agents / infrastructure / interfaces
mid-level area       datasets / training / evaluation / promotion / mcp / cli / tools
leaf module          one cohesive policy, use case, adapter, command group, or registry
```

Examples:

- `interfaces/mcp/tools/run_loop.py` is the MCP adapter for one exposed MCP
  tool.
- `infrastructure/tools/repo_read.py` is the concrete repository tool
  implementation used by executors.
- `application/tool_loop/run.py` is the protocol-neutral use case.
- `domain/tool_policy.py` is pure workflow policy, if extracted.

With that structure, the target tree becomes:

```text
micro_model_agent/
  domain/
    contracts.py
    datasets.py
    training.py
    evaluation.py              # scoring/rubric value objects and pure rules
    tool_policy.py             # pure tool workflow rules, if extracted

  application/
    ports.py
    workflows.py               # generic workflow use cases, if still needed
    tool_loop/
      requests.py
      run.py                   # run-loop application service
      policy.py                # app-level orchestration policy, if not domain-pure
    datasets/
      build.py
      export.py
      review.py
      curate.py
      validate.py
    training/
      run.py
      artifacts.py             # application-facing artifact workflows
    evaluation/
      synthetic.py
      traces.py
      workspace_staged.py
      compare.py
    promotion/
      gate.py
      registry.py
      package.py

  agents/
    coding_agent.py            # concrete agent implementation
    tool_loop_agent.py         # model-loop engine, slimmer after policy extraction
    prompts.py                 # prompt rendering, if kept agent-specific
    history.py                 # prompt-history compaction, if kept agent-specific

  infrastructure/
    composition.py             # shared factories for adapters and config
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
      executor.py              # built-in executor/registry, if moved from root
      git_diff.py
      repo_read.py
      repo_search.py
      repo_semantic_search.py
      repo_write_files.py
      repo_write_patch.py
    training/
      hf_peft_backend.py
      local_runner.py
      ollama_packaging.py
    evaluation/
      synthetic_behavior.py
      trace_behavior.py
      workspace_staged.py

  interfaces/
    cli/
      __init__.py              # re-export app for console-script compatibility
      app.py                   # Typer app construction only
      common.py                # CLI-only parsing/output helpers
      commands/
        repo.py
        loop.py
        dataset.py
        train.py
        eval.py
        promote.py
    mcp/
      __init__.py
      server.py                # FastMCP construction only
      tools/
        __init__.py
        run_loop.py            # micro_agent_run_loop wrapper
        workspace.py           # micro_agent_init_workspace
        comparison_trace.py    # start/record/stop/review trace tools
        builtin.py             # debug built-in tool bridge
        trace_read.py          # debug trace reading
        init.py                # conditional repository init tool
        registry.py            # tool registration coordinator
      prompts/
        __init__.py
        comparison.py
        trace_collection.py
        smoke_test.py
        registry.py            # prompt registration coordinator
      policy/
        __init__.py
        patch_policy.py        # MCP dry-run/apply-patches adapter policy
        tool_names.py          # compatibility aliases and normalization
        exposure.py            # debug/init exposure decisions
      schemas.py               # MCP request/response DTOs if needed
      workspace.py             # workspace-id resolution helpers
      traces.py                # comparison/workflow trace adapter helpers
```

This exact tree is a guide, not a mandate. Prefer smaller extractions that
follow existing names over a large package reshuffle. The important distinction
is that `interfaces/mcp/tools/` contains protocol wrappers for MCP tool
registration, while `infrastructure/tools/` contains the actual repository tool
implementations and contracts.

## Dependency Rules

Adopt these rules as acceptance criteria:

```text
domain -> standard library only
application -> domain + application ports
agents -> application ports + domain
infrastructure -> application ports + domain + external libraries
interfaces -> application use cases + infrastructure composition + external interface libraries
```

Specific constraints:

- `application` must not import `agents`, `infrastructure`, `interfaces`, Typer,
  FastMCP, or model/tool concrete classes.
- `domain` must not import Pydantic, Typer, FastMCP, Hugging Face, file-system
  adapters, or repository tools.
- `interfaces` may construct concrete adapters, but business decisions should
  happen in application services.
- compatibility aliases and protocol-specific response shapes should remain at
  the boundary.

## Current Dependency Findings

Known production-layer violations or suspicious edges:

| Import Edge | Example | Why It Matters | Suggested Fix |
| --- | --- | --- | --- |
| `application -> agents` | `application/workflows.py` imports `CodingAgent`, `CodingAgentTask`, `CodingAgentResult` | Application depends on a concrete runner instead of a port. | Introduce application-owned workflow runner port and task/result DTOs. |
| `interfaces -> agents` | `interfaces/mcp_server.py` imports `ToolLoopAgent` | Acceptable only as composition, but it is mixed with use-case logic. | Move composition to factory module and orchestration to application. |
| `interfaces -> many infrastructure modules` | `interfaces/cli.py` imports stores, validators, evaluators, training artifacts, packaging, metadata | CLI is doing use-case assembly and orchestration. | Split CLI adapter, add application services, centralize runtime factories. |
| `infrastructure -> application.workflows` | `infrastructure/trace_export.py` imports `TraceDatasetBuilder` | Infrastructure adapter depends on an application service. This may be okay only if the module is really an application export use case, not infrastructure. | Move trace export orchestration to `application/datasets.py`; keep JSONL/file writing in infrastructure. |
| `infrastructure evaluation -> application ports` | `synthetic_evaluation.py`, `workspace_staged_evaluation.py` import `ModelProvider` | A concrete evaluation suite can implement an application port, but current modules also own use-case behavior. | Split pure rubrics, application suite orchestration, and infrastructure serialization. |

Edges that look acceptable:

- `agents -> application.ports` and `agents -> domain`: concrete agents can
  implement application ports.
- `infrastructure -> application.ports` and `infrastructure -> domain`: concrete
  adapters can implement ports and serialize domain contracts.
- `interfaces -> infrastructure`: allowed for composition only, but currently
  too much behavior is coupled to those imports.

## Boundary Test Design

Add a test such as `src/micro_model_agent/test_architecture_boundaries.py`.
Start with known exceptions, then remove exceptions as phases complete.

Suggested first version:

```python
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent

FORBIDDEN_IMPORTS = {
    "domain": (
        "micro_model_agent.application",
        "micro_model_agent.agents",
        "micro_model_agent.infrastructure",
        "micro_model_agent.interfaces",
    ),
    "application": (
        "micro_model_agent.agents",
        "micro_model_agent.infrastructure",
        "micro_model_agent.interfaces",
    ),
}

KNOWN_VIOLATIONS = {
    ("application/workflows.py", "micro_model_agent.agents.coding_agent"),
}


def test_layer_import_boundaries() -> None:
    violations: list[tuple[str, str]] = []
    for layer, forbidden_prefixes in FORBIDDEN_IMPORTS.items():
        for path in (ROOT / layer).glob("*.py"):
            if path.name.startswith("test_"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported = _import_name(node)
                if imported is None:
                    continue
                if any(imported.startswith(prefix) for prefix in forbidden_prefixes):
                    relative = path.relative_to(ROOT).as_posix()
                    violation = (relative, imported)
                    if violation not in KNOWN_VIOLATIONS:
                        violations.append(violation)

    assert violations == []


def _import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom):
        return node.module
    if isinstance(node, ast.Import) and node.names:
        return node.names[0].name
    return None
```

When Phase 4 is complete, delete `KNOWN_VIOLATIONS`. Later, expand the test to
catch external framework imports in `domain` and `application`, such as `typer`,
`mcp`, `pydantic`, `torch`, `transformers`, `peft`, and `datasets`.

## Characterization Test Plan

Before moving code, lock down behavior that users and downstream traces rely on.

### CLI Characterization

Use Typer's `CliRunner` or subprocess-based tests for:

- `micro-agent --help` lists root commands and groups.
- `micro-agent dataset --help`, `train --help`, `eval --help`, and
  `promote --help` list current subcommands.
- `init` creates metadata and is idempotent.
- `index` reports indexed files, terms, skipped files, source types, and code
  metadata.
- `task` with `StaticModelProvider` still writes/saves traces and optional
  labeled examples.
- `loop` with scripted responses preserves trace output, allowed tool handling,
  schema prompt behavior, and nonzero exit behavior.
- Dataset commands preserve JSONL output shapes.
- Training dry-run preserves run metadata and artifact records.
- Promotion commands preserve gate report, registry, list/select behavior, and
  Ollama packaging invocation shape.

### MCP Characterization

Test at the Python function level first:

- `create_mcp_server` exposes default tools and prompts.
- debug tools appear only when debug exposure is enabled.
- init tool appears under the current initialization/exposure conditions.
- `run_agent_loop` with scripted responses preserves:
  - required tool handling
  - available tool normalization
  - `apply_patches=false` dry-run behavior
  - `allow_test_run` behavior
  - run-profile defaults
  - comparison trace event append behavior
- workspace ID resolution still maps through `JsonlWorkspaceRegistry`.

### Trace And Dataset Characterization

Trace and dataset compatibility should be treated as schema compatibility:

- Save a golden workflow trace generated by a scripted model.
- Save a golden comparison trace session.
- Save a golden dataset export from trace-derived examples.
- Assert important keys and enum values, not exact timestamps or UUIDs.
- Prefer JSON schema-style assertions over large string snapshots.

### Architecture Characterization

Add metrics-style tests only if they are helpful and low-noise:

- production module line-count warnings can be documented rather than enforced
  initially
- import-boundary tests should be enforced
- public entrypoint import tests should be enforced:
  - `from micro_model_agent.interfaces.cli import app`
  - `from micro_model_agent.interfaces.mcp_server import create_mcp_server`
  - `from micro_model_agent.interfaces.mcp_server import run_agent_loop`

## Refactor Plan

### Phase 0: Guardrails

Goal: make it safe to move code.

1. Add an import-boundary test.
   - Fail if `application` imports `agents`, `infrastructure`, or `interfaces`.
   - Fail if `domain` imports anything project-specific outside `domain`.
   - Initially mark known violations explicitly so the test can be introduced
     before all fixes are done.

2. Add characterization tests around public behavior before extraction.
   - CLI command smoke tests for command groups and key outputs.
   - MCP server registration tests for exposed tool and prompt names.
   - `run_agent_loop` behavior tests for dry-run patch policy, required tools,
     workspace resolution, and scripted model responses.

3. Decide a module-size target.
   - Suggested soft limit: production modules over 500 lines need an explicit
     reason or an extraction task.
   - Suggested hard review trigger: modules over 800 lines.

### Phase 1: Split MCP Without Moving Business Logic

Goal: make `mcp_server.py` small while preserving imports and public API.

1. Create `interfaces/mcp/`.
2. Move MCP registration into `interfaces/mcp/server.py`.
3. Move prompt registration strings into an `interfaces/mcp/prompts/` package.
4. Move tool registration wrappers into an `interfaces/mcp/tools/` package.
   - `run_loop.py` for `micro_agent_run_loop`
   - `workspace.py` for workspace initialization
   - `comparison_trace.py` for comparison trace tools
   - `builtin.py` and `trace_read.py` for debug tools
   - `init.py` for the conditional repository init tool
   - `registry.py` for binding all MCP tools to `FastMCP`
5. Move MCP-only policy and compatibility helpers into an
   `interfaces/mcp/policy/` package.
   - `PatchPolicyToolExecutor`
   - compatibility aliases
   - allowed/required tool normalization
   - debug/init exposure flags
6. Keep `interfaces/mcp_server.py` as a compatibility shim exporting the current
   public functions and `main`.

Acceptance criteria:

- `micro-agent serve-mcp` still works.
- Existing imports from `micro_model_agent.interfaces.mcp_server` still work.
- MCP registration tests assert the same tool/prompt names as before.
- `interfaces/mcp_server.py` drops below roughly 150 lines.

### Phase 2: Extract Shared Composition

Goal: stop duplicating model/config/trace/tool assembly in CLI and MCP.

1. Create `infrastructure/composition.py` or `infrastructure/runtime.py`.
2. Move shared factories out of interface modules:
   - repository config loading and model setting resolution
   - model provider construction and cache access
   - built-in tool executor construction
   - trace store and comparison trace store construction
   - workspace registry construction
   - allowed test command construction
3. Keep interface-specific defaults at the interface boundary, but pass them to
   factories as explicit parameters.

Acceptance criteria:

- CLI and MCP call the same factory functions for equivalent runtime assembly.
- Interface modules no longer import low-level model provider classes directly
  unless they are part of command-specific behavior.
- Model setting resolution has focused tests outside Typer/FastMCP.

### Phase 3: Move Run-Loop Use Case Into Application

Goal: make the model-driven loop an application use case, not an MCP helper.

1. Add an application service such as `RunToolLoopWorkflow`.
2. Define application-owned request/result dataclasses if the current MCP
   dictionary response is too boundary-specific.
3. Move orchestration currently in `interfaces/mcp_server.py::run_agent_loop`
   into the new service:
   - run-profile setting application
   - allowed/required tool policy inputs
   - tool schema selection
   - trace/comparison event recording hooks
4. Keep MCP response dictionaries in the MCP adapter.
5. Let CLI `loop` use the same application service where practical.

Acceptance criteria:

- Application can run the tool loop with fake model/tool/trace ports and no MCP
  imports.
- MCP adapter translates request/result only.
- CLI and MCP behavior remains compatible.

### Phase 4: Remove Application-to-Agent Dependency

Goal: restore inward dependency direction.

1. Introduce a port in `application/ports.py`, for example:

   ```python
   class CodingWorkflowRunner(Protocol):
       trace_store: TraceStore

       async def run(self, task: CodingWorkflowTask) -> CodingWorkflowResult:
           ...
   ```

2. Move `CodingAgentTask` and `CodingAgentResult` inward if they are generic
   use-case contracts, or create application equivalents and adapt the concrete
   `CodingAgent`.
3. Update `RunAgentWorkflow` to depend on the new port instead of importing
   `agents.coding_agent`.
4. Keep `agents.coding_agent.CodingAgent` as one implementation of the port.

Acceptance criteria:

- `application/workflows.py` has no import from `micro_model_agent.agents`.
- Existing workflow tests pass through the port.
- Concrete agent tests remain in `agents`.

### Phase 5: Split CLI Into Command Modules

Goal: keep Typer as an adapter, not a home for all workflows.

1. Create `interfaces/cli/`.
2. Move app construction into `interfaces/cli/app.py`.
3. Move common CLI helpers into `interfaces/cli/common.py`.
4. Split command modules by subdomain:
   - `repo.py`: `init`, `index`, possibly `task`
   - `loop.py`: model loop command
   - `dataset.py`: synthesize, validate, export, review, relabel, merge
   - `train.py`: synthetic training
   - `eval.py`: synthetic, traces, workspace-staged, compare
   - `promote.py`: gate, record, list, select, package-ollama
5. Convert `interfaces/cli.py` into an `interfaces/cli/` package rather than
   trying to keep both a file and directory with the same name. Preserve
   `micro_model_agent.interfaces.cli:app` by exporting `app` from
   `interfaces/cli/__init__.py`.

Acceptance criteria:

- `micro-agent` still resolves to the Typer app.
- Command modules contain Typer-specific parsing and output only.
- Dataset/training/evaluation command handlers delegate to application
  services.
- `interfaces/cli/__init__.py` contains only re-exports and compatibility
  aliases.

### Phase 6: Split Training and Promotion Infrastructure

Goal: make `training_artifacts.py` cohesive.

Suggested extractions:

- `infrastructure/persistence/training_artifacts.py`
  - `JsonTrainingArtifactStore`
  - artifact/run serialization helpers
- `infrastructure/training/fake_runner.py`
  - `FakeTrainingRunner`
- `infrastructure/training/hf_peft_backend.py`
  - `HuggingFacePeftFineTuningBackend`
  - PEFT/Hugging Face-specific parameter parsing
- `infrastructure/training/local_runner.py`
  - `LocalFineTuningRunner`
- `infrastructure/evaluation/artifact_suite.py`
  - `SyntheticEvaluationSuite`
- `infrastructure/promotion/registry.py`
  - promotion registry serialization and IO
- `application/promotion.py` or `domain/training.py`
  - `MinimumScorePromotionPolicy`, depending on whether it remains pure

Acceptance criteria:

- Hugging Face imports are isolated to the Hugging Face backend module.
- Promotion registry IO is separate from promotion policy.
- Local runner tests do not need to import unrelated registry or evaluation IO.

### Phase 7: Split Evaluation Rubrics From Adapters

Goal: make evaluation rules testable as pure code where possible.

1. Extract pure score/value objects from:
   - `synthetic_evaluation.py`
   - `workspace_staged_evaluation.py`
2. Move model-calling evaluation suites to application or infrastructure based
   on dependency:
   - suites that call `ModelProvider` and orchestrate examples are application
     services
   - serialization/report writing stays infrastructure
   - pure rubric checks can live in domain or application policy modules
3. Split workspace-staged evaluation into:
   - prompt/payload construction
   - stage/rubric scoring
   - suite orchestration
   - review-record construction

Acceptance criteria:

- Rubric scoring can be tested without model providers or filesystem setup.
- Evaluation suite modules drop below roughly 400-500 lines.
- CLI evaluation commands call application services.

### Phase 8: Slim `ToolLoopAgent`

Goal: keep the agent responsible for the model/tool conversation while moving
portable workflow policy into smaller units.

Candidate extractions:

- `agents/prompts.py`
  - prompt rendering
  - tool schema formatting
  - orchestration hints if kept prompt-specific
- `agents/history.py`
  - tool history summarization
  - path compaction
  - output truncation
- `application/tool_policy.py` or `domain/tool_policy.py`
  - required tool satisfaction
  - verification-after-write rule
  - duplicate write policy
  - final-response gating
- `agents/model_decision.py`
  - JSON response parsing and markdown-fence stripping

Acceptance criteria:

- `ToolLoopAgent.run` reads as the main loop, not the whole policy library.
- Policy units have focused tests.
- No behavior changes to trace shape without explicit migration notes.

### Phase 9: Enforce Boundaries

Goal: prevent future drift.

1. Turn the import-boundary test from Phase 0 into a strict test with no known
   violations.
2. Add a lightweight architecture section to `CONTRIBUTING.md` or update
   `docs/architecture.md`.
3. Add review guidance:
   - new CLI/MCP behavior starts as an application use case
   - new external integrations start behind a port
   - new scoring/rubric logic should be pure before it is wired into an adapter

Acceptance criteria:

- `pytest` fails on boundary violations.
- `docs/architecture.md` matches the actual module layout.
- Public entrypoints remain backward compatible or have explicit deprecation
  notes.

## Suggested Work Order

Use this order in the implementation chat:

1. Add characterization and import-boundary tests.
2. Split `interfaces/mcp_server.py` with compatibility shims.
3. Extract shared runtime composition.
4. Move `run_agent_loop` orchestration into application.
5. Remove `application -> agents` dependency.
6. Split `interfaces/cli.py`.
7. Split `training_artifacts.py`.
8. Split evaluation rubrics and suites.
9. Slim `ToolLoopAgent`.
10. Update `docs/architecture.md` and make boundary tests strict.

## PR-Sized Migration Slices

The safest implementation path is many small pull requests or commits. Each
slice should keep tests green and preserve public imports.

### Slice 1: Architecture Baseline

Changes:

- Add `test_architecture_boundaries.py` with the known
  `application/workflows.py -> agents.coding_agent` exception.
- Add CLI command-surface tests.
- Add MCP registration-surface tests.
- Add a short `docs/architecture.md` note linking to this plan.

Exit criteria:

- No production code moves yet.
- Tests prove the current public surface.
- Known boundary violations are explicit.

### Slice 2: MCP Package Shell

Changes:

- Create `src/micro_model_agent/interfaces/mcp/__init__.py`.
- Create empty or minimally wired mid-level packages/modules:
  - `server.py`
  - `tools/__init__.py`
  - `tools/registry.py`
  - `prompts/__init__.py`
  - `prompts/registry.py`
  - `policy/__init__.py`
  - `compat.py`
- Move constants that are clearly MCP-boundary concepts:
  - `CANONICAL_TOOL_NAMES_TEXT`
  - `COMPAT_TOOL_ALIASES`
  - `COMPAT_REQUIRED_TOOL_ALIASES`
  - MCP env var names
  - MCP transport type alias
- Keep re-exports from `interfaces/mcp_server.py`.

Exit criteria:

- No behavior changes.
- `interfaces/mcp_server.py` still provides all old imports.

### Slice 3: MCP Policy Extraction

Changes:

- Move `PatchPolicyToolExecutor` to `interfaces/mcp/policy/patch_policy.py`.
- Move `_allowed_tool_names`, `_required_tool_names`,
  `_normalized_tool_names`, and `_allowed_test_commands` to
  `interfaces/mcp/policy/tool_names.py`.
- Move debug/init exposure helpers to `interfaces/mcp/policy/exposure.py`.
- Move tool-list change notification helpers to the MCP server or registry
  module, depending on whether they need direct `FastMCP` access.
- Keep thin wrapper imports in `interfaces/mcp_server.py` if tests or users
  import private helpers today.

Exit criteria:

- Dry-run patch behavior remains covered.
- Compatibility aliases still normalize legacy names.

### Slice 4: MCP Registration Extraction

Changes:

- Move prompt builder functions to `interfaces/mcp/prompts/`.
- Move FastMCP tool registration to `interfaces/mcp/tools/registry.py`, with
  each exposed tool wrapper in its own module under `interfaces/mcp/tools/`.
  If nested decorators make a separate module awkward during the first move,
  register from `server.py` temporarily and split to the registry module in the
  next commit.
- Keep `create_mcp_server` in the new package and re-export it from
  `interfaces/mcp_server.py`.

Exit criteria:

- MCP tool and prompt names are unchanged.
- `interfaces/mcp_server.py` is mostly a shim plus `main`.

### Slice 5: Runtime Composition

Changes:

- Create `infrastructure/composition.py`.
- Move shared runtime factories:
  - model settings resolution
  - model provider construction and cache
  - allowed test command construction
  - trace store construction
  - comparison trace store construction
  - workspace registry construction
  - built-in tool executor construction
- Use these factories from MCP first, then CLI.

Exit criteria:

- Interface code no longer knows about low-level provider constructors except
  through composition helpers.
- Composition helper tests cover env/config/default precedence.

### Slice 6: Application Tool-Loop Use Case

Changes:

- Add `application/tool_loop.py`.
- Add request/result dataclasses with protocol-neutral names, for example:

  ```python
  @dataclass(frozen=True, slots=True)
  class RunToolLoopRequest:
      goal: str
      repository_root: str
      context: str = ""
      available_tools: tuple[str, ...] = ()
      required_tools: tuple[str, ...] = ()
      max_turns: int = 4
      max_tool_calls: int | None = 1
      capture_prompts: bool = False
      apply_patches: bool = False

  @dataclass(frozen=True, slots=True)
  class RunToolLoopResult:
      ok: bool
      response: str
      trace_id: str
      trace_path: str | None
      tool_calls_made: int
      turns_used: int
      details: dict[str, object]
  ```

- Move orchestration from MCP `run_agent_loop` behind this application service.
- Keep MCP dictionaries at the MCP boundary.

Exit criteria:

- Application tests run the use case with fake ports and no FastMCP.
- MCP and CLI can share the same use case.

### Slice 7: Application-Agent Inversion

Changes:

- Add an application workflow runner port.
- Move or duplicate minimal task/result contracts inward.
- Update `RunAgentWorkflow` to accept the port.
- Make `CodingAgent` implement the port.

Exit criteria:

- `application` has no production imports from `agents`.
- Boundary test known-violation list is empty for application.

### Slice 8: CLI Package Split

Changes:

- Replace the module file `interfaces/cli.py` with the package directory
  `interfaces/cli/`.
- Move Typer app creation to `interfaces/cli/app.py`.
- Move `_run`, `_fail`, `_load_dotenv`, formatting helpers, and CLI-only path
  parsing to `interfaces/cli/common.py`.
- Move command handlers by group.
- Preserve the public import path with `interfaces/cli/__init__.py`:

  ```python
  from micro_model_agent.interfaces.cli.app import app

  __all__ = ["app"]
  ```

  If tests import private helpers from `interfaces.cli`, keep temporary
  re-exports and add TODOs to migrate tests.

Exit criteria:

- Console script works.
- Help output and command behavior remain stable.
- `micro_model_agent.interfaces.cli` imports as a package and exposes `app`.

### Slice 9: Dataset, Training, Evaluation, Promotion Use Cases

Changes:

- Add application services only where CLI currently orchestrates multiple
  adapters or policies.
- Move file-format-specific code to infrastructure.
- Keep thin command handlers that create request objects and print results.

Exit criteria:

- Dataset/training/eval/promote commands mostly delegate.
- Infrastructure modules are more cohesive.

### Slice 10: Large Infrastructure Module Split

Changes:

- Split `training_artifacts.py`.
- Split `synthetic_evaluation.py`.
- Split `workspace_staged_evaluation.py`.
- Preserve old import paths with re-export modules until all internal imports
  and tests are migrated.

Exit criteria:

- No production module above 800 lines unless documented.
- Heavy optional dependencies are isolated.

### Slice 11: ToolLoopAgent Slimming

Changes:

- Extract decision parsing.
- Extract prompt rendering.
- Extract history compaction.
- Extract pure workflow policy.

Exit criteria:

- Trace shape stays compatible.
- Existing `agents/test_tool_loop_agent.py` coverage is preserved or moved with
  the extracted units.

## Compatibility Shim Pattern

Use shims during the migration so imports keep working while internal modules
move.

Example for `interfaces/mcp_server.py`:

```python
from micro_model_agent.interfaces.mcp.server import create_mcp_server, serve
from micro_model_agent.interfaces.mcp.tools.builtin import (
    call_builtin_tool,
    list_builtin_tools,
)
from micro_model_agent.interfaces.mcp.tools.run_loop import run_agent_loop
from micro_model_agent.interfaces.mcp.tools.trace_read import read_trace
from micro_model_agent.interfaces.mcp.tools.workspace import init_workspace
from micro_model_agent.interfaces.mcp.traces import (
    record_comparison_event,
    review_comparison_trace,
    start_comparison_trace,
    stop_comparison_trace,
)
from micro_model_agent.interfaces.mcp.workspace import init_repository


def main() -> None:
    ...
```

When a shim re-exports private helpers only for tests, add a comment and migrate
the tests in the same or next slice. Avoid keeping private compatibility forever.

## Application Service Design Guidelines

When extracting from CLI or MCP, prefer this shape:

1. Define a request dataclass in `application`.
2. Define a result dataclass in `application`.
3. Inject ports or concrete dependencies through the constructor.
4. Keep path strings or abstract identifiers in request objects unless the use
   case is explicitly repository-local.
5. Return structured results instead of printing, raising Typer exits, or
   returning MCP dictionaries.
6. Let adapters translate:
   - Typer options to request dataclasses
   - application errors to exit codes
   - application results to stdout/stderr
   - application results to MCP JSON-compatible dictionaries

Prefer this flow:

```text
CLI/MCP args
  -> interface DTO parsing
  -> application request
  -> application service
  -> ports
  -> infrastructure adapters
  -> application result
  -> interface response formatting
```

Avoid this flow:

```text
CLI/MCP args
  -> concrete infrastructure classes
  -> ad hoc dictionaries
  -> direct file writes
  -> mixed printing/reporting/business rules
```

## Naming Guidance

Use names that describe the business action, not the transport:

- Good application names:
  - `RunToolLoopWorkflow`
  - `BuildDatasetSplit`
  - `ValidateDatasetExamples`
  - `ExportTraceDataset`
  - `RunTrainingWorkflow`
  - `EvaluateModelArtifact`
  - `ApplyPromotionGate`
- Keep transport names at the edge:
  - `mcp_run_agent_loop`
  - `promote_gate`
  - `eval_workspace_staged`

For infrastructure packages, name by external mechanism:

- `persistence`
- `models`
- `repositories`
- `tools`
- `training`
- `evaluation`
- `promotion`

## Non-Goals

Do not combine this refactor with:

- changing trace JSON formats
- changing CLI command names or MCP tool names
- replacing Typer/FastMCP
- changing training/evaluation semantics
- introducing a dependency injection framework
- moving tests out of `src` unless the project decides to change test layout

## Risks

- Public import compatibility: tests currently import helpers from
  `interfaces.mcp_server` and `interfaces.cli`. Keep shims until callers are
  migrated.
- Trace compatibility: downstream docs and datasets likely rely on current trace
  shapes. Treat trace schema changes as separate migrations.
- Over-extraction: split by real responsibilities, not by one-class-per-file
  rules.
- Hidden behavior in CLI output: command tests should capture important stdout
  and stderr before moving handlers.
- Optional training dependencies: keep heavy imports isolated so basic CLI/MCP
  usage does not require the `training` dependency group.
- CLI package conversion: `interfaces/cli.py` must be replaced by an
  `interfaces/cli/` package in one atomic move because a file and directory
  cannot share that path name.

## Definition of Done

The refactor is complete when:

- `domain` and `application` import only allowed layers.
- CLI and MCP modules are thin adapters with compatibility shims.
- Use-case orchestration lives in `application`.
- Files over 800 lines are either split or explicitly documented.
- Public CLI commands and MCP tool/prompt names are unchanged.
- Existing tests pass, and architecture boundary tests protect the new shape.
