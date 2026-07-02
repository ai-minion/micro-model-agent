# Architecture Refactor Follow-Up

This is an older continuation handoff for the architecture refactor. The current
handoff lives in `docs/architecture-refactor-left-to-do.md`; use that document
first unless deeper historical context is needed.

## Current State

Completed through the CLI package conversion, common-helper extraction,
command-module split, and the first CLI-orchestration move:

- Architecture boundary tests exist for `domain` and `application`.
- CLI and MCP public surface characterization tests exist.
- The production `application -> agents` dependency has been removed.
- `interfaces/mcp_server.py` is now a compatibility shim over
  `interfaces/mcp/*`.
- Shared runtime assembly remains available through
  `infrastructure.composition`.
- Model runtime helpers live in `infrastructure/models/runtime.py` and remain
  available to interface adapters through `infrastructure.composition`.
- Workflow factory helpers live in package-level runtime modules under
  `infrastructure/agents/`, `infrastructure/repositories/`,
  `infrastructure/datasets/`,
  `infrastructure/training/`, `infrastructure/evaluation/`, and
  `infrastructure/promotion/`; they remain available to interface adapters
  through `infrastructure.composition`.
- Trace/workspace persistence runtime helpers live in
  `infrastructure/persistence/runtime.py` and remain available to interface
  adapters through `infrastructure.composition`.
- Built-in tool runtime helpers live in `infrastructure/tools/runtime.py` and
  remain available to interface adapters through `infrastructure.composition`.
- MCP runtime assembly uses composition helpers.
- CLI `loop` uses composition helpers for model, trace, tool executor, model
  option resolution, and runtime response assembly.
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
  `.env` loading, and scripted-response loading. Private loop model-resolution
  helpers have been retired; tests now target the composition owner.
- `interfaces/cli/__init__.py` re-exports only `app` for the console script.
- `interfaces/cli/commands/repo.py` owns root `init`, `index`, and `task`
  command handlers.
- `interfaces/cli/commands/loop.py` owns the root `loop` command handler.
- `interfaces/cli/commands/mcp.py` owns the root `serve-mcp` command handler.
- `interfaces/cli/commands/dataset.py` owns the `dataset` command group.
- `interfaces/cli/commands/train.py` owns the `train synthetic` command.
- `interfaces/cli/commands/eval.py` owns the `eval` command group.
- `interfaces/cli/commands/promote.py` owns the `promote` command group.
- `application/promotion.py` owns `RunPromotionGateWorkflow`,
  `RunPromotionRecordWorkflow`, `RunPromotionListWorkflow`,
  `RunPromotionSelectWorkflow`, and `RunPromotionPackageOllamaWorkflow`; the
  full `promote` CLI group now delegates artifact loading, evaluation report
  loading, policy checks, promotion report writing, registry recording/listing,
  registry lookup, repository config selection, and Ollama package requests
  through application ports.
- `LocalPromotionGateStore` adapts the existing training-artifact JSON helpers
  to the promotion application ports.
- `LocalRepositoryModelConfigurationStore` adapts repository config updates to
  the promotion selection application port.
- `LocalOllamaAdapterPackager` adapts Ollama Modelfile/manifest generation and
  optional `ollama create` execution to the promotion packaging application
  port.
- `application/datasets.py` owns `RunDatasetSynthesisWorkflow`,
  `RunDatasetValidationWorkflow`, `RunDatasetExportWorkflow`,
  `RunDatasetMergeWorkflow`, `RunDatasetRelabelWorkflow`, and
  `RunTraceDatasetExportWorkflow`, and `RunTraceReviewWorkflow`; `dataset
  synthesize`, `dataset validate`, `dataset export`, `dataset merge`, `dataset
  relabel`, `dataset export-traces`, and `dataset review-trace` now delegate
  template-based generation, JSONL loading/writing, validation,
  merging/deduplication, relabeling, trace/review loading, trace-derived export,
  corrected-target parsing, review persistence, and SFT JSONL export through
  application ports while keeping CLI output formatting in the adapter.
- `LocalDatasetExampleReader` adapts existing JSONL dataset loading helpers to
  the dataset validation application port.
- `LocalDatasetExampleWriter` adapts existing JSONL dataset writing helpers to
  the dataset merge application port.
- `LocalDatasetMerger` adapts existing dataset merge/deduplication helpers to
  the dataset merge application port.
- `LocalDatasetRelabeler` adapts existing dataset relabeling helpers to the
  dataset relabel application port.
- `SftJsonlDatasetExporter` adapts existing SFT JSONL export helpers to the
  dataset export application port.
- `LocalWorkflowTraceReader`, `LocalTraceReviewReader`,
  `LocalTraceReviewWriter`, `LocalTraceDatasetExporter`, and
  `LocalTraceDatasetExportValidator` adapt existing trace JSONL loading, review
  loading/writing, trace export, and trace export validation helpers to the
  trace dataset export and review application ports.
- `SyntheticTemplateGenerator` implements the dataset synthesis generator port.
- `application/training.py` owns `RunSyntheticTrainingWorkflow`; `train
  synthetic` now delegates dataset loading, validation, run-local SFT export,
  training config construction, backend execution, and artifact recording
  through application ports while keeping dotenv loading, backend selection,
  CLI output formatting, and exit behavior in the adapter.
- `LocalDatasetFileHasher` and `LocalDatasetToolProfileSummarizer` adapt
  existing dataset metadata helpers to the synthetic training application ports.
- `application/evaluation_workflows.py` owns `RunSyntheticEvaluationWorkflow`,
  `RunTraceEvaluationWorkflow`, `RunWorkspaceStagedEvaluationWorkflow`,
  `RunWorkspaceStagedReviewWorkflow`, and `RunEvaluationComparisonWorkflow`;
  `application/evaluation.py` remains as a compatibility facade.
  `eval synthetic` now delegates dataset loading, behavior/artifact evaluation,
  report metadata construction, and evaluation report writing through
  application ports while keeping environment loading, model selection, CLI
  output formatting, and exit behavior in the adapter. `eval traces` and `eval
  workspace-staged` now delegate dataset loading, behavior evaluation, report
  metadata construction, and evaluation report writing through application
  ports. `eval review-workspace-staged` now delegates dataset/report loading,
  review queue construction, and JSONL writing through application ports while
  keeping interactive prompts in the adapter. `eval compare` now delegates
  persisted report loading, score/metric comparison, and comparison report
  writing through application ports while keeping metric threshold option
  parsing, CLI output formatting, and exit behavior in the adapter.
- `LocalEvaluationResultReader` and `LocalEvaluationResultWriter` adapt existing
  evaluation report loading/writing to evaluation application ports.
- `LocalEvaluationComparisonReportWriter` adapts comparison report JSON writing
  to the evaluation comparison application port. The old
  `infrastructure.evaluation_comparison` comparison imports remain available as
  compatibility re-exports.
- `LocalWorkspaceStagedReviewBuilder` and
  `LocalWorkspaceStagedReviewQueueWriter` adapt staged workspace review record
  construction and JSONL queue writing to evaluation application ports.

Last known verification:

```text
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m ruff check src docs'
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m pytest'
```

Result:

```text
313 passed
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

## Next Slice: Larger Structural Cleanup

Goal: keep the completed CLI/application/MCP boundaries intact while reducing
the size and coupling of the remaining large modules.

Suggested implementation:

1. Pick one large infrastructure module, such as `training_artifacts.py`,
   `synthetic_evaluation.py`, or `workspace_staged_evaluation.py`.
2. Split cohesive helpers into smaller infrastructure modules while preserving
   existing public import paths with compatibility re-exports where needed.
3. Extract pure evaluation rubrics where scoring can be tested without model
   providers or filesystem setup.
4. Keep CLI adapters responsible for Typer options, printing, interactive
   prompts, and exit behavior.

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
