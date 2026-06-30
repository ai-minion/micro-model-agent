# Architecture

micro-model-agent is organized as a DDD-oriented Clean/Hexagonal architecture: domain concepts and policies sit at the center, application services orchestrate use cases, and infrastructure/interfaces connect through explicit ports and adapters.

## Philosophy

The model is a workflow executor. The platform owns retrieval, tool execution,
trace capture, verification, evaluation, and future training pipelines.

```text
User
  -> Workflow
  -> Retrieval
  -> Tools
  -> Verification
```

## Layers

### Domain

Pure business contracts and policy. No infrastructure, framework, network, file
system, or MCP dependencies.

Implemented concepts:

- `AgentProfile`
- `ModelProfile`
- `ToolDefinition`
- `ToolCall`
- `ToolResult`
- `WorkflowTrace`
- `WorkflowStep`
- `RepositoryProfile`
- `RetrievalQuery`
- `RetrievalResult`
- `SemanticSearchResult`
- `EvaluationResult`

### Application

Use cases and orchestration. This layer coordinates domain contracts with
infrastructure ports.

Implemented services:

- `RunAgentWorkflow`
- `RunToolLoopWorkflow`
- `RunDatasetSynthesisWorkflow`
- `RunDatasetValidationWorkflow`
- `RunDatasetExportWorkflow`
- `RunDatasetMergeWorkflow`
- `RunDatasetRelabelWorkflow`
- `RunTraceDatasetExportWorkflow`
- `RunTraceReviewWorkflow`
- `RunSyntheticTrainingWorkflow`
- `RunSyntheticEvaluationWorkflow`
- `RunTraceEvaluationWorkflow`
- `RunWorkspaceStagedEvaluationWorkflow`
- `RunWorkspaceStagedReviewWorkflow`
- `RunEvaluationComparisonWorkflow`
- `RunPromotionGateWorkflow`
- `RunPromotionRecordWorkflow`
- `RunPromotionListWorkflow`
- `RunPromotionSelectWorkflow`
- `RunPromotionPackageOllamaWorkflow`
- `TraceDatasetBuilder`
- `DefaultWorkflowEvaluator`

Application-owned ports include `CodingWorkflowRunner`, `ModelProvider`,
`ToolExecutor`, `TraceStore`, and dataset/training/evaluation storage and runner
contracts. The reference `CodingAgent` implements `CodingWorkflowRunner`; the
application layer depends on that port rather than importing the concrete agent.

The model-driven tool-loop use case is represented by `RunToolLoopWorkflow`,
`RunToolLoopRequest`, and `RunToolLoopResult`. The reference
`agents.ToolLoopAgent` implements the application `ToolLoopRunner` port, while
safe tool execution is exposed through the `ToolExecutor` application port and
implemented by `BuiltinToolExecutor`.

Dataset synthesis, validation, export, merge, and relabel use cases are
represented by `RunDatasetSynthesisWorkflow`, `RunDatasetValidationWorkflow`,
`RunDatasetExportWorkflow`, `RunDatasetMergeWorkflow`, and
`RunDatasetRelabelWorkflow`. Trace-derived dataset export is represented by
`RunTraceDatasetExportWorkflow`; human trace review recording is represented by
`RunTraceReviewWorkflow`. Interfaces supply CLI options and output formatting,
infrastructure owns template-based generation, JSONL dataset loading/writing,
trace/review JSONL loading, trace-example export, review persistence, concrete
validation rules, merge/deduplication and relabel policies, and SFT JSONL
writing, and the application coordinates the generator, reader, writer,
validator, merger, relabeler, trace exporter, review writer, and exporter ports.

Synthetic fine-tuning is represented by `RunSyntheticTrainingWorkflow`.
Interfaces supply CLI options, environment loading, backend selection, output
formatting, and exit behavior. Infrastructure owns JSONL dataset loading,
validation rules, SFT JSONL rendering, dataset metadata hashing/profile
summaries, concrete training backends, and artifact persistence. The
application coordinates validation, run-local export, training config assembly,
runner execution, and artifact recording through ports.

Synthetic behavior and metadata-only artifact evaluation are represented by
`RunSyntheticEvaluationWorkflow`; trace-derived behavior evaluation is
represented by `RunTraceEvaluationWorkflow`; staged workspace behavior
evaluation and review queue generation are represented by
`RunWorkspaceStagedEvaluationWorkflow` and `RunWorkspaceStagedReviewWorkflow`.
Evaluation report comparison is represented by `RunEvaluationComparisonWorkflow`.
Interfaces supply CLI model selection, interactive review prompts, threshold
parsing, output formatting, and exit behavior. Infrastructure owns persisted
report loading, JSON/JSONL report writing, concrete behavior/artifact
evaluators, model providers, staged review record construction, and dataset
metadata helpers, while the application coordinates dataset loading, evaluation
execution, report metadata, score/metric comparison, review queue
build/write orchestration, and report persistence through ports.

Promotion gate, registry-record, registry-list, model-selection, and Ollama
packaging use cases are represented by `RunPromotionGateWorkflow`,
`RunPromotionRecordWorkflow`, `RunPromotionListWorkflow`,
`RunPromotionSelectWorkflow`, and `RunPromotionPackageOllamaWorkflow`.
Interfaces supply CLI options and output formatting, infrastructure owns JSON
artifact/report, registry, repository configuration, and Ollama package storage,
and the application coordinates promotion policy, registry orchestration,
repository-local model selection, and packaging requests.

### Infrastructure

Adapters for external systems and local capabilities:

- `OllamaModelProvider`
- `LocalSemanticRetriever`
- `LocalLexicalIndexWriter`
- `LocalLexicalIndexReader`
- `BuiltinToolExecutor`
- `RepoSearchTool`
- `RepoReadTool`
- `RepoSemanticSearchTool`
- `RepoWritePatchTool`
- `TestRunTool`
- `GitDiffTool`
- `TraceStore`
- `McpServerAdapter`

`LocalSemanticRetriever` adapts the same lexical repository retrieval used by
`repo.semantic_search` to the application `SemanticRetriever` port. Future
vector or hybrid stores can be added behind that port without changing domain
contracts.

### Interfaces

User-facing entrypoints:

- CLI
- MCP server
- Public Python API

### Agents

Reference agents built from workflows and tools. The first implementation is
`CodingAgent`; `ToolLoopAgent` is the reference model-driven implementation of
the application tool-loop runner port.

## Dependency Direction

```text
interfaces -> application -> domain
agents -> application -> domain
infrastructure -> application/domain ports
domain -> nothing project-specific
```

## Dependency Injection

Runtime assembly should happen at the edges, usually in `interfaces` or a small
composition module. Application services accept explicit dependencies through
constructors. Domain objects remain plain contracts and policy.

## Retrieval

Retrieval is a first-class platform capability. The local implementation starts
simple, but contracts should support source code, docs, ADRs, traces, recipes,
error history, and future vector stores such as Qdrant and Chroma.

## Trace Capture

Trace capture is automatic orchestration behavior. The model never records its
own traces. Workflows produce `WorkflowTrace` records containing goals, tool
calls, tool results, generated outputs, verification results, and outcomes.

## Fine-Tuning Data

Fine-tuning is a core goal. V1 should capture training-ready data and include a
minimal synthetic-data training pipeline. CLI workflows should store labeled good
and bad outcomes, tool-use examples, retrieved documentation context, codebase
context, patches, verification results, and reviewer notes. Synthetic examples
should be validated against tool contracts before training. See
[fine-tuning-data-plan.md](fine-tuning-data-plan.md) and
[training-pipeline.md](training-pipeline.md).

## Security

MCP and CLI tools must avoid arbitrary shell execution and unrestricted writes.
Dangerous operations need explicit allowlists, dry-run support, patch previews,
and approval workflows.

## Planning

The implementation roadmap, milestones, tool plan, testing strategy, and MVP
acceptance criteria live in [project-plan.md](project-plan.md).

The current architecture refactor handoff and migration slices live in
[architecture-refactor-plan.md](architecture-refactor-plan.md).
