# Architecture

micro-model-agent is organized as a classical DDD (Domain-Driven Design)
architecture with bounded contexts as the top-level structural unit. Each context
owns its own domain, application, and infrastructure layers. The shared kernel
provides base types used across all contexts.

## Philosophy

The model is a workflow executor. The platform owns retrieval, tool execution,
trace capture, verification, evaluation, and training pipelines. Domain policy
and invariants live in aggregate roots; application workflows are pure
orchestrators that call domain methods, persist via repositories, and publish
events.

```text
User
  -> Workflow
  -> Retrieval
  -> Tools
  -> Verification
```

## Package Structure

```text
micro_model_agent/
  shared/           <- Shared Kernel (cross-context base types only)
  execution/        <- Bounded Context: running agent workflows & tool loops
  dataset/          <- Bounded Context: training dataset lifecycle
  training/         <- Bounded Context: fine-tuning job lifecycle
  evaluation/       <- Bounded Context: model evaluation & scoring
  promotion/        <- Bounded Context: model promotion & registry
  repository_ops/   <- Supporting Context: source code repo retrieval
  interfaces/       <- Cross-cutting: CLI and MCP entry points
```

Each bounded context follows a consistent internal layering:

```text
<context>/
  domain/           <- Aggregates, entities, value objects, events,
                       repository Protocols, domain services
  application/      <- Commands/results, workflow orchestration, port Protocols
  infrastructure/   <- Concrete adapters, repositories, ACLs, composition/factory
```

## Shared Kernel (`shared/`)

Minimal base types shared across all contexts. Never imports from any bounded
context.

- `DomainEvent` — frozen immutable base for all domain events (`event_id`, `occurred_at`)
- `Entity` — UUID identity, equality by id, pending-events list with `pull_events()`
- `EventBus` Protocol — `publish(event)` / `publish_all(events)`
- `InProcessEventBus` — synchronous in-process implementation; handlers registered by event type
- `DomainException` — root for all domain rule violations
- `EvaluationResult` — shared VO used by both evaluation and promotion contexts

## Bounded Contexts

### `execution/` — Workflow Execution

Runs agent workflows and model-driven tool loops.

**Aggregate:** `WorkflowExecution` — guarded state machine with transitions
`start()` → `add_step()` → `complete()` / `fail()`. Emits
`WorkflowStarted`, `StepAdded`, `WorkflowCompleted`, `WorkflowFailed`.

**Application workflows:** `RunAgentWorkflow`, `RunToolLoopWorkflow`

**Infrastructure:** `JsonlWorkflowRepository`, model providers
(`OllamaModelProvider`, `TransformersPeftModelProvider`, `StaticModelProvider`,
`ScriptedModelProvider`), `JsonlTraceStore`, `build_coding_workflow()`,
`build_event_pipeline()` (wires `InProcessEventBus` with all cross-context handlers)

### `dataset/` — Dataset Lifecycle

Manages training dataset examples: synthesis, validation, export, merge, relabel,
and trace-derived examples.

**Aggregate:** `Dataset` — owns `DatasetExample` entities; enforces
`DuplicateExampleError` / `ExampleNotFoundError` on `add_example()` / `relabel()`.
Emits `ExampleAdded`, `ExampleLabelled`, `DatasetExported`, `DatasetMerged`.

**Application workflows:** `RunDatasetSynthesisWorkflow`,
`RunDatasetValidationWorkflow`, `RunDatasetExportWorkflow`,
`RunDatasetMergeWorkflow`, `RunDatasetRelabelWorkflow`,
`RunTraceDatasetExportWorkflow`, `RunTraceReviewWorkflow`

**ACL:** `OnWorkflowCompleted` event handler translates `WorkflowCompleted`
(execution) into `DatasetExample` via `ExecutionToDatasetTranslator`

**Infrastructure:** `JsonlDatasetRepository`, `SyntheticTemplateGenerator`,
trace export and review adapters

### `training/` — Fine-Tuning Job Lifecycle

Runs PEFT fine-tuning jobs and stores artifacts.

**Aggregate:** `TrainingJob` — transitions `start()` → `complete()` / `fail()`.
Emits `TrainingJobCreated`, `TrainingJobStarted`, `TrainingJobCompleted`,
`TrainingJobFailed`, `ArtifactProduced`.

**Application workflows:** `RunSyntheticTrainingWorkflow`

**Infrastructure:** `LocalFineTuningRunner`, `FakeTrainingRunner`,
`JsonTrainingArtifactStore`, `JsonlTrainingJobRepository`,
SFT JSONL export, Ollama adapter packaging

### `evaluation/` — Model Evaluation & Scoring

Evaluates model behavior against synthetic benchmarks, trace replay, and
staged workspace tasks.

**Aggregate:** `EvaluationReport` — collects `EvaluationResult` entities and
`finalize(score)` emits `EvaluationCompleted` plus `ThresholdMet` or
`ThresholdBreached` based on `EvaluationThreshold`.

**Application workflows:** `RunSyntheticEvaluationWorkflow`,
`RunTraceEvaluationWorkflow`, `RunWorkspaceStagedEvaluationWorkflow`,
`RunWorkspaceStagedReviewWorkflow`, `RunEvaluationComparisonWorkflow`

**Event handler:** `OnArtifactProduced` triggers evaluation when training
produces a new artifact

**Infrastructure:** `JsonlEvaluationReportRepository`, rubric suites
(`SyntheticEvaluationSuite`, `TraceBehaviorEvaluationSuite`,
`WorkspaceStagedEvaluationSuite`)

### `promotion/` — Model Promotion & Registry

Gates and records model promotions; packages adapters for Ollama.

**Aggregate:** `ModelRegistry` — records `PromotedModel` entries; enforces
deduplication. Emits `ModelPromoted`, `PromotionGatePassed`,
`PromotionGateFailed`, `ModelPackaged`.

**Domain service:** `PromotionGateService` — evaluates `PromotionCriteria`
against `EvaluationScore` VOs and returns a `PromotionGateDecision`.

**Application workflows:** `RunPromotionGateWorkflow`,
`RunPromotionRecordWorkflow`, `RunPromotionListWorkflow`,
`RunPromotionSelectWorkflow`, `RunPromotionPackageOllamaWorkflow`

**Event handler:** `OnThresholdEvent` reacts to `ThresholdMet` /
`ThresholdBreached` events from the evaluation context

**Infrastructure:** `JsonlModelRegistryRepository`,
`LocalOllamaAdapterPackager`

### `repository_ops/` — Source Code Retrieval (Supporting)

Exposes safe, constrained repository tools used by the execution context.

**Value objects:** `RepositoryProfile`, `RetrievalQuery`, `RetrievalResult`,
`SemanticSearchResult`

**Tools:** `RepoSearchTool`, `RepoReadTool`, `RepoWritePatchTool`,
`RepoWriteFilesTool`, `RepoSemanticSearchTool`, `GitDiffTool`, `TestRunTool`

**Infrastructure:** `LocalSemanticRetriever`, `LocalLexicalIndexWriter`,
`LocalLexicalIndexReader`, `BuiltinToolExecutor`, `PatchPolicyToolExecutor`

## Cross-Context Event Flow

Contexts communicate exclusively through domain events on the shared event bus.
No bounded context imports another context's `domain/` package directly.

```text
execution  --WorkflowCompleted-->   dataset    (ACL: ExecutionToDatasetTranslator)
training   --ArtifactProduced-->    evaluation
evaluation --ThresholdMet/Breached--> promotion
```

The `InProcessEventBus` and handler subscriptions are wired in
`execution/infrastructure/composition.py` via `build_event_pipeline()`.

## Interfaces (`interfaces/`)

Cross-cutting entry points. Import from bounded-context `application/` packages
and reach infrastructure only through the `interfaces.composition` facade.

- **CLI** — `micro-agent` commands: `task`, `loop`, `dataset`, `train`, `eval`,
  `promote`, `repo`, `index`
- **MCP server** — tool-call server exposing the model-driven tool loop and
  repository operations

## Dependency Direction

```text
interfaces  -> application -> domain
agents      -> application -> domain
infrastructure -> application/domain (via ports)
domain      -> shared kernel only
```

Cross-context rule: **contexts never import each other's `domain/` packages.**
Events pass through the shared event bus; data crossing boundaries is translated
by an ACL in the receiving context's `infrastructure/`.

## Dependency Injection

Runtime assembly happens at the edges. Application services accept explicit
dependencies through constructors (repository, event bus, ports). Domain
aggregates are plain objects with no external dependencies.

Concrete adapter wiring for CLI and MCP commands goes through
`interfaces.composition`, which delegates to per-context
`<context>/infrastructure/composition.py` modules. The facade defines an
explicit public `__all__`; composition tests assert export identity.

## Retrieval

Retrieval is a first-class platform capability. The local implementation uses a
lexical index (`repo.semantic_search`) backed by
`.micro_model_agent/index/lexical-index.json`. Run `micro-agent index` to build
or refresh the index before longer agent sessions. The `LocalSemanticRetriever`
adapts this to the `SemanticRetriever` application port; future vector or hybrid
stores can be swapped in behind that port.

## Trace Capture

Trace capture is automatic orchestration behavior. The model never records its
own traces. `WorkflowExecution` aggregates produce `WorkflowTrace` snapshots
containing goals, tool calls, tool results, generated outputs, verification
results, and outcomes. Traces are persisted by `JsonlWorkflowRepository` and
`JsonlTraceStore`.

## Fine-Tuning Data

Fine-tuning is a core goal. The current pipeline:

```text
committed synthetic templates
  -> generated JSONL dataset
  -> dataset validation
  -> optional curated trace examples merged into a mixed dataset
  -> SFT chat JSONL export
  -> dry-run metadata or local HF/PEFT LoRA adapter training
  -> held-out behavioral synthetic evaluation
```

See [fine-tuning-data-plan.md](fine-tuning-data-plan.md) and
[training-pipeline.md](training-pipeline.md).

## Security

MCP and CLI tools must avoid arbitrary shell execution and unrestricted writes.
Dangerous operations require explicit allowlists, dry-run support, patch
previews, and approval workflows.

## Planning

The implementation roadmap, milestones, tool plan, testing strategy, and MVP
acceptance criteria live in [project-plan.md](project-plan.md).

Contributor-facing boundary guidance lives in
[../CONTRIBUTING.md](../CONTRIBUTING.md).
