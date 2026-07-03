# Classical DDD Migration Plan

This document describes how to migrate micro-model-agent to classical DDD.

The key structural inversion: **Bounded Contexts become the top-level packages.**
Clean Architecture layers (`domain/`, `application/`, `infrastructure/`) live
*inside* each context. The domain is no longer a single flat package — it is
owned by the context it belongs to.

---

## Organizing Principle

**Current (layer-first):**
```
micro_model_agent/
  domain/           ← single shared layer
  application/      ← single shared layer
  infrastructure/   ← single shared layer
  interfaces/
  agents/
```

**Target (bounded-context-first):**
```
micro_model_agent/
  shared/                  ← Shared Kernel (cross-context base types only)
    domain/
  execution/               ← Bounded Context: running agent workflows & tool loops
    domain/
    application/
    infrastructure/
  dataset/                 ← Bounded Context: training dataset lifecycle
    domain/
    application/
    infrastructure/
  training/                ← Bounded Context: fine-tuning job lifecycle
    domain/
    application/
    infrastructure/
  evaluation/              ← Bounded Context: model evaluation & scoring
    domain/
    application/
    infrastructure/
  promotion/               ← Bounded Context: model promotion & registry
    domain/
    application/
    infrastructure/
  repository_ops/          ← Supporting: source code repo retrieval
    domain/
    infrastructure/
  interfaces/              ← Top-level entry points (CLI, MCP — cross-cutting)
  agents/                  ← Reference agent implementations (unchanged role)
```

---

## Current State vs Classical DDD

| Concept | Current state | Target |
|---|---|---|
| Package organization | Layer-first: `domain/`, `application/`, `infrastructure/` | Context-first: each bounded context owns its own layers |
| Domain objects | Flat frozen dataclasses — no behavior | Aggregates enforce invariants; Entities mutate via methods |
| Entity vs Value Object | Not distinguished; everything is a frozen dataclass | Explicit: Entities have identity and lifecycle; VOs are immutable and equal by value |
| Aggregates | None | Aggregate Roots with guarded state transitions |
| Domain Events | None | Events raised by Aggregates; published through an event bus |
| Repositories | Application-layer `Protocol`s (service-interface style) | Collection-semantics DDD Repository per Aggregate Root, defined inside the context's `domain/` |
| Domain Services | Business logic in application layer | Stateless domain services inside the context's `domain/` |
| Bounded Contexts | Implicit groupings under one flat `domain/` | Explicit top-level packages with strict no-cross-import rule |
| Context Map | Implicit | Explicit; cross-context communication via events or ACLs only |
| Anti-Corruption Layers | None | Translators in each context's `infrastructure/` |

---

## Context Map

```
+-------------------------------------------------------------------+
| Core Domains                                                      |
|                                                                   |
|  +-----------+    +---------+    +----------+    +-----------+   |
|  | execution |--->| dataset |--->| training |--->| promotion |   |
|  +-----------+    +---------+    +----------+    +-----------+   |
|        |                               |                          |
|        +-------------------------------+                          |
|                                        v                          |
|                                  +------------+                  |
|                                  | evaluation |                  |
|                                  +------------+                  |
|                                        |                          |
|                                        +-------------> promotion  |
+-------------------------------------------------------------------+

Supporting: shared/, repository_ops/
Cross-cutting: interfaces/ (CLI, MCP)
```

### Integration patterns between contexts

| Upstream | Downstream | Pattern |
|---|---|---|
| `execution` | `dataset` | Published Language — `WorkflowCompleted` event; ACL translates to `DatasetExample` |
| `dataset` | `training` | Open Host Service — training reads dataset via its own port; no direct domain import |
| `training` | `evaluation` | Published Language — `ArtifactProduced` event carries artifact ID |
| `evaluation` | `promotion` | Customer/Supplier — `ThresholdMet` / `ThresholdBreached` events; promotion ACL translates |
| `execution` | `repository_ops` | Conformist — tool loop consumes retrieval results as-is |

Cross-context communication rule: **contexts never import each other's `domain/` packages directly.**
Events pass through the shared event bus; data crossing boundaries is translated by an ACL.

---

## Target Package Structure (full)

```
micro_model_agent/
|
+-- shared/
|   +-- domain/
|       +-- __init__.py
|       +-- domain_event.py       # DomainEvent base
|       +-- entity.py             # Entity base (UUID identity + event list)
|       +-- event_bus.py          # EventBus Protocol
|       +-- exceptions.py         # DomainException base
|
+-- execution/                    # Bounded Context
|   +-- domain/
|   |   +-- aggregate.py          # WorkflowExecution (aggregate root)
|   |   +-- entities.py           # WorkflowStep
|   |   +-- value_objects.py      # Goal, ToolCall, ToolResult, AgentProfile, ModelProfile, ToolDefinition, WorkflowStatus
|   |   +-- events.py             # WorkflowStarted, StepAdded, WorkflowCompleted, WorkflowFailed
|   |   +-- repository.py         # WorkflowRepository Protocol
|   |   +-- services.py           # WorkflowEvaluationService
|   |   +-- exceptions.py         # InvalidTransitionError
|   +-- application/
|   |   +-- commands.py           # RunAgentCommand, RunToolLoopCommand
|   |   +-- results.py            # RunAgentResult, RunToolLoopResult
|   |   +-- workflows.py          # RunAgentWorkflow, RunToolLoopWorkflow
|   |   +-- ports.py              # CodingWorkflowRunner, ToolLoopRunner, ModelProvider ports
|   +-- infrastructure/
|       +-- persistence.py        # JsonlWorkflowRepository
|       +-- models.py             # OllamaModelProvider, StaticModelProvider, etc.
|       +-- agents.py             # CodingAgent, ToolLoopAgent adapter wiring
|       +-- composition.py        # Factory / wiring for execution context
|
+-- dataset/                      # Bounded Context
|   +-- domain/
|   |   +-- aggregate.py          # Dataset (aggregate root)
|   |   +-- entities.py           # DatasetExample
|   |   +-- value_objects.py      # DatasetLabel, DatasetKind, DatasetSplit, OutcomeLabel, QualityLabel, FailureMode
|   |   +-- events.py             # ExampleAdded, ExampleLabelled, DatasetExported, DatasetMerged
|   |   +-- repository.py         # DatasetRepository Protocol
|   |   +-- services.py           # DatasetValidationService, DatasetMergeService, ExampleRelabelService
|   |   +-- exceptions.py         # DuplicateExampleError, InvalidLabelError
|   +-- application/
|   |   +-- commands.py           # SynthesizeDatasetCommand, ExportDatasetCommand, MergeDatasetCommand, RelabelCommand
|   |   +-- results.py            # DatasetOperationResult
|   |   +-- workflows.py          # RunDatasetSynthesisWorkflow, RunDatasetExportWorkflow, etc.
|   |   +-- ports.py              # SyntheticGenerator, DatasetExporter, TraceExporter ports
|   |   +-- event_handlers.py     # OnWorkflowCompleted -> build DatasetExample (ACL from execution)
|   +-- infrastructure/
|       +-- persistence.py        # JsonlDatasetRepository
|       +-- generator.py          # SyntheticTemplateGenerator
|       +-- acl.py                # ExecutionToDatasetTranslator (WorkflowCompleted -> DatasetExample)
|       +-- composition.py
|
+-- training/                     # Bounded Context
|   +-- domain/
|   |   +-- aggregate.py          # TrainingJob (aggregate root)
|   |   +-- entities.py           # TrainingRun, ModelArtifact
|   |   +-- value_objects.py      # TrainingConfig, TrainingRunStatus, ModelArtifactKind
|   |   +-- events.py             # TrainingJobCreated, TrainingJobStarted, TrainingJobCompleted, TrainingJobFailed, ArtifactProduced
|   |   +-- repository.py         # TrainingJobRepository Protocol
|   |   +-- services.py           # TrainingConfigValidationService
|   |   +-- exceptions.py         # InvalidConfigError, TrainingAlreadyStartedError
|   +-- application/
|   |   +-- commands.py           # RunSyntheticTrainingCommand
|   |   +-- results.py            # TrainingResult
|   |   +-- workflows.py          # RunSyntheticTrainingWorkflow
|   |   +-- ports.py              # TrainingRunner, ArtifactStore ports
|   +-- infrastructure/
|       +-- persistence.py        # JsonlTrainingJobRepository
|       +-- runners.py            # LocalFineTuningRunner, FakeTrainingRunner
|       +-- sft_export.py         # SFT JSONL rendering
|       +-- composition.py
|
+-- evaluation/                   # Bounded Context
|   +-- domain/
|   |   +-- aggregate.py          # EvaluationReport (aggregate root)
|   |   +-- entities.py           # EvaluationResult
|   |   +-- value_objects.py      # EvaluationScore, EvaluationThreshold, RubricCategory
|   |   +-- events.py             # EvaluationCompleted, ThresholdMet, ThresholdBreached
|   |   +-- repository.py         # EvaluationReportRepository Protocol
|   |   +-- services.py           # ScoringService, RubricEvaluationService
|   |   +-- exceptions.py         # ScoreOutOfRangeError
|   +-- application/
|   |   +-- commands.py           # RunSyntheticEvaluationCommand, RunTraceEvaluationCommand, RunWorkspaceEvaluationCommand
|   |   +-- results.py            # EvaluationOperationResult
|   |   +-- workflows.py          # RunSyntheticEvaluationWorkflow, RunTraceEvaluationWorkflow, RunWorkspaceStagedEvaluationWorkflow, RunEvaluationComparisonWorkflow
|   |   +-- ports.py              # EvaluationSuite, ModelBehaviorEvaluationSuite ports
|   |   +-- event_handlers.py     # OnArtifactProduced -> trigger evaluation
|   +-- infrastructure/
|       +-- persistence.py        # JsonlEvaluationReportRepository
|       +-- suites.py             # SyntheticEvaluationSuite, TraceBehaviorEvaluationSuite, WorkspaceStagedEvaluationSuite
|       +-- composition.py
|
+-- promotion/                    # Bounded Context
|   +-- domain/
|   |   +-- aggregate.py          # ModelRegistry (aggregate root)
|   |   +-- entities.py           # PromotedModel
|   |   +-- value_objects.py      # PromotionCriteria, PromotionGateDecision, OllamaPackageSpec
|   |   +-- events.py             # PromotionGatePassed, PromotionGateFailed, ModelPromoted, ModelPackaged
|   |   +-- repository.py         # ModelRegistryRepository Protocol
|   |   +-- services.py           # PromotionGateService
|   |   +-- exceptions.py         # GateThresholdNotMetError
|   +-- application/
|   |   +-- commands.py           # RunPromotionGateCommand, RunPromotionRecordCommand, RunPromotionSelectCommand, RunPromotionPackageCommand
|   |   +-- results.py            # PromotionOperationResult
|   |   +-- workflows.py          # RunPromotionGateWorkflow, RunPromotionRecordWorkflow, RunPromotionListWorkflow, RunPromotionSelectWorkflow, RunPromotionPackageOllamaWorkflow
|   |   +-- ports.py              # PromotedAdapterPackager, RepositoryModelConfigurationWriter ports
|   |   +-- event_handlers.py     # OnThresholdMet -> open gate; OnThresholdBreached -> close gate
|   +-- infrastructure/
|       +-- persistence.py        # JsonlModelRegistryRepository
|       +-- packager.py           # LocalOllamaAdapterPackager
|       +-- acl.py                # EvaluationToPromotionTranslator
|       +-- composition.py
|
+-- repository_ops/               # Supporting Context (source code retrieval)
|   +-- domain/
|   |   +-- value_objects.py      # RepositoryProfile, RetrievalQuery, RetrievalResult, SemanticSearchResult
|   +-- infrastructure/
|       +-- retrieval.py          # LocalSemanticRetriever, LocalLexicalIndexReader/Writer
|       +-- tools.py              # RepoSearchTool, RepoReadTool, RepoWritePatchTool, RepoSemanticSearchTool, GitDiffTool, TestRunTool
|       +-- composition.py
|
+-- interfaces/                   # Cross-cutting (CLI + MCP — unchanged role)
|   +-- cli/
|   +-- mcp/
|
+-- agents/                       # Reference implementations (unchanged role)
    +-- coding_agent.py
    +-- tool_loop_agent.py
```

---

## Phase 1 — Shared Kernel

**Goal:** Establish base types that every context imports. Keep this package
small — it must never import from any bounded context.

### `DomainEvent` base

```python
# shared/domain/domain_event.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Base for all domain events. Immutable; carry only what listeners need."""
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### `Entity` base

```python
# shared/domain/entity.py
from uuid import UUID, uuid4
from .domain_event import DomainEvent

class Entity:
    """Equality by identity; owns a pending-events list."""
    def __init__(self, id: UUID | None = None) -> None:
        self.id: UUID = id or uuid4()
        self._events: list[DomainEvent] = []

    def pull_events(self) -> list[DomainEvent]:
        events, self._events = self._events, []
        return events

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
```

### `EventBus` Protocol

```python
# shared/domain/event_bus.py
class EventBus(Protocol):
    async def publish(self, event: DomainEvent) -> None: ...
    async def publish_all(self, events: list[DomainEvent]) -> None: ...
```

### `DomainException` base

```python
# shared/domain/exceptions.py
class DomainException(Exception):
    """Root for all domain rule violations."""
```

**Create:** `shared/domain/{__init__,domain_event,entity,event_bus,exceptions}.py`

---

## Phase 2 — Execution Context

**Goal:** Move `WorkflowTrace` + `WorkflowStep` into `execution/domain/` as a
proper Aggregate with guarded transitions and domain events. Application
workflows and infrastructure adapters follow into `execution/application/` and
`execution/infrastructure/`.

### Domain layer

**Value Objects** — existing shapes preserved, add invariants where useful:

```python
# execution/domain/value_objects.py
@dataclass(frozen=True, slots=True)
class Goal:
    text: str
    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Goal text must not be blank")

# ToolCall, ToolResult, AgentProfile, ModelProfile, ToolDefinition — moved here unchanged
# WorkflowStatus StrEnum — moved here
```

**Entity** (`execution/domain/entities.py`):

```python
class WorkflowStep(Entity):
    def __init__(self, name: str, id: UUID | None = None) -> None:
        super().__init__(id)
        self.name = name
        self.status = WorkflowStatus.PENDING
        self.tool_call: ToolCall | None = None
        self.tool_result: ToolResult | None = None
        self.output: dict[str, Any] = {}
```

**Aggregate Root** (`execution/domain/aggregate.py`):

```python
class WorkflowExecution(Entity):
    def __init__(self, goal: Goal, id: UUID | None = None) -> None:
        super().__init__(id)
        self.goal = goal
        self.status = WorkflowStatus.PENDING
        self.steps: list[WorkflowStep] = []
        self.final_output: dict[str, Any] = {}
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def start(self) -> None:
        if self.status != WorkflowStatus.PENDING:
            raise InvalidTransitionError(f"Cannot start a {self.status} workflow")
        self.status = WorkflowStatus.RUNNING
        self._events.append(WorkflowStarted(execution_id=self.id, goal=self.goal.text))

    def add_step(self, step: WorkflowStep) -> None:
        if self.status != WorkflowStatus.RUNNING:
            raise InvalidTransitionError("Steps can only be added to a running workflow")
        self.steps.append(step)
        self._events.append(StepAdded(execution_id=self.id, step_id=step.id, step_name=step.name))

    def complete(self, final_output: dict[str, Any]) -> None:
        if self.status != WorkflowStatus.RUNNING:
            raise InvalidTransitionError(f"Cannot complete a {self.status} workflow")
        self.final_output = final_output
        self.status = WorkflowStatus.SUCCEEDED
        self.updated_at = datetime.now(UTC)
        self._events.append(WorkflowCompleted(execution_id=self.id, output=final_output))

    def fail(self, reason: str) -> None:
        if self.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PENDING):
            raise InvalidTransitionError(f"Cannot fail a {self.status} workflow")
        self.status = WorkflowStatus.FAILED
        self.updated_at = datetime.now(UTC)
        self._events.append(WorkflowFailed(execution_id=self.id, reason=reason))

    def to_snapshot(self) -> WorkflowTrace:
        """Frozen read-model snapshot for backward-compatible callers."""
        ...
```

**Domain Events** (`execution/domain/events.py`):

```python
@dataclass(frozen=True, slots=True)
class WorkflowStarted(DomainEvent):
    execution_id: UUID
    goal: str

@dataclass(frozen=True, slots=True)
class StepAdded(DomainEvent):
    execution_id: UUID
    step_id: UUID
    step_name: str

@dataclass(frozen=True, slots=True)
class WorkflowCompleted(DomainEvent):
    execution_id: UUID
    output: dict[str, Any]

@dataclass(frozen=True, slots=True)
class WorkflowFailed(DomainEvent):
    execution_id: UUID
    reason: str
```

**Repository Protocol** (`execution/domain/repository.py`):

```python
class WorkflowRepository(Protocol):
    async def add(self, execution: WorkflowExecution) -> None: ...
    async def get(self, id: UUID) -> WorkflowExecution | None: ...
    async def find_by_status(self, status: WorkflowStatus) -> list[WorkflowExecution]: ...
    async def save(self, execution: WorkflowExecution) -> None: ...
```

**Domain Service** (`execution/domain/services.py`):
Move `DefaultWorkflowEvaluator` here as `WorkflowEvaluationService`. It
operates on `WorkflowExecution` and returns a local `EvaluationScore` VO.

### Application layer (`execution/application/workflows.py`)

```python
class RunAgentWorkflow:
    def __init__(self, runner: CodingWorkflowRunner, repo: WorkflowRepository, bus: EventBus) -> None: ...

    async def run(self, command: RunAgentCommand) -> RunAgentResult:
        execution = WorkflowExecution(goal=Goal(command.goal))
        execution.start()
        await self.runner.run(execution)           # runner mutates aggregate
        await self.repo.save(execution)
        await self.bus.publish_all(execution.pull_events())
        return RunAgentResult.from_execution(execution)
```

### Infrastructure layer

- `execution/infrastructure/persistence.py` — `JsonlWorkflowRepository` implementing `WorkflowRepository` (was `persistence.JsonlTraceStore`)
- `execution/infrastructure/composition.py` — wires repository, model providers, agents, event bus

**Backward compat:** `WorkflowTrace` becomes a frozen read-model snapshot
returned by `WorkflowExecution.to_snapshot()`. Old importers continue to
receive it; they just receive a snapshot, not the live aggregate.

**Remove / replace:**
- `domain/contracts.py` — split across contexts
- `application/agent/workflows.py` → `execution/application/workflows.py`
- `infrastructure/persistence/` (trace parts) → `execution/infrastructure/persistence.py`

---

## Phase 3 — Dataset Context

**Goal:** Introduce `Dataset` aggregate root owning `DatasetExample` Entities.
Move synthesis, export, merge, and relabel use cases into
`dataset/application/workflows.py`. Validation moves into `dataset/domain/services.py`.

### Aggregate Root (`dataset/domain/aggregate.py`)

```python
class Dataset(Entity):
    def __init__(self, name: str, id: UUID | None = None) -> None:
        super().__init__(id)
        self.name = name
        self._examples: dict[UUID, DatasetExample] = {}

    def add_example(self, example: DatasetExample) -> None:
        if example.id in self._examples:
            raise DuplicateExampleError(example.id)
        self._examples[example.id] = example
        self._events.append(ExampleAdded(dataset_id=self.id, example_id=example.id))

    def relabel(self, example_id: UUID, new_label: DatasetLabel) -> None:
        if example_id not in self._examples:
            raise ExampleNotFoundError(example_id)
        self._examples[example_id] = self._examples[example_id].with_label(new_label)
        self._events.append(ExampleLabelled(dataset_id=self.id, example_id=example_id))

    @property
    def examples(self) -> tuple[DatasetExample, ...]:
        return tuple(self._examples.values())
```

### Application event handler (ACL at the execution boundary)

```python
# dataset/application/event_handlers.py
class OnWorkflowCompleted:
    def __init__(self, translator: ExecutionToDatasetTranslator, repo: DatasetRepository) -> None: ...

    async def handle(self, event: WorkflowCompleted) -> None:
        example = self.translator.translate(event)
        dataset = await self.repo.find_by_name("default") or Dataset(name="default")
        dataset.add_example(example)
        await self.repo.save(dataset)
```

### Infrastructure ACL (`dataset/infrastructure/acl.py`)

```python
class ExecutionToDatasetTranslator:
    """Translates WorkflowCompleted event into a DatasetExample without importing execution domain."""
    def translate(self, event: WorkflowCompleted) -> DatasetExample: ...
```

**Remove / replace:**
- `domain/datasets.py` → `dataset/domain/`
- `application/datasets/workflows.py` → `dataset/application/workflows.py`
- `infrastructure/datasets/` → `dataset/infrastructure/`

---

## Phase 4 — Training Context

**Goal:** Introduce `TrainingJob` aggregate root. Training workflows move to
`training/application/`.

### Aggregate Root (`training/domain/aggregate.py`)

```python
class TrainingJob(Entity):
    def __init__(self, config: TrainingConfig, id: UUID | None = None) -> None:
        super().__init__(id)
        self.config = config
        self._run: TrainingRun | None = None
        self._artifacts: list[ModelArtifact] = []
        self._events.append(TrainingJobCreated(job_id=self.id))

    def start(self, run: TrainingRun) -> None:
        if self._run is not None:
            raise TrainingAlreadyStartedError(self.id)
        self._run = run
        self._events.append(TrainingJobStarted(job_id=self.id, run_id=run.id))

    def complete(self, run: TrainingRun, artifact: ModelArtifact) -> None:
        if self._run is None:
            raise InvalidTransitionError("Job not started")
        self._run = run
        self._artifacts.append(artifact)
        self._events.append(TrainingJobCompleted(job_id=self.id, artifact_id=artifact.id))
        self._events.append(ArtifactProduced(job_id=self.id, artifact_id=artifact.id, kind=artifact.kind))

    def fail(self, reason: str) -> None:
        self._events.append(TrainingJobFailed(job_id=self.id, reason=reason))
```

**Domain Service:** `TrainingConfigValidationService` validates `TrainingConfig`
invariants before a job is created.

**Remove / replace:**
- `domain/training.py` → `training/domain/`
- `application/training/workflows.py` → `training/application/workflows.py`
- `infrastructure/training/` → `training/infrastructure/`

---

## Phase 5 — Evaluation Context

**Goal:** Introduce `EvaluationReport` aggregate. Scoring rubrics move from
`application/evaluation_rubrics/` into `evaluation/domain/services.py`. The
`ThresholdMet` and `ThresholdBreached` events are consumed by the promotion
context via its ACL.

### Aggregate Root (`evaluation/domain/aggregate.py`)

```python
class EvaluationReport(Entity):
    def __init__(self, run_id: str, threshold: EvaluationThreshold, id: UUID | None = None) -> None:
        super().__init__(id)
        self.run_id = run_id
        self.threshold = threshold
        self._results: list[EvaluationResult] = []
        self.summary_score: float | None = None

    def add_result(self, result: EvaluationResult) -> None:
        self._results.append(result)

    def finalize(self, summary_score: float) -> None:
        self.summary_score = summary_score
        self._events.append(EvaluationCompleted(report_id=self.id, score=summary_score))
        if summary_score >= self.threshold.minimum_score:
            self._events.append(ThresholdMet(report_id=self.id, score=summary_score))
        else:
            self._events.append(ThresholdBreached(
                report_id=self.id, score=summary_score,
                threshold=self.threshold.minimum_score,
            ))
```

**Value Objects:**

```python
@dataclass(frozen=True, slots=True)
class EvaluationThreshold:
    minimum_score: float
    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {self.minimum_score}")

@dataclass(frozen=True, slots=True)
class EvaluationScore:
    value: float
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
```

**Remove / replace:**
- `application/evaluation_rubrics/` → `evaluation/domain/services.py`
- `application/evaluation/` → `evaluation/application/`
- `infrastructure/evaluation/` → `evaluation/infrastructure/`

---

## Phase 6 — Promotion Context

**Goal:** Introduce `ModelRegistry` aggregate. Gate policy moves into
`promotion/domain/services.py`. ACL in `promotion/infrastructure/acl.py`
translates evaluation events into promotion-local VOs.

### Aggregate Root (`promotion/domain/aggregate.py`)

```python
class ModelRegistry(Entity):
    def __init__(self, id: UUID | None = None) -> None:
        super().__init__(id)
        self._models: dict[UUID, PromotedModel] = {}

    def record_promotion(self, model: PromotedModel) -> None:
        self._models[model.id] = model
        self._events.append(ModelPromoted(registry_id=self.id, model_id=model.id))

    def list_models(self) -> tuple[PromotedModel, ...]:
        return tuple(self._models.values())

    def select(self, artifact_id: UUID) -> PromotedModel | None:
        return next((m for m in self._models.values() if m.artifact_id == artifact_id), None)
```

### Domain Service (`promotion/domain/services.py`)

```python
class PromotionGateService:
    def evaluate(self, criteria: PromotionCriteria, scores: tuple[EvaluationScore, ...]) -> PromotionGateDecision:
        passing = all(s.value >= criteria.minimum_score for s in scores)
        return PromotionGateDecision(passed=passing, scores=scores, criteria=criteria)
```

### Infrastructure ACL (`promotion/infrastructure/acl.py`)

```python
class EvaluationToPromotionTranslator:
    """Translates ThresholdMet/Breached events into promotion-local EvaluationScore VOs."""
    def translate(self, event: ThresholdMet | ThresholdBreached) -> EvaluationScore:
        return EvaluationScore(
            value=event.score,
            passed=isinstance(event, ThresholdMet),
            summary=f"Score {event.score:.2f}",
        )
```

**Remove / replace:**
- `application/promotion/workflows.py` → `promotion/application/workflows.py`
- `infrastructure/promotion/` → `promotion/infrastructure/`

---

## Phase 7 — Repository Implementations

Replace current service-interface ports with DDD Repository implementations.

| Current adapter | New adapter | Implements |
|---|---|---|
| `persistence.JsonlTraceStore` | `execution/infrastructure/persistence.JsonlWorkflowRepository` | `execution.domain.repository.WorkflowRepository` |
| `persistence.JsonlDatasetExampleStore` | `dataset/infrastructure/persistence.JsonlDatasetRepository` | `dataset.domain.repository.DatasetRepository` |
| `training.JsonTrainingArtifactStore` | `training/infrastructure/persistence.JsonlTrainingJobRepository` | `training.domain.repository.TrainingJobRepository` |
| `evaluation.LocalEvaluationResultWriter/Reader` | `evaluation/infrastructure/persistence.JsonlEvaluationReportRepository` | `evaluation.domain.repository.EvaluationReportRepository` |
| `promotion.LocalPromotionGateStore` | `promotion/infrastructure/persistence.JsonlModelRegistryRepository` | `promotion.domain.repository.ModelRegistryRepository` |

Repository semantics — uniform across all contexts:

```python
async def add(self, aggregate: T) -> None: ...   # first persist
async def save(self, aggregate: T) -> None: ...  # update existing
async def get(self, id: UUID) -> T | None: ...   # load by identity
async def find_by_*(self, ...) -> list[T]: ...   # simple predicate queries
```

Complex multi-aggregate reads stay as **Query Services** in the application
layer, returning DTOs rather than live aggregates.

---

## Phase 8 — Application Layer Thinning

Each context's `application/workflows.py` becomes a pure orchestrator:

1. Build or load the aggregate from the repository.
2. Call domain methods (events accumulate inside the aggregate).
3. Persist via repository.
4. Pull and publish events via `EventBus`.
5. Return a result DTO.

### Cross-context event handler wiring

| Event | Handler | Registered in |
|---|---|---|
| `WorkflowCompleted` (execution) | `dataset.application.event_handlers.OnWorkflowCompleted` | `dataset/infrastructure/composition.py` |
| `ArtifactProduced` (training) | `evaluation.application.event_handlers.OnArtifactProduced` | `evaluation/infrastructure/composition.py` |
| `ThresholdMet` / `ThresholdBreached` (evaluation) | `promotion.application.event_handlers.OnThresholdEvent` | `promotion/infrastructure/composition.py` |

---

## Phase 9 — Domain Services Extraction

| Current location | New location |
|---|---|
| `application.agent.workflows.DefaultWorkflowEvaluator` | `execution/domain/services.WorkflowEvaluationService` |
| `application.agent.workflows.TraceDatasetBuilder` | `dataset/infrastructure/acl.ExecutionToDatasetTranslator` |
| `application.datasets.workflows` (merge policy) | `dataset/domain/services.DatasetMergeService` |
| `application.datasets.workflows` (relabel policy) | `dataset/domain/services.ExampleRelabelService` |
| `application.evaluation_rubrics.*` | `evaluation/domain/services.ScoringService` / `RubricEvaluationService` |
| `application.promotion.workflows` (gate policy) | `promotion/domain/services.PromotionGateService` |

---

## Phase 10 — `interfaces/` and `agents/` Updates

These are cross-cutting and keep their top-level role. Only import paths change:

- CLI commands import from `<context>/application/` not `application/<context>/`
- MCP handlers import from `<context>/infrastructure/composition` not `infrastructure/composition`
- `agents/` imports execution ports from `execution/application/ports`

Compatibility facades in the old locations re-export from new paths during
transition, then are removed.

---

## Phase 11 — Legacy Cleanup

After all callers are migrated:

- Remove `domain/contracts.py`, `domain/datasets.py`, `domain/training.py`
- Remove `application/agent/`, `application/datasets/`, `application/evaluation/`, `application/promotion/`, `application/training/`
- Remove `infrastructure/persistence/`, `infrastructure/datasets/`, `infrastructure/evaluation/`, `infrastructure/promotion/`, `infrastructure/training/`
- Replace `infrastructure/composition.py` with per-context `composition.py` files

---

## Migration Sequence

```
Phase 1  (shared kernel)
    |
    +---> Phase 2 (execution)
    |         |
    |         v
    +---> Phase 3 (dataset)        <- consumes WorkflowCompleted event
    |
    +---> Phase 4 (training)
    |         |
    |         v
    +---> Phase 5 (evaluation)     <- consumes ArtifactProduced event
    |         |
    |         v
    +---> Phase 6 (promotion)      <- consumes ThresholdMet/Breached events
    |
Phases 7-9 done per-context alongside phases 2-6
Phase 10 (interfaces + agents import updates)
Phase 11 (facade + old-path removal)
```

Each context migration (phases 2-6):
1. Create new context package (`<context>/domain/`, `application/`, `infrastructure/`).
2. Move domain objects; add aggregate, events, invariants.
3. Move application workflows; thin to pure orchestration.
4. Move infrastructure adapters; rename to repository semantics.
5. Run existing tests; add domain unit tests for invariants and events.
6. Remove old paths once no callers remain.

---

## Test Strategy

| Layer | What to test |
|---|---|
| `<context>/domain/` | Invariant violations raise exceptions; transitions emit correct events |
| `<context>/domain/services.py` | Pure function behavior; zero infrastructure dependencies |
| `<context>/infrastructure/persistence.py` | Integration tests against JSONL files |
| `<context>/application/workflows.py` | Mock repository + event bus; assert events published and repo methods called |
| `<context>/infrastructure/acl.py` | Translation correctness across context boundaries |
| `test_architecture_boundaries.py` | Extend: no cross-context `domain/` imports; `shared/` imports nothing from any context |

---

## Risk Areas

| Risk | Mitigation |
|---|---|
| `WorkflowTrace` imported throughout the codebase | Keep as frozen read-model snapshot via `WorkflowExecution.to_snapshot()`; old importers receive a snapshot not the live aggregate |
| Application port contracts widely imported | Per-context `application/ports.py` re-exports during transition; remove after full migration |
| `infrastructure/composition.py` is the current wiring hub | Migrate one context at a time; keep old composition as a delegating facade until all contexts have their own |
| `dict[str, Any]` payloads in domain objects | Convert to typed VOs incrementally; don't block migration on full VO conversion |
| Event bus adds async complexity | Start synchronous in-process; upgrade to async queue only if cross-process is needed |
| Architecture boundary tests fail mid-migration | Add new boundary rules incrementally; use `xfail` markers during active migration of each context |
