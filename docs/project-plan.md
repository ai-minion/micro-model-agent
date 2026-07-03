# Project Plan

## What We Are Building

micro-model-agent is a Python framework for building specialized agents powered by
small local models. The first target is a CLI coding agent backed by an
Qwen-Coder 7B-class model running locally on developer hardware such as an RTX
3090. Ollama is the initial local inference interface, not a remote hosting
dependency.

The project is not a general chatbot. It is an orchestration framework where the
agent operates inside a constrained workflow:

```text
User task
  -> task routing
  -> structured retrieval
  -> typed tool calls
  -> patch generation
  -> verification
  -> trace capture
  -> structured result
```

The model should execute workflow steps. The platform owns retrieval, tool
execution, verification, trace persistence, evaluation, dataset generation, and
future model promotion.

## Model And Fine-Tuning Direction

The initial model target is Qwen-Coder 7B running locally, with Ollama used for
inference at first. The exact Ollama model tag and training base model stay
configurable, but the project should optimize the first workflow and tool schema
for that class of model and for local RTX 3090-class hardware.

Fine-tuning is a core project goal. V1 should include a minimal synthetic-data
training pipeline, not just data capture. Every CLI task should be able to
produce trace and dataset records, and the project should also be able to
generate synthetic seed data, validate it, run a local training job, and evaluate
the resulting artifact.

The training data should teach the model:

- the micro-model-agent tool schema
- the project documentation
- the local codebase
- safe tool-use behavior
- good and bad workflow outcomes
- verification-driven repair behavior

See [fine-tuning-data-plan.md](fine-tuning-data-plan.md) and
[training-pipeline.md](training-pipeline.md). The immediate priority is
[trained-model-proof-plan.md](trained-model-proof-plan.md): prove that a trained
local adapter beats the base model on held-out tool-use, repair, and workspace
tasks before expanding the framework surface.

## Product Thesis

Small specialized models can perform useful software engineering work when the
system around them provides:

- clear task boundaries
- targeted context retrieval
- typed tools
- safe write paths
- verification loops
- durable traces
- evaluation feedback

This lets expensive frontier models become optional escalation paths rather than
the default execution engine for every request.

## Non-Goals For V1

- No general-purpose chat assistant.
- No arbitrary shell execution through MCP.
- No unrestricted file writes.
- No cloud training orchestration.
- No automatic upload of private code or traces to external training services.
- No model promotion automation before evaluation scaffolding exists.
- No production-grade distributed vector database requirement.
- No attempt to support every agent type before the coding-agent slice works.

## Core Constraints

- Python 3.12 or newer.
- Modern `src/` package layout.
- Strict Domain Driven Design boundaries.
- Domain layer has no framework, file system, network, or MCP dependencies.
- Pydantic models define external contracts, tool payloads, and interface-layer
  validation.
- Application services depend on ports, not concrete infrastructure adapters.
- Infrastructure adapters are replaceable through dependency injection.
- Tests live beside the files they test.

## Architecture Targets

### Domain Layer

Path: `src/micro_model_agent/domain`

Owns framework-independent concepts:

- agent profiles
- model profiles
- tool definitions
- tool calls and results
- workflow traces and steps
- repository profiles
- retrieval queries and results
- evaluation results

Domain objects should stay boring, explicit, and portable. If a concept needs
Pydantic validation for an API boundary, create a separate contract in the
interface, application, or infrastructure layer.

### Application Layer

Path: `src/micro_model_agent/application`

Owns use cases and orchestration:

- `RunAgentWorkflow`
- `TraceDatasetBuilder`
- `DefaultWorkflowEvaluator`
- trace capture and labeling helpers

Application services accept dependencies through constructors. They should be
easy to test with fake model providers, fake tools, fake retrievers, and in-memory
trace stores.

### Infrastructure Layer

Each bounded context owns its adapters under `<context>/infrastructure/`. For example:

- `execution/infrastructure/` — model providers (`OllamaModelProvider`, `TransformersModelProvider`), `JsonlTraceStore`, coding/tool-loop agents
- `repository_ops/infrastructure/` — `BuiltinToolExecutor`, `RepoReadTool`, `RepoWritePatchTool`, `GitDiffTool`, `LocalSemanticRetriever`
- `dataset/infrastructure/` — prompting adapters, SFT export, dataset store
- `training/infrastructure/` — `LocalFineTuningRunner`, packaging helpers
- `evaluation/infrastructure/` — `WorkspaceStagedEvaluationSuite`, `SyntheticBehaviorEvaluationSuite`
- `interfaces/` — CLI, MCP server, public Python API

Infrastructure can use external libraries, file system access, subprocesses
where explicitly allowed, and provider-specific details.

### Interface Layer

Path: `src/micro_model_agent/interfaces`

Owns user-facing entrypoints:

- CLI
- MCP server
- public Python API

Interfaces assemble the application through dependency injection and expose
typed request/response contracts.

### Agents

Path: `src/micro_model_agent/execution/infrastructure/`

Owns reference workflows. The first agent is `CodingAgent`, which coordinates:

- task intake
- context gathering
- semantic retrieval
- repository search/read calls
- patch proposal
- dry-run patch preview
- safe patch application
- verification
- structured final result

## MVP Vertical Slice

The first working slice should prove the full control loop with fake model
support first and Ollama support second.

```text
micro-agent task "Add a small function and test"
  -> build CodingAgent request
  -> retrieve repository context
  -> ask model runner for a patch plan
  -> execute repo.search and repo.read as needed
  -> generate a unified patch
  -> preview patch
  -> apply patch when approved or when non-interactive policy allows
  -> run allowlisted tests
  -> capture WorkflowTrace automatically
  -> store labeled good/bad dataset records
  -> return CodingAgentResult

micro-agent dataset synthesize --count 500
  -> load tool schemas and docs
  -> generate synthetic tool-use and repair examples
  -> validate examples against Pydantic contracts
  -> write train/validation JSONL

micro-agent train synthetic --dry-run
  -> load synthetic dataset
  -> validate training config
  -> run fake or real local training adapter
  -> write training artifact metadata
  -> run held-out synthetic evaluation
```

### MVP Acceptance Criteria

- A user can initialize local metadata with `micro-agent init`.
- A user can index a small repository with `micro-agent index`.
- A user can run a dry-run coding task with `micro-agent task --dry-run`.
- The task path records a `WorkflowTrace` without model-authored trace steps.
- Repository reads and writes are restricted to the configured repository root.
- Patch writing supports preview and dry-run.
- Tests are run only through allowlisted commands.
- Good and bad results can be labeled from the CLI for later fine-tuning.
- Synthetic seed data can be generated for the tool schema and workflow formats.
- Synthetic datasets can be validated before training.
- A local synthetic training command exists with dry-run and fake-runner support.
- Training runs record artifact metadata and evaluation results.
- `micro-agent serve-mcp` exposes the initial MCP tools.
- The end-to-end test uses fake model and fake tool implementations.

## Milestones

### Milestone 0: Bootstrap

Goal: create a clean project foundation.

Deliverables:

- Git repository initialized.
- MIT license.
- `pyproject.toml` with runtime and dev dependencies.
- `.env.example`.
- `src/micro_model_agent` package.
- DDD folders.
- architecture documentation.
- local virtual environment.

Exit criteria:

- Package imports.
- Source compiles.
- Git status shows only intentional uncommitted bootstrap files.

### Milestone 1: Domain And Contracts

Goal: define stable vocabulary before adding adapters.

Deliverables:

- domain contracts for profiles, tools, traces, retrieval, and evaluation.
- application ports for model providers, tools, retrievers, evaluators, and trace
  stores.
- Pydantic request/response contracts for tools and public interfaces.
- colocated tests for contract behavior.

Exit criteria:

- Domain layer imports no infrastructure or interface modules.
- Contract tests pass.
- mypy can check the domain and application ports.

### Milestone 2: Local Repository Tools

Goal: provide safe coding-agent capabilities.

Deliverables:

- `repo.search`
- `repo.read`
- `repo.semantic_search`
- `repo.write_patch`
- `test.run`
- `git.diff`
- path safety utilities.
- command allowlist model.

Exit criteria:

- Tools use typed Pydantic input and output contracts.
- Tools reject paths outside the repository root.
- Patch writes can run in dry-run mode.
- Test execution has timeouts and structured output.

### Milestone 3: Structured Retrieval And Indexing

Goal: make retrieval more than chunk-only RAG.

Deliverables:

- local repository indexer.
- document/source-code metadata capture.
- symbol/import/dependency extraction scaffolding.
- persisted vector or hybrid retrieval implementation.
- retrieval source types for code, docs, ADRs, traces, recipes, errors, and
  domain notes.

Exit criteria:

- `micro-agent index` creates a local index.
- semantic search returns typed `RetrievedItem` records with source metadata.
- retrieval contracts can map cleanly to future Qdrant or Chroma adapters.

### Milestone 4: Workflow Orchestration And Data Capture

Goal: make the coding-agent loop real with fake dependencies and capture
fine-tuning-ready data from every run.

Deliverables:

- `RunAgentWorkflow`.
- `CodingAgent`.
- automatic trace capture.
- outcome labeling.
- good/bad result storage.
- workflow result evaluation.
- fake model provider for tests.
- one end-to-end workflow test.

Exit criteria:

- The end-to-end fake-model test exercises retrieval, tool calls, patch
  generation, verification, and trace storage.
- The model does not persist traces directly.
- Good and bad outcomes can be stored with labels and failure modes.
- The application service can run without Ollama.

### Milestone 5: Local Inference With Ollama And Transformers

Goal: connect the local inference provider without contaminating the domain
layer.

Deliverables:

- `OllamaModelProvider`.
- `TransformersModelProvider`.
- model profile configuration.
- provider abstraction for future vLLM, Hugging Face, OpenAI-compatible APIs,
  LM Studio, and SGLang.

Exit criteria:

- Provider-specific logic stays in infrastructure.
- A local Ollama model can execute a dry-run task.
- Provider failures return structured errors.

### Milestone 6: CLI And MCP Interfaces

Goal: expose the system safely.

Deliverables:

- `micro-agent init`
- `micro-agent index`
- `micro-agent task`
- `micro-agent serve-mcp`
- MCP tools:
  - `micro_agent.run_task`
  - `micro_agent.index_repo`
  - `micro_agent.semantic_search`
  - `micro_agent.get_trace`

Exit criteria:

- CLI and MCP share application services.
- MCP exposes no arbitrary shell tool.
- All write-capable operations support dry-run or explicit approval policy.

### Milestone 7: Synthetic Training Pipeline

Goal: train from validated synthetic data from day one.

Deliverables:

- `TraceStore` interface and local implementation.
- `DatasetExampleStore` interface and local implementation.
- `DatasetBuilder` interface.
- synthetic seed data generator.
- dataset export command.
- dataset validation command.
- `TrainingRunner` interface.
- fake training runner for tests.
- local fine-tuning runner scaffold.
- training artifact store.
- `EvaluationSuite` interface.
- `ModelPromotionPolicy` interface.
- held-out trace evaluation command.
- local promotion gate and registry commands.
- documentation for trace-to-dataset and evaluation workflows.

Exit criteria:

- `micro-agent dataset synthesize` writes JSONL examples.
- `micro-agent dataset validate` rejects invalid examples.
- `micro-agent train synthetic --dry-run` validates configuration and writes a
  dry-run artifact record.
- A fake training runner can complete in tests without a GPU.
- Real Qwen-Coder 7B fine-tuning is available behind an explicit local runner
  and hardware-dependent configuration.
- Traces contain enough structure to support later evaluation and dataset
  generation.
- Synthetic tool-use examples exist for the initial 5-6 tools.
- Promotion requires persisted evaluation reports and writes a local approval
  record before any human changes runtime defaults.

## Tool Plan

### `repo.search`

Capabilities:

- glob filtering
- text search
- symbol search scaffolding

Safety:

- read-only
- repository-root constrained
- result limits

### `repo.read`

Capabilities:

- single file reads
- multiple file reads
- line ranges

Safety:

- repository-root constrained
- binary-file rejection or safe metadata response
- maximum bytes per request

### `repo.semantic_search`

Capabilities:

- source-code retrieval
- documentation retrieval
- architecture docs and ADR retrieval
- workflow trace retrieval
- workflow recipe retrieval
- error history retrieval
- domain-documentation retrieval

Safety:

- read-only
- structured metadata
- bounded result count

### `repo.write_patch`

Capabilities:

- patch preview
- dry-run application
- safe application

Safety:

- no arbitrary file writes
- repository-root constrained
- approval-aware
- patch-only write path

### `test.run`

Capabilities:

- run allowlisted test commands
- enforce timeout
- capture stdout, stderr, exit code, and duration

Safety:

- no arbitrary command input
- explicit allowlist
- bounded execution time

### `git.diff`

Capabilities:

- return working-tree diff
- optional path filtering

Safety:

- read-only
- repository-root constrained

## Testing Plan

Tests live beside the code they test. Use `test_*.py` names so pytest can import
colocated tests without colliding with same-named modules:

```text
contracts.py
test_contracts.py
```

Required coverage:

- domain contract behavior
- application service orchestration
- tool request/response validation
- repository path safety
- patch dry-run behavior
- trace capture behavior
- fake-model coding-agent end-to-end workflow
- CLI smoke tests
- MCP contract smoke tests where practical

## Configuration Plan

Configuration sources, in priority order:

1. explicit CLI or API arguments
2. environment variables
3. project config file
4. sensible local defaults

Initial environment variables:

- `MICRO_MODEL_AGENT_MODEL_PROVIDER`
- `MICRO_MODEL_AGENT_OLLAMA_BASE_URL`
- `MICRO_MODEL_AGENT_DEFAULT_MODEL`
- `MICRO_MODEL_AGENT_TRACE_DIR`
- `MICRO_MODEL_AGENT_INDEX_DIR`
- `MICRO_MODEL_AGENT_DATASET_DIR`
- `MICRO_MODEL_AGENT_TRAINING_DIR`
- `MICRO_MODEL_AGENT_TRAINING_BASE_MODEL`

## Security Plan

Security is part of the architecture, not a later hardening pass.

Rules:

- never expose arbitrary shell execution through MCP.
- never allow unrestricted writes.
- default write-capable operations to dry-run where possible.
- require repository-root path validation for file access.
- prefer patch application over direct writes.
- make approvals explicit in contracts.
- keep command execution allowlisted.
- include trace records for sensitive operations without storing secrets.

## Dependency Injection Strategy

Composition should happen at the edge:

- CLI builds application services from local config.
- MCP server builds the same application services.
- tests build services with fake dependencies.

Application services should receive:

- model provider
- agent runner
- retriever
- tool registry or executor
- trace store
- evaluator
- repository adapter

This keeps runtime choices replaceable without changing domain or application
logic.

## Proposed Initial File Map

```text
src/micro_model_agent/
|-- domain/
|   |-- contracts.py
|   `-- test_contracts.py
|-- application/
|   |-- ports.py
|   |-- workflow.py
|   `-- test_workflow.py
|-- infrastructure/
|   |-- config.py
|   |-- ollama_provider.py
|   |-- local_vector_store.py
|   |-- repository_indexer.py
|   |-- filesystem_repository.py
|   |-- git_adapter.py
|   |-- trace_store.py
|   |-- dataset_store.py
|   |-- synthetic_data.py
|   |-- training_runner.py
|   |-- artifact_store.py
|   `-- tools/
|       |-- contracts.py
|       |-- repo_search.py
|       |-- repo_read.py
|       |-- repo_semantic_search.py
|       |-- repo_write_patch.py
|       |-- test_run.py
|       `-- git_diff.py
|-- interfaces/
|   |-- cli.py
|   |-- mcp_server.py
|   `-- public_api.py
`-- agents/
    |-- coding_agent.py
    `-- test_coding_agent.py
```

## Open Questions

- What is the first real repository we want the coding agent to operate on?
- Should V1 patch approval be interactive CLI-only, config-driven, or both?
- Which Qwen-Coder 7B base model and Ollama inference tag should be the
  recommended defaults for an RTX 3090?
- Should local retrieval start lexical-first, vector-first, or hybrid?
- What is the minimum trace schema needed for future evaluation without
  overdesigning V1?
- What labels are required before a trace is eligible for fine-tuning export?
- How much synthetic data should be generated before real workflow data is
  preferred?
- What is the smallest real model we should use for local training smoke tests?
- Should the first real training backend be plain Hugging Face PEFT, Axolotl,
  Unsloth, or another adapter behind our own `TrainingRunner` port?

## Immediate Next Steps

The initial V1 foundation is implemented: domain contracts, typed repository
tools, trace capture, dataset synthesis/export/curation, local training
metadata, behavioral evaluation, held-out trace evaluation, and manual
promotion records.

1. Expand held-out synthetic and trace-derived evaluation fixtures, especially
   failure cases and patch-repair tasks.
2. Run and document the first real local adapter training proof against the
   configured tool profile.
3. Add tool-profile metadata to datasets, training artifacts, and evaluation
   reports.
4. Add an explicit default-adapter selection command that can read the local
   promotion registry but still requires a human action.
5. Add a vector or hybrid retrieval backend behind the existing retrieval port
   if lexical search stops being enough.
6. Add an Ollama packaging path for promoted PEFT adapters after the direct
   Transformers adapter path is proven.
7. Continue curating real trace examples and keep a strict held-out split before
   merging accepted traces into training.
