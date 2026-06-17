# Synthetic Training Pipeline

## Purpose

micro-model-agent should be trainable from day one, starting with synthetic data.
The first training pipeline should be intentionally small:

```text
tool schemas + docs + codebase facts
  -> synthetic examples
  -> validation
  -> train/validation split
  -> local adapter fine-tuning
  -> evaluation
  -> artifact record
```

The goal is not to solve all model training in V1. The goal is to make training
a normal part of the product loop from the first working CLI.

The current priority is the proof plan in
[trained-model-proof-plan.md](trained-model-proof-plan.md): train a local
adapter, compare it against the base model on held-out examples, and only then
promote it for agent-loop and MCP use.

## Initial Training Target

The first target is a Qwen-Coder 7B-class base model trained locally on developer
hardware such as an RTX 3090. The specific Hugging Face or local model identifier
should be configuration, not a hard-coded domain choice.

Ollama remains the initial local inference interface. Training will likely
produce an adapter or merged model artifact first, then a later packaging step
can make that artifact available to the local Ollama runtime.

## V1 Training Scope

V1 should include:

- synthetic data generation
- dataset validation
- dataset export
- a local supervised fine-tuning command
- a fake or tiny training runner for tests
- a real training runner interface for local Qwen-Coder 7B experiments on an RTX
  3090-class machine
- evaluation on held-out synthetic tasks
- model artifact metadata

V1 should not include:

- cloud training orchestration
- automatic uploads of private code or traces
- production model registry management
- automatic model promotion without evaluation
- reinforcement learning infrastructure

## Pipeline Stages

### 1. Source Material Collection

Inputs:

- tool schemas
- architecture docs
- project plan docs
- codebase index
- workflow recipes
- rejected action examples
- verification and repair examples

The synthetic generator should prefer small, explicit examples over broad coding
tasks at first. We want the model to learn micro-model-agent's operating rules
before we ask it to write large patches.

### 2. Synthetic Example Generation

Initial generated categories:

- choose `repo.search` for finding files and symbols
- choose `repo.read` before editing a file
- choose `repo.semantic_search` for architecture rules and workflow examples
- choose `repo.write_patch` for dry-run patch proposals
- choose `test.run` only from allowlisted commands
- choose `git.diff` for self-review
- reject arbitrary shell execution
- reject unsafe path traversal
- repair invalid tool arguments
- repair failed patches
- respond to test failures with a focused next action

Generated examples should include both positive and negative cases. Negative
cases should usually train correction behavior rather than directly train the
model to produce bad output.

### 3. Validation

Every synthetic example must be validated before it enters a training set:

- valid JSONL
- valid dataset schema
- valid tool names
- valid Pydantic tool arguments
- no unsafe path examples marked as accepted
- no missing labels
- no secrets
- compatible training format

Invalid generated records should be stored separately for debugging, not mixed
into the training dataset.

### 4. Dataset Build

The dataset builder should support multiple output views:

- supervised tool-use examples
- documentation-grounded instruction examples
- codebase-grounded instruction examples
- repair examples
- preference pairs for later ranking work
- evaluation-only held-out examples

The first training run should use supervised fine-tuning records. Preference
pairs can be collected early but used later.

### 5. Local Fine-Tuning

The first real training backend should be an infrastructure adapter, not
application logic. A reasonable implementation path is:

- supervised fine-tuning
- LoRA or QLoRA-style adapter output
- configurable base model
- configurable max steps
- configurable batch size and gradient accumulation
- checkpoint output under `.micro_model_agent/training/`

The training adapter must expose structured results:

- run id
- base model
- dataset version
- output artifact path
- training parameters
- start and end timestamps
- status
- metrics
- errors

### 6. Evaluation

Every training run should evaluate against held-out synthetic examples before it
is considered usable.

Initial evaluation checks:

- valid tool-call JSON rate
- correct tool selection rate
- unsafe action refusal rate
- repository path safety compliance
- schema validation pass rate
- repair action accuracy
- documentation rule compliance

The model promotion policy can be simple in V1: record the metrics and require
manual approval. Automatic promotion should come later.

### 7. Artifact Recording

Training artifacts should be recorded locally:

```text
.micro_model_agent/training/
  runs/
  adapters/
  merged-models/
  reports/
```

Artifact metadata should be JSON so future tooling can compare runs.

## CLI Shape

The MVP CLI should expose training as a normal workflow:

```bash
micro-agent dataset synthesize --count 500
micro-agent dataset validate
micro-agent dataset export --format sft-jsonl
micro-agent train synthetic --dry-run
micro-agent train synthetic --base-model Qwen/Qwen2.5-Coder-7B-Instruct
micro-agent eval synthetic --run-id latest
```

The exact flags can evolve. The capability should exist from the beginning.

For the first real local training attempt, install the optional training
dependencies and run with `--no-dry-run`:

```bash
pip install accelerate datasets peft torch transformers trl
micro-agent dataset synthesize --count 500
micro-agent train synthetic --no-dry-run --base-model Qwen/Qwen2.5-Coder-7B-Instruct
micro-agent eval synthetic --run-id latest
```

`train synthetic` exports the validated source dataset into SFT chat JSONL under
the run directory, then trains a PEFT LoRA adapter with Hugging Face
Transformers. The first local run should use conservative defaults, such as
small batch size, gradient accumulation, and a low `--max-steps`, before trying a
longer adapter run.

## Testing Strategy

Most tests should not require a GPU or a real 7B model.

Use:

- fake dataset generator for deterministic examples
- fake training runner for application tests
- tiny local model or dry-run backend for smoke tests when available
- schema validation tests for every generated example type
- held-out evaluation tests using fake model outputs

This lets the pipeline stay real without making normal development dependent on
special hardware.

## Boundaries

Domain:

- dataset concepts
- training run profiles
- model artifact metadata
- evaluation result vocabulary

Application:

- synthesize dataset use case
- validate dataset use case
- run synthetic training use case
- evaluate training run use case
- model promotion policy use case

Infrastructure:

- JSONL dataset store
- synthetic template loader
- Hugging Face-compatible training adapter
- fake training adapter
- artifact store

Interfaces:

- CLI commands
- future MCP read-only status tools

## First Implementation Slice

1. Define dataset and training run contracts.
2. Create hand-authored synthetic templates for the initial tools.
3. Generate JSONL records from templates.
4. Validate generated records with Pydantic contracts.
5. Add a fake training runner that writes artifact metadata.
6. Add a real training runner interface with a guarded implementation path.
7. Add CLI commands for synthesize, validate, train, and eval.
8. Add tests that exercise the full synthetic pipeline without a GPU.
