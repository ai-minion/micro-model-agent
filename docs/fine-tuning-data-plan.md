# Fine-Tuning Data Plan

## Purpose

Fine-tuning is a central goal for MicroModelAgent, and the MVP should include a
minimal local training pipeline based on synthetic data. The first pipeline does
not need production-scale training infrastructure, but it should make training a
normal workflow from day one:

- trace every CLI workflow
- store good and bad outcomes
- label failures clearly
- preserve tool schemas and tool-call examples
- collect documentation-grounded examples
- collect codebase-grounded examples
- generate a small synthetic seed dataset
- train a local adapter from validated synthetic data
- evaluate the trained artifact on held-out synthetic examples

The first target model family is Qwen-Coder 7B running locally on developer
hardware such as an RTX 3090. Ollama is the initial local inference interface.
The exact Ollama model tag and fine-tuning base model should remain configurable
so we can compare future Qwen-Coder releases without changing the application
architecture.

See [training-pipeline.md](training-pipeline.md) for the synthetic training
pipeline.

## Fine-Tuning Thesis

The fine-tuned model should learn the local operating environment, not become a
general coding oracle. It should become better at:

- choosing the correct MicroModelAgent tool
- producing valid tool arguments
- respecting repository safety rules
- using retrieved documentation instead of guessing
- following project-specific architecture boundaries
- proposing smaller and more verifiable patches
- reacting to test and diff feedback
- knowing when to stop and ask for approval

The orchestration layer remains responsible for enforcing schemas, permissions,
retrieval, verification, and trace persistence.

## What To Capture From Day One

Each workflow should create a structured record with enough information to build
supervised fine-tuning, preference, and evaluation datasets later.

Required fields:

- task id
- user goal
- repository profile
- agent profile
- model profile
- tool schema version
- system and workflow instructions
- retrieval query records
- retrieved context records
- model prompts or normalized prompt summaries
- model outputs
- tool calls
- tool results
- patch previews
- applied patches
- verification commands
- verification results
- final response
- final outcome label
- reviewer notes
- failure modes

Sensitive values must be redacted before persistence. Trace capture should record
that redaction happened without storing secrets.

## Outcome Labels

Every workflow should be labelable, even if no human review happens immediately.

Primary label:

- `accepted`
- `rejected`
- `needs_review`
- `partial`
- `errored`

Quality label:

- `good`
- `bad`
- `mixed`
- `unknown`

Useful failure modes:

- `invalid_tool_schema`
- `wrong_tool_selected`
- `unsafe_path_requested`
- `hallucinated_file`
- `retrieval_missed_context`
- `ignored_retrieved_context`
- `bad_patch`
- `patch_failed_to_apply`
- `test_failed`
- `verification_skipped`
- `too_large_change`
- `architecture_violation`
- `unclear_user_goal`
- `provider_error`

Bad examples are valuable. A rejected trace with a clear failure mode is useful
training and evaluation data.

## Dataset Record Types

The trace store should support deriving multiple dataset formats from the same
raw workflow records.

### Tool-Use Examples

Teach the model to select tools and produce valid tool arguments.

Input:

- goal
- available tool schemas
- retrieved context summary
- current workflow state

Target:

- tool name
- valid JSON arguments
- reason for tool selection

### Documentation-Grounded Examples

Teach the model to follow project docs and architecture rules.

Input:

- goal
- relevant documentation excerpts
- repository context

Target:

- plan or patch proposal that explicitly follows the retrieved rule

### Codebase-Grounded Examples

Teach the model local code conventions.

Input:

- goal
- retrieved similar code
- relevant tests

Target:

- small patch
- test update
- verification command

### Repair Examples

Teach the model to recover from verification output.

Input:

- initial patch
- test failure
- diff
- relevant code context

Target:

- corrected patch or next tool call

### Preference Pairs

Teach ranking later through good/bad comparisons.

Input:

- same goal and context

Preferred output:

- accepted or verified response

Rejected output:

- bad patch, invalid tool call, unsafe request, or failed verification response

## Synthetic Seed Data

Synthetic data should be small, explicit, and schema-focused at first. It should
teach the model how MicroModelAgent works before it tries to teach broad coding
skill.

Initial synthetic categories:

- valid `repo.search` calls for text, glob, and symbol searches
- invalid `repo.search` calls with corrected versions
- valid `repo.read` calls with line ranges
- unsafe `repo.read` path attempts and safe rejection behavior
- `repo.semantic_search` calls for architecture, tests, docs, and error history
- `repo.write_patch` dry-run examples
- patch application failure examples
- `test.run` allowlist examples
- rejected arbitrary shell requests
- `git.diff` self-review examples
- good coding-task traces with small patches
- bad coding-task traces with labeled failure modes

The first synthetic dataset should live outside source control by default when
generated:

```text
.micro_model_agent/datasets/synthetic_seed.jsonl
```

Hand-authored templates that are safe to commit can live under:

```text
examples/synthetic-data/
```

## Local Storage Layout

Local generated data should be ignored by Git:

```text
.micro_model_agent/
  index/
  traces/
  datasets/
  evaluations/
```

Suggested generated files:

```text
.micro_model_agent/traces/workflows.jsonl
.micro_model_agent/datasets/tool_use.jsonl
.micro_model_agent/datasets/documentation_grounded.jsonl
.micro_model_agent/datasets/codebase_grounded.jsonl
.micro_model_agent/datasets/repair.jsonl
.micro_model_agent/datasets/preference_pairs.jsonl
.micro_model_agent/evaluations/runs.jsonl
```

## CLI Requirements

The MVP CLI should make data capture visible and controllable:

```bash
micro-agent task "..." --dry-run
micro-agent task "..." --label accepted --quality good
micro-agent task "..." --label rejected --quality bad --failure-mode bad_patch
micro-agent dataset synthesize --count 100
micro-agent dataset validate
micro-agent dataset export --format jsonl
micro-agent train synthetic --dry-run
micro-agent eval synthetic --run-id latest
```

The exact command shape can change, but the capabilities should exist early:

- run task
- store trace
- label outcome
- generate synthetic examples
- validate synthetic examples
- export training-ready JSONL
- run local synthetic fine-tuning
- evaluate the trained artifact

## V1 Boundary

In V1, MicroModelAgent should:

- capture traces
- label outcomes
- generate synthetic seed data
- export dataset records
- run a local synthetic-data fine-tuning pipeline
- record training artifacts and metrics
- define evaluation suite interfaces
- define model promotion policy interfaces

In V1, MicroModelAgent should not:

- manage GPU clusters
- upload private code to external training services
- automatically promote models without evaluation

This keeps the data flywheel alive while making training a first-class workflow.
The initial training runner can be fake or tiny for tests, while real Qwen-Coder
7B fine-tuning targets local hardware such as the RTX 3090.
