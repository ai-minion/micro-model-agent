# micro-model-agent

micro-model-agent is an MIT-licensed Python framework for building specialized agents
powered by small local models, starting with Qwen-Coder 7B-class models running
locally. Ollama is the initial local inference interface.

The goal is not to build a smaller ChatGPT replacement. The goal is to prove that
small specialized models can perform meaningful software engineering tasks when they
operate inside constrained workflows with retrieval, tools, verification, and trace
capture.

## Status

Pre-alpha, but past the skeleton stage. The repository includes a runnable CLI,
an MCP server, a model-driven tool loop, constrained repository tools, trace
capture, synthetic dataset generation, local training artifacts, behavioral
evaluation, and a manual promotion gate.

## Stack

- Python 3.12+
- uv
- Pydantic
- Official MCP Python SDK
- Ollama
- pytest
- ruff
- mypy

## Development

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
```

Training dependencies are optional and intentionally separate:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --group training
```

The local training command supports a fast metadata dry run and a real PEFT
adapter run when training dependencies and a base model are available:

```bash
micro-agent dataset synthesize --count 500
micro-agent train synthetic --dry-run
micro-agent train synthetic --no-dry-run --base-model Qwen/Qwen2.5-Coder-7B-Instruct
```

If `uv` is not installed yet, create the WSL virtual environment directly:

```bash
python3 -m venv .venv-wsl
```

## Package

The import package is:

```python
import micro_model_agent
```

## Architecture

micro-model-agent follows a classical DDD bounded-context-first structure:

- `shared`: shared kernel (base domain types used across all contexts).
- `execution`: agent workflows, tool loops, model providers.
- `dataset`: training dataset lifecycle — synthesis, export, merge, relabel.
- `training`: fine-tuning job lifecycle and artifact storage.
- `evaluation`: model evaluation, scoring rubrics, threshold events.
- `promotion`: model promotion gate, registry, Ollama packaging.
- `repository_ops`: source code retrieval, safe repo tools.
- `interfaces`: CLI, MCP server, and public API entrypoints.

See [docs/architecture.md](docs/architecture.md).

## Project Plan

See [docs/project-plan.md](docs/project-plan.md) for the detailed implementation
plan, milestones, tool plan, testing strategy, and MVP acceptance criteria.

See [docs/fine-tuning-data-plan.md](docs/fine-tuning-data-plan.md) for the plan
to capture good and bad workflow outcomes, synthetic tool-use examples, and
fine-tuning datasets from day one.

See [docs/training-pipeline.md](docs/training-pipeline.md) for the synthetic
training pipeline that will generate validated examples, run local fine-tuning,
evaluate artifacts, and keep production promotion manual.

See [docs/trained-model-proof-plan.md](docs/trained-model-proof-plan.md) for
the current priority: proving a trained local adapter beats the base model in
the agent loop and MCP path.

See [docs/usage.md](docs/usage.md) for the current setup, agent execution,
fine-tuning, MCP, and synthetic-to-real data workflow.
