# micro-model-agent

micro-model-agent is an MIT-licensed Python framework for building specialized agents
powered by small local models, starting with Qwen-Coder 7B-class models running
locally. Ollama is the initial local inference interface.

The goal is not to build a smaller ChatGPT replacement. The goal is to prove that
small specialized models can perform meaningful software engineering tasks when they
operate inside constrained workflows with retrieval, tools, verification, and trace
capture.

## Status

Pre-alpha bootstrap. The repository currently contains the project skeleton, DDD
boundaries, dependency metadata, and documentation scaffolding for the first coding
agent vertical slice.

## Stack

- Python 3.12+
- uv
- Pydantic
- Pydantic AI
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

micro-model-agent follows strict Domain Driven Design boundaries:

- `domain`: framework-independent business contracts and policy.
- `application`: use cases and orchestration.
- `infrastructure`: model providers, repositories, vector stores, tracing, MCP adapters.
- `interfaces`: CLI, MCP server, and public API entrypoints.
- `agents`: reference agents built from workflows and tools.

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

See [docs/usage.md](docs/usage.md) for the current setup, agent execution,
fine-tuning, MCP, and synthetic-to-real data workflow.
