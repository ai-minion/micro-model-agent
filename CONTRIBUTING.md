# Contributing

## Architecture Boundaries

micro-model-agent uses a Clean/Hexagonal layout:

- `domain` contains framework-independent data and policy. It must not import
  project outer layers or framework/provider packages.
- `application` owns use-case orchestration. It depends on domain objects and
  application ports, not concrete infrastructure, CLI, MCP, Typer, Pydantic, or
  model-provider packages.
- `infrastructure` implements ports for local files, model providers, training,
  evaluation, tool execution, and other concrete integrations.
- `interfaces` adapt CLI and MCP inputs/outputs. Keep Typer option parsing,
  prompts, printing, exit behavior, and compatibility shims here.

When adding a use case, prefer this shape:

1. Add request/result dataclasses and a workflow in `application`.
2. Add small ports in `application/ports.py` for concrete capabilities the
   workflow needs.
3. Implement those ports in `infrastructure`.
4. Keep CLI/MCP handlers thin: parse inputs, compose adapters, call workflows,
   and format outputs.
5. Add application tests with fakes for orchestration, plus focused
   infrastructure tests for file formats and provider behavior.

Run these before committing:

```text
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m ruff check src docs'
wsl -e bash -lc 'cd /mnt/d/Projects/code/micro-model-agent && .venv/bin/python -m pytest'
```

Architecture boundary tests live in
`src/micro_model_agent/test_architecture_boundaries.py`; update those tests only
when the intended layer rules change.
