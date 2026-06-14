# Synthetic Data Templates

This directory is for hand-authored synthetic-data templates that are safe to
commit.

Generated datasets should not be committed. They should be written under:

```text
.micro_model_agent/datasets/
```

Initial template categories:

- valid tool calls
- invalid tool calls with corrections
- unsafe path attempts with safe refusals
- documentation-grounded workflow decisions
- codebase-grounded patch plans
- verification repair examples
- good and bad labeled coding-agent traces

Templates should be small, explicit, and schema-focused. They exist to teach the
model MicroModelAgent's tools and operating rules before broader coding behavior.
