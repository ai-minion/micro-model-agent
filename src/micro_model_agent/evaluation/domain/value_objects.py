"""Evaluation context value objects.

``EvaluationResult`` is a cross-cutting type — its canonical home is
``shared.domain.value_objects``.  It is re-exported here for convenience so
code inside the evaluation context can import from its own package.
"""

from __future__ import annotations

from micro_model_agent.shared.domain.value_objects import EvaluationResult  # noqa: F401
