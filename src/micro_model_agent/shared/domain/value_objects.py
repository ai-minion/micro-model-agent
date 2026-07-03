"""Shared kernel value objects used across multiple bounded contexts.

Types that are inherently cross-cutting (i.e. appear in more than one bounded
context's domain layer) live here so no context needs to import another
context's domain package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Pass/fail result plus optional score and details for later inspection.

    Used in the execution, training, evaluation, and promotion contexts.
    Canonical home: ``shared.domain.value_objects``.
    """

    passed: bool
    summary: str
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
