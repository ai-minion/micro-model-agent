"""Integration tests for promotion infrastructure: gate store and registry."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from micro_model_agent.promotion.domain.aggregate import ModelRegistry, PromotedModel
from micro_model_agent.promotion.infrastructure.repository import (
    JsonlModelRegistryRepository,
)


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _model(artifact_id: UUID | None = None) -> PromotedModel:
    return PromotedModel(
        id=uuid4(),
        artifact_id=artifact_id or uuid4(),
        artifact_name="tinyllama-adapter",
        artifact_path="/tmp/adapter",
        base_model="tinyllama",
        promoted_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# JsonlModelRegistryRepository
# ---------------------------------------------------------------------------


def test_get_or_create_returns_new_registry(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    registry = _run(repo.get_or_create())
    assert registry is not None
    assert len(registry.list_models()) == 0


def test_get_or_create_returns_same_registry_on_second_call(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    r1 = _run(repo.get_or_create())
    r2 = _run(repo.get_or_create())
    assert r1.id == r2.id


def test_save_and_reload_preserves_promoted_models(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    registry = _run(repo.get_or_create())
    model = _model()
    registry.record_promotion(model)
    _run(repo.save(registry))

    reloaded = _run(repo.get_or_create())
    assert len(reloaded.list_models()) == 1
    assert reloaded.list_models()[0].artifact_name == "tinyllama-adapter"


def test_get_by_id_returns_correct_registry(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    registry = _run(repo.get_or_create())
    found = _run(repo.get(registry.id))
    assert found is not None
    assert found.id == registry.id


def test_get_by_wrong_id_returns_none(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    _run(repo.get_or_create())  # create the registry file
    assert _run(repo.get(uuid4())) is None


def test_multiple_promotions_accumulated(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    registry = _run(repo.get_or_create())
    for _ in range(3):
        registry.record_promotion(_model())
    _run(repo.save(registry))

    reloaded = _run(repo.get_or_create())
    assert len(reloaded.list_models()) == 3


def test_select_by_artifact_id_after_reload(tmp_path: Path) -> None:
    repo = JsonlModelRegistryRepository(tmp_path)
    registry = _run(repo.get_or_create())
    artifact_id = uuid4()
    model = _model(artifact_id=artifact_id)
    registry.record_promotion(model)
    _run(repo.save(registry))

    reloaded = _run(repo.get_or_create())
    found = reloaded.select(artifact_id)
    assert found is not None
    assert found.artifact_id == artifact_id


# ---------------------------------------------------------------------------
# PromotionGateService integration
# ---------------------------------------------------------------------------


def test_gate_service_with_registry_evaluate_gate_roundtrip(tmp_path: Path) -> None:
    """ModelRegistry.evaluate_gate() uses PromotionGateService internally."""
    registry = ModelRegistry()
    artifact_id = uuid4()

    # All scores above threshold → gate passes
    passed = registry.evaluate_gate(
        artifact_id=artifact_id,
        scores=[0.92, 0.88, 0.95],
        minimum_score=0.85,
    )
    assert passed is True

    # Verify the correct event was emitted
    from micro_model_agent.promotion.domain.events import PromotionGatePassed
    events = registry.pull_events()
    assert any(isinstance(e, PromotionGatePassed) for e in events)


def test_gate_service_failure_case(tmp_path: Path) -> None:
    registry = ModelRegistry()
    artifact_id = uuid4()

    # One score below threshold → gate fails
    passed = registry.evaluate_gate(
        artifact_id=artifact_id,
        scores=[0.92, 0.60],
        minimum_score=0.85,
    )
    assert passed is False

    from micro_model_agent.promotion.domain.events import PromotionGateFailed
    events = registry.pull_events()
    assert any(isinstance(e, PromotionGateFailed) for e in events)
