"""Tests for promotion runtime composition helpers."""

from __future__ import annotations

from micro_model_agent.promotion.application.workflows import (
    RunPromotionGateWorkflow,
    RunPromotionListWorkflow,
    RunPromotionPackageOllamaWorkflow,
    RunPromotionRecordWorkflow,
    RunPromotionSelectWorkflow,
)
from micro_model_agent.promotion.infrastructure.runtime import (
    build_promotion_gate_workflow,
    build_promotion_list_workflow,
    build_promotion_package_ollama_workflow,
    build_promotion_record_workflow,
    build_promotion_select_workflow,
)


def test_promotion_runtime_builds_standard_workflows() -> None:
    assert isinstance(build_promotion_gate_workflow(), RunPromotionGateWorkflow)
    assert isinstance(build_promotion_record_workflow(), RunPromotionRecordWorkflow)
    assert isinstance(build_promotion_list_workflow(), RunPromotionListWorkflow)
    assert isinstance(build_promotion_select_workflow(), RunPromotionSelectWorkflow)
    assert isinstance(
        build_promotion_package_ollama_workflow(),
        RunPromotionPackageOllamaWorkflow,
    )
