"""Runtime promotion workflow composition helpers."""

from __future__ import annotations

from micro_model_agent.promotion.application.workflows import (
    RunPromotionGateWorkflow,
    RunPromotionListWorkflow,
    RunPromotionPackageOllamaWorkflow,
    RunPromotionRecordWorkflow,
    RunPromotionSelectWorkflow,
)
from micro_model_agent.promotion.domain.services import PromotionGateService
from micro_model_agent.promotion.infrastructure.gate import (
    LocalPromotionGateStore,
)
from micro_model_agent.repository_ops.infrastructure.metadata import (
    LocalRepositoryModelConfigurationStore,
)
from micro_model_agent.training.infrastructure.ollama_packaging import (
    LocalOllamaAdapterPackager,
)

__all__ = [
    "build_promotion_gate_workflow",
    "build_promotion_list_workflow",
    "build_promotion_package_ollama_workflow",
    "build_promotion_record_workflow",
    "build_promotion_select_workflow",
]


def build_promotion_gate_workflow() -> RunPromotionGateWorkflow:
    """Build the standard promotion gate workflow."""

    store = LocalPromotionGateStore()
    return RunPromotionGateWorkflow(
        artifact_reader=store,
        evaluation_reader=store,
        result_writer=store,
        gate_service=PromotionGateService(),
    )


def build_promotion_record_workflow() -> RunPromotionRecordWorkflow:
    """Build the standard promotion registry record workflow."""

    store = LocalPromotionGateStore()
    return RunPromotionRecordWorkflow(
        artifact_reader=store,
        registry_writer=store,
    )


def build_promotion_list_workflow() -> RunPromotionListWorkflow:
    """Build the standard promotion registry listing workflow."""

    return RunPromotionListWorkflow(registry_reader=LocalPromotionGateStore())


def build_promotion_select_workflow() -> RunPromotionSelectWorkflow:
    """Build the standard repository model selection workflow."""

    return RunPromotionSelectWorkflow(
        registry_reader=LocalPromotionGateStore(),
        configuration_writer=LocalRepositoryModelConfigurationStore(),
    )


def build_promotion_package_ollama_workflow() -> RunPromotionPackageOllamaWorkflow:
    """Build the standard Ollama packaging workflow for promoted adapters."""

    return RunPromotionPackageOllamaWorkflow(
        registry_reader=LocalPromotionGateStore(),
        packager=LocalOllamaAdapterPackager(),
    )
