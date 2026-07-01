"""Compatibility imports for promotion gate adapters."""

from micro_model_agent.infrastructure.promotion.gate import (
    LocalPromotionGateStore,
    MinimumScorePromotionPolicy,
    PromotionRegistryEntry,
    load_promotion_registry,
    promotion_registry_entry_from_record,
    promotion_registry_entry_to_record,
    record_promoted_artifact,
    write_promotion_gate_result,
)

__all__ = [
    "LocalPromotionGateStore",
    "MinimumScorePromotionPolicy",
    "PromotionRegistryEntry",
    "load_promotion_registry",
    "promotion_registry_entry_from_record",
    "promotion_registry_entry_to_record",
    "record_promoted_artifact",
    "write_promotion_gate_result",
]
