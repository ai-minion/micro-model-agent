"""Promotion infrastructure adapters and policies."""

from micro_model_agent.infrastructure.promotion.gate import (
    LocalPromotionGateStore,
    MinimumScorePromotionPolicy,
    PromotionRegistryEntry,
)

__all__ = [
    "LocalPromotionGateStore",
    "MinimumScorePromotionPolicy",
    "PromotionRegistryEntry",
]
