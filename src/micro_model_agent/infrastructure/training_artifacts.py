"""Compatibility imports for training artifacts, runners, and evaluation helpers."""

from micro_model_agent.infrastructure.evaluation.artifact import SyntheticEvaluationSuite
from micro_model_agent.infrastructure.evaluation.reports import (
    LocalEvaluationResultReader,
    LocalEvaluationResultWriter,
    load_evaluation_result,
    write_evaluation_result,
)
from micro_model_agent.infrastructure.persistence.training_records import (
    load_artifact_from_training_run,
    model_artifact_from_record,
    model_artifact_to_record,
    training_run_to_record,
)
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
from micro_model_agent.infrastructure.training.artifacts import (
    FakeTrainingRunner,
    JsonTrainingArtifactStore,
)
from micro_model_agent.infrastructure.training.local_finetuning import (
    HuggingFacePeftFineTuningBackend,
    LocalFineTuningBackend,
    LocalFineTuningResult,
    LocalFineTuningRunner,
    _bool_parameter,
    _float_parameter,
    _int_parameter,
    _model_load_kwargs,
    _numeric_metrics,
    _target_modules,
    _training_text_from_record,
)

__all__ = [
    "FakeTrainingRunner",
    "HuggingFacePeftFineTuningBackend",
    "JsonTrainingArtifactStore",
    "LocalEvaluationResultReader",
    "LocalEvaluationResultWriter",
    "LocalFineTuningBackend",
    "LocalFineTuningResult",
    "LocalFineTuningRunner",
    "LocalPromotionGateStore",
    "MinimumScorePromotionPolicy",
    "PromotionRegistryEntry",
    "SyntheticEvaluationSuite",
    "load_artifact_from_training_run",
    "load_evaluation_result",
    "load_promotion_registry",
    "model_artifact_from_record",
    "model_artifact_to_record",
    "promotion_registry_entry_from_record",
    "promotion_registry_entry_to_record",
    "record_promoted_artifact",
    "training_run_to_record",
    "write_evaluation_result",
    "write_promotion_gate_result",
    "_bool_parameter",
    "_float_parameter",
    "_int_parameter",
    "_model_load_kwargs",
    "_numeric_metrics",
    "_target_modules",
    "_training_text_from_record",
]
