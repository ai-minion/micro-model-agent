"""Training context domain services."""

from __future__ import annotations

from micro_model_agent.training.domain.exceptions import InvalidConfigError
from micro_model_agent.training.domain.value_objects import TrainingConfig


class TrainingConfigValidationService:
    """Validate a TrainingConfig before a TrainingJob is created.

    Rules:
    - ``base_model`` must not be blank.
    - ``output_dir`` must not be blank.
    - ``max_steps``, if set, must be positive.
    - ``learning_rate``, if set, must be positive.
    - ``batch_size``, if set, must be positive.
    """

    def validate(self, config: TrainingConfig) -> None:
        """Raise ``InvalidConfigError`` if the config violates any invariant."""

        if not config.base_model.strip():
            raise InvalidConfigError("base_model must not be blank")
        if not config.output_dir.strip():
            raise InvalidConfigError("output_dir must not be blank")
        if config.max_steps is not None and config.max_steps <= 0:
            raise InvalidConfigError(
                f"max_steps must be positive, got {config.max_steps}"
            )
        if config.learning_rate is not None and config.learning_rate <= 0:
            raise InvalidConfigError(
                f"learning_rate must be positive, got {config.learning_rate}"
            )
        if config.batch_size is not None and config.batch_size <= 0:
            raise InvalidConfigError(
                f"batch_size must be positive, got {config.batch_size}"
            )
