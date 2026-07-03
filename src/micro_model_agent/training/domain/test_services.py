"""Tests for TrainingConfigValidationService."""

from __future__ import annotations

import pytest

from micro_model_agent.training.domain.exceptions import InvalidConfigError
from micro_model_agent.training.domain.services import TrainingConfigValidationService
from micro_model_agent.training.domain.value_objects import TrainingConfig


def _cfg(**kwargs) -> TrainingConfig:
    defaults = dict(base_model="tinyllama", output_dir="/tmp/out")
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def test_valid_config_does_not_raise() -> None:
    svc = TrainingConfigValidationService()
    svc.validate(_cfg(max_steps=10, learning_rate=2e-4, batch_size=1))


def test_blank_base_model_raises() -> None:
    svc = TrainingConfigValidationService()
    with pytest.raises(InvalidConfigError, match="base_model"):
        svc.validate(_cfg(base_model="  "))


def test_blank_output_dir_raises() -> None:
    svc = TrainingConfigValidationService()
    with pytest.raises(InvalidConfigError, match="output_dir"):
        svc.validate(_cfg(output_dir=""))


def test_zero_max_steps_raises() -> None:
    svc = TrainingConfigValidationService()
    with pytest.raises(InvalidConfigError, match="max_steps"):
        svc.validate(_cfg(max_steps=0))


def test_negative_learning_rate_raises() -> None:
    svc = TrainingConfigValidationService()
    with pytest.raises(InvalidConfigError, match="learning_rate"):
        svc.validate(_cfg(learning_rate=-0.001))


def test_zero_batch_size_raises() -> None:
    svc = TrainingConfigValidationService()
    with pytest.raises(InvalidConfigError, match="batch_size"):
        svc.validate(_cfg(batch_size=0))


def test_none_optional_fields_are_valid() -> None:
    svc = TrainingConfigValidationService()
    svc.validate(_cfg(max_steps=None, learning_rate=None, batch_size=None))
