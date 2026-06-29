"""Tests for the Transformers-backed model provider helpers."""

from micro_model_agent.infrastructure.transformers_model_provider import (
    _contains_complete_json_object,
)


def test_contains_complete_json_object_detects_finished_turn() -> None:
    assert _contains_complete_json_object('{"final_response":"done","ok":true}') is True
    assert _contains_complete_json_object('prefix {"tool_name":"repo.read","arguments":{}}') is True


def test_contains_complete_json_object_rejects_incomplete_turn() -> None:
    assert _contains_complete_json_object("") is False
    assert _contains_complete_json_object("thinking...") is False
    assert _contains_complete_json_object('{"tool_name":"repo.write_files","arguments":') is False
