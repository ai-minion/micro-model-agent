"""Tests for the Ollama model provider response parsing helpers."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_provider() -> object:
    """Import OllamaModelProvider with the ollama package stubbed out."""
    stub = types.ModuleType("ollama")
    stub.AsyncClient = MagicMock  # type: ignore[attr-defined]
    sys.modules.setdefault("ollama", stub)
    from micro_model_agent.execution.infrastructure.models.ollama import (  # noqa: PLC0415
        OllamaModelProvider,
    )
    return OllamaModelProvider.__new__(OllamaModelProvider)


class TestResponseTextFromDictResult:
    def test_reads_content_when_present(self) -> None:
        p = _make_provider()
        result = {"message": {"content": "hello world"}}
        assert p._response_text(result) == "hello world"  # type: ignore[attr-defined]

    def test_reads_native_tool_call_when_content_empty(self) -> None:
        p = _make_provider()
        result = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "repo.read", "arguments": {"files": []}}}
                ],
            }
        }
        text = p._response_text(result)  # type: ignore[attr-defined]
        assert text is not None
        parsed = json.loads(text)
        assert parsed == {"name": "repo.read", "arguments": {"files": []}}

    def test_reads_native_tool_call_when_content_none(self) -> None:
        p = _make_provider()
        result = {
            "message": {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "repo.search", "arguments": {"query": "foo"}}}
                ],
            }
        }
        text = p._response_text(result)  # type: ignore[attr-defined]
        assert text is not None
        parsed = json.loads(text)
        assert parsed["name"] == "repo.search"
        assert parsed["arguments"] == {"query": "foo"}

    def test_returns_none_when_no_content_and_no_tool_calls(self) -> None:
        p = _make_provider()
        assert p._response_text({"message": {"content": ""}}) is None  # type: ignore[attr-defined]
        assert p._response_text({"message": {}}) is None  # type: ignore[attr-defined]


class TestResponseTextFromObjectResult:
    def test_reads_content_when_present(self) -> None:
        p = _make_provider()
        msg = SimpleNamespace(content="answer", tool_calls=None)
        result = SimpleNamespace(message=msg)
        assert p._response_text(result) == "answer"  # type: ignore[attr-defined]

    def test_reads_native_tool_call_from_object_style(self) -> None:
        p = _make_provider()
        func = SimpleNamespace(name="repo.write_patch", arguments={"patch": "..."})
        call = SimpleNamespace(function=func)
        msg = SimpleNamespace(content="", tool_calls=[call])
        result = SimpleNamespace(message=msg)
        text = p._response_text(result)  # type: ignore[attr-defined]
        assert text is not None
        parsed = json.loads(text)
        assert parsed["name"] == "repo.write_patch"
        assert parsed["arguments"] == {"patch": "..."}

    def test_returns_none_when_message_is_none(self) -> None:
        p = _make_provider()
        result = SimpleNamespace(message=None)
        assert p._response_text(result) is None  # type: ignore[attr-defined]
