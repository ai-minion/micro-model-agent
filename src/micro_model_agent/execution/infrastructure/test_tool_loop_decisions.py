"""Tests for parsing model-authored tool-loop decisions."""

from __future__ import annotations

from micro_model_agent.execution.infrastructure.tool_loop_decisions import parse_model_response


def test_parse_model_response_reads_native_tool_call() -> None:
    decision = parse_model_response(
        'I need to inspect the file.\n<tool_call>\n'
        '{"name":"repo.read","arguments":{"files":[{"path":"README.md"}]}}\n'
        "</tool_call>"
    )

    assert decision.kind == "tool_call"
    assert decision.tool_name == "repo.read"
    assert decision.arguments == {"files": [{"path": "README.md"}]}
    assert decision.reason == "I need to inspect the file."


def test_parse_model_response_reads_inner_native_tool_json() -> None:
    decision = parse_model_response(
        '{"name":"repo.search","arguments":{"query":"trace_store"}}'
    )

    assert decision.kind == "tool_call"
    assert decision.tool_name == "repo.search"
    assert decision.arguments == {"query": "trace_store"}


def test_parse_model_response_treats_plain_content_as_final_response() -> None:
    decision = parse_model_response("The implementation already does that.")

    assert decision.kind == "final_response"
    assert decision.response == "The implementation already does that."
    assert decision.ok is True


def test_parse_model_response_keeps_legacy_json_tool_call_compatibility() -> None:
    decision = parse_model_response(
        '{"tool_name":"repo.read","arguments":{"files":[{"path":"app.py"}]}}'
    )

    assert decision.kind == "tool_call"
    assert decision.tool_name == "repo.read"
    assert decision.arguments == {"files": [{"path": "app.py"}]}


def test_parse_model_response_keeps_legacy_json_final_response_compatibility() -> None:
    decision = parse_model_response('{"final_response":"done","ok":true}')

    assert decision.kind == "final_response"
    assert decision.response == "done"
    assert decision.ok is True


# ── P2 fixes ──────────────────────────────────────────────────────────────────

def test_parse_model_response_native_final_response_not_treated_as_tool_call() -> None:
    """{"name":"final_response","arguments":{...}} must become kind=final_response."""
    decision = parse_model_response(
        '{"name":"final_response","arguments":{"body":"All done."}}'
    )

    assert decision.kind == "final_response"
    assert decision.response == "All done."
    assert decision.ok is True


def test_parse_model_response_native_final_response_in_tool_call_tags() -> None:
    """Same normalization must apply when wrapped in <tool_call> tags."""
    decision = parse_model_response(
        '<tool_call>{"name":"final_response","arguments":{"response":"done","ok":true}}</tool_call>'
    )

    assert decision.kind == "final_response"
    assert decision.response == "done"
    assert decision.ok is True


def test_strip_native_response_markup_removes_stop_tag() -> None:
    from micro_model_agent.execution.infrastructure.tool_loop_decisions import strip_native_response_markup

    assert strip_native_response_markup("Done.<stop>").strip() == "Done."
    assert strip_native_response_markup("Done.<stop/>").strip() == "Done."
    assert strip_native_response_markup("Done.<stop />").strip() == "Done."


def test_strip_native_response_markup_removes_tools_wrapper() -> None:
    from micro_model_agent.execution.infrastructure.tool_loop_decisions import strip_native_response_markup

    result = strip_native_response_markup(
        '<tools>[{"type":"function","function":{"name":"repo.read"}}]</tools>Answer here.'
    )
    assert "tools" not in result.lower()
    assert "Answer here." in result


def test_parse_model_response_stop_tag_stripped_from_plain_response() -> None:
    """A model that appends <stop> to its final answer should still be parseable."""
    decision = parse_model_response("The file has 42 lines.<stop>")

    assert decision.kind == "final_response"
    assert "<stop>" not in decision.response
    assert "42 lines" in decision.response
