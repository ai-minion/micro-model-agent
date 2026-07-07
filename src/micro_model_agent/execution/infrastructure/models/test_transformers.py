"""Tests for the Transformers-backed model provider helpers."""

from micro_model_agent.execution.infrastructure.models.transformers import (
    _contains_complete_json_object,
    _normalize_messages,
    _render_plain_chat_prompt,
    _render_prompt,
)


class TemplateTokenizer:
    chat_template = "template"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"templated:{messages[0]['role']}:{messages[0]['content']}"


class PlainTokenizer:
    chat_template = None


def test_contains_complete_json_object_detects_finished_turn() -> None:
    assert _contains_complete_json_object('{"final_response":"done","ok":true}') is True
    assert _contains_complete_json_object('prefix {"tool_name":"repo.read","arguments":{}}') is True


def test_contains_complete_json_object_rejects_incomplete_turn() -> None:
    assert _contains_complete_json_object("") is False
    assert _contains_complete_json_object("thinking...") is False
    assert _contains_complete_json_object('{"tool_name":"repo.write_files","arguments":') is False


def test_normalize_messages_uses_output_when_content_is_missing() -> None:
    messages = _normalize_messages(
        [
            {"role": "user", "output": {"ok": True}},
            {"role": "assistant", "content": "done"},
        ]
    )

    assert messages == [
        {"role": "user", "content": '{"ok": true}'},
        {"role": "assistant", "content": "done"},
    ]


def test_render_prompt_uses_tokenizer_chat_template_when_present() -> None:
    rendered = _render_prompt(
        TemplateTokenizer(),
        [{"role": "user", "content": "hello"}],
    )

    assert rendered == "templated:user:hello"


def test_render_prompt_falls_back_without_chat_template() -> None:
    rendered = _render_prompt(
        PlainTokenizer(),
        [
            {"role": "system", "content": "follow instructions"},
            {"role": "user", "content": "return json"},
        ],
    )

    assert rendered == (
        "SYSTEM:\nfollow instructions\n\n"
        "USER:\nreturn json\n\n"
        "ASSISTANT:\n"
    )


def test_render_plain_chat_prompt_defaults_missing_role_to_user() -> None:
    rendered = _render_plain_chat_prompt([{"content": "hello"}])

    assert rendered == "USER:\nhello\n\nASSISTANT:\n"
