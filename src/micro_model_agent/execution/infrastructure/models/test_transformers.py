"""Tests for the Transformers-backed model provider helpers."""

from micro_model_agent.execution.infrastructure.models.transformers import (
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
        tools: list[dict[str, object]] | None = None,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        tool_count = len(tools or [])
        return f"templated:{messages[0]['role']}:{messages[0]['content']}:tools={tool_count}"


class PlainTokenizer:
    chat_template = None


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

    assert rendered == "templated:user:hello:tools=0"


def test_render_prompt_passes_tools_to_tokenizer_chat_template() -> None:
    rendered = _render_prompt(
        TemplateTokenizer(),
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "repo.read"}}],
    )

    assert rendered == "templated:user:hello:tools=1"


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


def test_render_plain_chat_prompt_includes_tools_when_provided() -> None:
    rendered = _render_plain_chat_prompt(
        [{"role": "user", "content": "do stuff"}],
        tools=[{"type": "function", "function": {"name": "repo.read"}}],
    )

    assert "TOOLS:" in rendered
    assert "repo.read" in rendered
    assert "USER:\ndo stuff" in rendered
    assert "ASSISTANT:\n" in rendered


def test_render_plain_chat_prompt_omits_tools_section_when_none() -> None:
    rendered = _render_plain_chat_prompt([{"role": "user", "content": "hi"}])

    assert "TOOLS:" not in rendered


def test_render_prompt_passes_tools_to_plain_fallback() -> None:
    rendered = _render_prompt(
        PlainTokenizer(),
        [{"role": "user", "content": "task"}],
        tools=[{"type": "function", "function": {"name": "repo.search"}}],
    )

    assert "TOOLS:" in rendered
    assert "repo.search" in rendered
