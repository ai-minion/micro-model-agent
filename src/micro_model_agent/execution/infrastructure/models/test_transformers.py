"""Tests for the Transformers-backed model provider helpers."""

from micro_model_agent.execution.infrastructure.models.transformers import (
    TransformersPeftModelProvider,
    _normalize_messages,
    _render_plain_chat_prompt,
    _render_prompt,
    _strip_end_tokens,
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


class FakeInputIds:
    shape = (1, 2)


class FakeInputs(dict):
    def to(self, _device: object) -> "FakeInputs":
        return self


class GeneratingTokenizer(TemplateTokenizer):
    eos_token = "<|endoftext|>"
    eos_token_id = 0

    def __call__(self, prompt: str, *, return_tensors: str) -> FakeInputs:
        assert prompt == "templated:user:hello:tools=1"
        assert return_tensors == "pt"
        return FakeInputs({"input_ids": FakeInputIds()})

    def decode(self, generated_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert generated_ids == [3]
        assert skip_special_tokens is False
        return "done<|im_end|>"


class GeneratingModel:
    device = "cpu"

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.kwargs = kwargs
        return [[1, 2, 3]]


class NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


class FakeTorch:
    def no_grad(self) -> NoGrad:
        return NoGrad()


def test_generate_records_rendered_request_text_before_tokenization() -> None:
    provider = TransformersPeftModelProvider("base")
    provider._tokenizer = GeneratingTokenizer()
    model = GeneratingModel()
    provider._model = model
    provider._torch = FakeTorch()

    response = provider._generate(
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "repo.read"}}],
    )

    assert response == "done"
    assert provider.last_request_text == "templated:user:hello:tools=1"
    assert provider.last_response_text == "done<|im_end|>"
    assert "max_time" not in model.kwargs


def test_generate_passes_timeout_to_transformers_max_time() -> None:
    provider = TransformersPeftModelProvider("base")
    provider._tokenizer = GeneratingTokenizer()
    model = GeneratingModel()
    provider._model = model
    provider._torch = FakeTorch()

    response = provider._generate(
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "repo.read"}}],
        timeout_seconds=2.5,
    )

    assert response == "done"
    assert model.kwargs["max_time"] == 2.5


class FakeTokenizer:
    eos_token = "<|endoftext|>"


def test_strip_end_tokens_removes_im_end() -> None:
    text = "<tool_call>\n{\"name\": \"repo.search\"}\n</tool_call><|im_end|>"
    assert _strip_end_tokens(text, FakeTokenizer()) == (
        "<tool_call>\n{\"name\": \"repo.search\"}\n</tool_call>"
    )


def test_strip_end_tokens_removes_endoftext() -> None:
    text = "some response<|endoftext|>"
    assert _strip_end_tokens(text, FakeTokenizer()) == "some response"


def test_strip_end_tokens_removes_eos_token_from_tokenizer() -> None:
    class CustomTokenizer:
        eos_token = "<custom_eos>"

    text = "hello<custom_eos>"
    assert _strip_end_tokens(text, CustomTokenizer()) == "hello"


def test_strip_end_tokens_preserves_tool_call_delimiters() -> None:
    text = "<tool_call>{\"name\": \"repo.read\", \"arguments\": {}}</tool_call>"
    assert _strip_end_tokens(text, FakeTokenizer()) == text


def test_strip_end_tokens_handles_tokenizer_with_no_eos_token() -> None:
    class NoEosTokenizer:
        eos_token = None

    text = "plain response<|im_end|>"
    assert _strip_end_tokens(text, NoEosTokenizer()) == "plain response"
