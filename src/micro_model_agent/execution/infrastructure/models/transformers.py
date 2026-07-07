"""Transformers/PEFT-backed model provider.

This provider runs a Hugging Face causal language model directly in Python,
optionally with a PEFT adapter loaded on top of the base model.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class TransformersPeftModelProvider:
    """Model provider that loads a local or Hugging Face PEFT adapter."""

    def __init__(
        self,
        base_model: str,
        *,
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 384,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        dtype: str = "float16",
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.dtype = dtype
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            # Loading a model can block for a while, so run it in a worker thread
            # instead of blocking the async event loop.
            await asyncio.to_thread(self._load)
        return await asyncio.to_thread(self._generate, messages, tools=tools)

    def _load(self) -> None:
        """Load tokenizer, base model, and optional PEFT adapter lazily."""

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_source = self.adapter_path if self.adapter_path else self.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            # Some causal language models do not define a separate pad token.
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if torch.cuda.is_available():
            # Prefer a single GPU when it has enough free VRAM (avoids CPU offloading
            # when system RAM is constrained). Fall back to device_map for multi-GPU.
            free_vram, _ = torch.cuda.mem_get_info(0)
            if torch.cuda.device_count() == 1 and free_vram > 10 * 1024 ** 3:
                model_kwargs["device_map"] = "cuda:0"
            else:
                model_kwargs["device_map"] = self.device_map
            model_kwargs["dtype"] = self._torch_dtype(torch)

        model: Any = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)
        if self.adapter_path is not None:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def _generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate text for one turn using the already-loaded model."""

        if self._torch is None or self._model is None or self._tokenizer is None:
            raise RuntimeError("model provider has not been loaded")

        prompt = _render_prompt(
            self._tokenizer,
            _normalize_messages(messages),
            tools=tools,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        with self._torch.no_grad():
            # do_sample=False makes generation deterministic for repeatable runs.
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=self._json_stopping_criteria(
                    prompt_token_count=inputs["input_ids"].shape[1]
                ),
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        # Slice off the prompt tokens so callers only receive newly generated text.
        decoded = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return str(decoded).strip()

    def _json_stopping_criteria(self, *, prompt_token_count: int) -> Any:
        """Stop generation once the model has emitted one complete JSON object."""

        from transformers import StoppingCriteria, StoppingCriteriaList

        if self._tokenizer is None:
            raise RuntimeError("model provider has not been loaded")
        tokenizer: Any = self._tokenizer

        class CompleteJsonObjectCriteria(StoppingCriteria):  # type: ignore[misc]
            def __call__(
                self,
                input_ids: Any,
                scores: Any,
                **kwargs: Any,
            ) -> bool:
                del scores, kwargs
                generated_ids = input_ids[0][prompt_token_count:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                return _contains_complete_json_object(str(text))

        return StoppingCriteriaList([CompleteJsonObjectCriteria()])

    def _torch_dtype(self, torch: Any) -> Any:
        """Translate a small string option into the torch dtype object."""

        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32
        return torch.float16


def _contains_complete_json_object(text: str) -> bool:
    """Return true once text contains a syntactically complete first JSON object."""

    stripped = text.strip()
    if not stripped:
        return False
    first_brace = stripped.find("{")
    if first_brace < 0:
        return False
    try:
        _, end_index = json.JSONDecoder().raw_decode(stripped[first_brace:])
    except json.JSONDecodeError:
        return False
    return end_index > 0


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize model messages into role/content pairs."""

    normalized: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content")
        if content is None and "output" in message:
            content = json.dumps(message["output"])
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(content or ""),
            }
        )
    return normalized


def _render_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Render chat messages with the tokenizer template, or a plain fallback."""

    if getattr(tokenizer, "chat_template", None):
        template_kwargs: dict[str, Any] = {}
        if tools:
            template_kwargs["tools"] = tools
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        )
    return _render_plain_chat_prompt(messages, tools=tools)


def _render_plain_chat_prompt(
    messages: list[dict[str, str]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Fallback prompt format for tokenizers without a chat template.

    When ``tools`` is provided (native tool-schema list), the schemas are
    rendered as a ``TOOLS:`` section so the model can still read argument
    names and types even without a tokenizer chat template.
    """

    parts: list[str] = []
    if tools:
        parts.append(f"TOOLS:\n{json.dumps(tools, separators=(',', ':'))}")
    parts.extend(
        f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}"
        for message in messages
    )
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)
