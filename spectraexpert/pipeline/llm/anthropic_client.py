"""
Anthropic (Claude) LLM client using the Anthropic SDK.
"""

from __future__ import annotations

from typing import Any

from .base import LLMClient


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API.

    Reads ``ANTHROPIC_API_KEY`` from the environment by default.
    Configure via ``config['stage3']['llm']['model']`` (e.g. ``claude-sonnet-4-6``).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        if not self._api_key:
            import warnings
            warnings.warn(
                "ANTHROPIC_API_KEY is not set. "
                "Set the environment variable or configure 'api_key_env'."
            )

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # Convert OpenAI-style messages to Anthropic format
        system_msg = None
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] in ("user", "assistant"):
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        # Ensure at least one user message
        if not anthropic_messages or all(
            m["role"] != "user" for m in anthropic_messages
        ):
            anthropic_messages.append({"role": "user", "content": "Continue."})

        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=self._model,
                    system=system_msg,
                    messages=anthropic_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                )
            except anthropic.RateLimitError:
                if attempt < 2:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise

    def chat_stream(self, messages: list[dict], **kwargs: Any):
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        system_msg = None
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] in ("user", "assistant"):
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        if not anthropic_messages or all(
            m["role"] != "user" for m in anthropic_messages
        ):
            anthropic_messages.append({"role": "user", "content": "Continue."})

        with client.messages.stream(
            model=self._model,
            system=system_msg,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text
