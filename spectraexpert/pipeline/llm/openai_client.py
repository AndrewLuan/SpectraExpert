"""
OpenAI-compatible LLM client.

Covers:
- OpenAI API (gpt-4o, gpt-4, etc.)
- Azure OpenAI endpoints
- Local/self-hosted OpenAI-compatible endpoints (vLLM, Ollama, LiteLLM)
"""

from __future__ import annotations

import time
from typing import Any

from .base import LLMClient


class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible chat completion APIs.

    Reads API key from the environment variable specified in the config
    (default ``OPENAI_API_KEY``).  Supports a custom base URL via
    ``config['stage3']['llm']['endpoint']``.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        if not self._api_key:
            import warnings
            warnings.warn(
                "OPENAI_API_KEY is not set. "
                "Set the environment variable or configure 'api_key_env'."
            )

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        import openai

        client_kwargs: dict = {}
        if self._endpoint:
            client_kwargs["base_url"] = self._endpoint

        client = openai.OpenAI(api_key=self._api_key, **client_kwargs)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # Retry up to 3 times on transient errors
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self._timeout,
                )
                return resp.choices[0].message.content or ""
            except openai.RateLimitError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except openai.APITimeoutError:
                if attempt < 2:
                    time.sleep(1)
                else:
                    raise

    def chat_stream(self, messages: list[dict], **kwargs: Any):
        import openai

        client_kwargs: dict = {}
        if self._endpoint:
            client_kwargs["base_url"] = self._endpoint

        client = openai.OpenAI(api_key=self._api_key, **client_kwargs)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        stream = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=self._timeout,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
