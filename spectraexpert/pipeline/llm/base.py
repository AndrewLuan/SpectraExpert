"""
Abstract base class for LLM API clients.

All LLM backends (OpenAI, Anthropic, dummy) implement this interface so
that the pipeline can swap providers without changing calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract base for LLM API clients used by the SpectraExpert pipeline.

    Parameters
    ----------
    config : dict with keys under stage3 → llm
        e.g. {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 4096,
            "api_key_env": "OPENAI_API_KEY",
            "endpoint": None,
            "timeout_seconds": 120,
        }
    """

    def __init__(self, config: dict):
        llm_cfg = config.get("stage3", config).get("llm", config)
        self._model = llm_cfg.get("model", "gpt-4o")
        self._temperature = float(llm_cfg.get("temperature", 0.2))
        self._max_tokens = int(llm_cfg.get("max_tokens", 4096))
        self._timeout = int(llm_cfg.get("timeout_seconds", 120))
        self._endpoint = llm_cfg.get("endpoint", None)
        # Read API key from env (fail open — subclass may validate)
        env_var = llm_cfg.get("api_key_env", "")
        self._api_key = ""
        if env_var:
            import os
            self._api_key = os.environ.get(env_var, "")

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """Send a chat-completion request and return the response text.

        Parameters
        ----------
        messages : list[{"role", "content"}] dicts (system, user, assistant).
        **kwargs : overrides for temperature, max_tokens, etc.

        Returns
        -------
        response_text : str — the model's response.
        """
        ...

    def chat_stream(self, messages: list[dict], **kwargs: Any):
        """Generator yielding response chunks (optional override)."""
        yield self.chat(messages, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model
