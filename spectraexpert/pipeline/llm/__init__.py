"""LLM client package — factory function and client classes."""

from .base import LLMClient
from .dummy_client import DummyClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient

__all__ = [
    "LLMClient",
    "DummyClient",
    "OpenAIClient",
    "AnthropicClient",
    "create_llm_client",
]


def create_llm_client(config: dict) -> LLMClient:
    """Factory: return the appropriate LLMClient subclass based on config.

    Parameters
    ----------
    config : dict
        Pipeline config dict.  ``config['stage3']['llm']['provider']``
        determines which client is created (``openai`` | ``anthropic`` | ``dummy``).

    Returns
    -------
    LLMClient
    """
    provider = config.get("stage3", config).get("llm", config).get("provider", "dummy")
    mapping = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "dummy": DummyClient,
    }
    if provider not in mapping:
        import warnings
        warnings.warn(
            f"Unknown LLM provider '{provider}'; falling back to 'dummy'."
        )
        provider = "dummy"
    return mapping[provider](config)
