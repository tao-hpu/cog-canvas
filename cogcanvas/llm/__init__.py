"""LLM backends for CogCanvas extraction."""

from typing import Optional
from cogcanvas.llm.base import LLMBackend, MockLLMBackend
from cogcanvas.llm.openai import OpenAIBackend
from cogcanvas.llm.anthropic_backend import AnthropicBackend

__all__ = ["LLMBackend", "MockLLMBackend", "OpenAIBackend", "AnthropicBackend", "get_backend"]


def get_backend(name: str, **kwargs) -> LLMBackend:
    """
    Factory function to create LLM backends by name.

    Args:
        name: Backend name ("mock", "openai", "anthropic")
        **kwargs: Additional arguments passed to backend constructor

    Returns:
        LLMBackend instance

    Raises:
        ValueError: If backend name is not recognized

    Examples:
        >>> backend = get_backend("openai", model="gpt-4o-mini")
        >>> backend = get_backend("anthropic", model="claude-3-5-haiku-latest")
        >>> backend = get_backend("mock")
    """
    backends = {
        "mock": MockLLMBackend,
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
    }

    if name.lower() not in backends:
        available = ", ".join(backends.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    return backends[name.lower()](**kwargs)
