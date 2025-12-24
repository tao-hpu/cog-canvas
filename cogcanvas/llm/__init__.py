"""LLM backends for CogCanvas extraction."""

from typing import Optional
from cogcanvas.llm.base import LLMBackend, MockLLMBackend
from cogcanvas.llm.openai import OpenAIBackend

__all__ = ["LLMBackend", "MockLLMBackend", "OpenAIBackend", "get_backend"]


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
        >>> backend = get_backend("mock")
    """
    backends = {
        "mock": MockLLMBackend,
        "openai": OpenAIBackend,
    }

    if name.lower() not in backends:
        available = ", ".join(backends.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    return backends[name.lower()](**kwargs)
