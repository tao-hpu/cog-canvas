"""LLM utilities with exponential backoff retry for rate limiting and errors."""

import time
import random
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

# Retry configuration
DEFAULT_MAX_RETRIES = 8
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 120.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2
JITTER_FACTOR = 0.25  # Add 0-25% random jitter


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
) -> float:
    """
    Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current retry attempt (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential growth

    Returns:
        Delay in seconds
    """
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)
    # Add jitter to avoid thundering herd
    jitter = delay * JITTER_FACTOR * random.random()
    return delay + jitter


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable (rate limit or server error).

    Args:
        error: The exception to check

    Returns:
        True if the error should be retried
    """
    error_str = str(error).lower()

    # Check for rate limit errors
    if "429" in error_str or "rate" in error_str and "limit" in error_str:
        return True

    # Check for server errors (5xx)
    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return True

    # Check for timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return True

    # Check for connection errors
    if "connection" in error_str and ("reset" in error_str or "refused" in error_str or "error" in error_str):
        return True

    # Check OpenAI-specific errors
    if hasattr(error, 'status_code'):
        status_code = error.status_code
        if status_code in (429, 500, 502, 503, 504):
            return True

    return False


def call_llm_with_retry(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 200,
    temperature: float = 0,
    base_delay: float = DEFAULT_BASE_DELAY,
    verbose: bool = True,
    max_retries: int = 10,  # Maximum retry attempts (0 = infinite)
    **kwargs
) -> str:
    """
    Call LLM with retry until success or max retries reached.

    Args:
        client: OpenAI-compatible client
        model: Model name
        messages: List of message dicts
        max_tokens: Max tokens for response
        temperature: Sampling temperature
        base_delay: Initial delay between retries
        verbose: Whether to print retry messages
        max_retries: Maximum retry attempts (0 = infinite, default=10)
        **kwargs: Additional arguments for the API call

    Returns:
        Response content string

    Raises:
        Exception: If max_retries exceeded
    """
    attempt = 0

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            attempt += 1

            # Check if max retries exceeded
            if max_retries > 0 and attempt >= max_retries:
                if verbose:
                    print(f"[FAILED] Max retries ({max_retries}) exceeded: {type(e).__name__}: {e}")
                raise

            delay = calculate_backoff_delay(min(attempt, 10), base_delay)  # Cap at 10 for delay calc
            if verbose:
                print(f"[Retry {attempt}/{max_retries if max_retries > 0 else '∞'}] {type(e).__name__}: {e}")
                print(f"  Waiting {delay:.1f}s before retry...")
            time.sleep(delay)


def create_retry_wrapper(client: Any, model: str, **default_kwargs) -> Callable:
    """
    Create a reusable LLM call wrapper with retry logic.

    Args:
        client: OpenAI-compatible client
        model: Model name
        **default_kwargs: Default arguments for all calls

    Returns:
        A function that calls LLM with retry

    Example:
        llm_call = create_retry_wrapper(client, "gpt-4o-mini", max_tokens=200)
        response = llm_call([{"role": "user", "content": "Hello"}])
    """
    def wrapper(
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        merged_kwargs = {**default_kwargs, **kwargs}
        return call_llm_with_retry(
            client=client,
            model=model,
            messages=messages,
            **merged_kwargs
        )
    return wrapper


class RetryableLLMClient:
    """
    Wrapper around OpenAI client with automatic infinite retry logic.

    Usage:
        from openai import OpenAI
        client = OpenAI(api_key=..., base_url=...)
        retry_client = RetryableLLMClient(client, model="gpt-4o-mini")
        response = retry_client.chat_completion([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        client: Any,
        model: str,
        base_delay: float = DEFAULT_BASE_DELAY,
        verbose: bool = True,
    ):
        """
        Initialize retryable LLM client.

        Args:
            client: OpenAI-compatible client
            model: Model name
            base_delay: Initial delay between retries
            verbose: Whether to print retry messages
        """
        self._client = client
        self.model = model
        self.base_delay = base_delay
        self.verbose = verbose

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0,
        **kwargs
    ) -> str:
        """
        Call chat completion with infinite retry logic.

        Args:
            messages: List of message dicts
            max_tokens: Max tokens for response
            temperature: Sampling temperature
            **kwargs: Additional API arguments

        Returns:
            Response content string
        """
        return call_llm_with_retry(
            client=self._client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            base_delay=self.base_delay,
            verbose=self.verbose,
            **kwargs
        )

    @property
    def raw_client(self) -> Any:
        """Get the underlying client for direct access if needed."""
        return self._client
