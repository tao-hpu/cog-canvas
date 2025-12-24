"""
Demo script showing multiple LLM backend usage.

This demonstrates how to use OpenAI, Anthropic, and Mock backends
for extracting canvas objects from dialogue.
"""

import os
from cogcanvas import Canvas
from cogcanvas.llm import get_backend, OpenAIBackend, MockLLMBackend


def demo_mock_backend():
    """Demo using the mock backend (no API keys needed)."""
    print("\n" + "=" * 70)
    print("DEMO: Mock Backend (Rule-based extraction)")
    print("=" * 70)

    backend = get_backend("mock")
    canvas = Canvas(llm_backend=backend)

    # Test conversation
    result = canvas.extract(
        user="Let's use PostgreSQL for the database",
        assistant="Great choice! PostgreSQL is reliable and feature-rich.",
    )

    print(f"\nExtracted {len(result.objects)} objects:")
    for obj in result.objects:
        print(f"  [{obj.type.value}] {obj.content}")
        print(f"    Context: {obj.context}")
        print(f"    Confidence: {obj.confidence}")


def demo_openai_backend():
    """Demo using OpenAI backend (requires OPENAI_API_KEY)."""
    print("\n" + "=" * 70)
    print("DEMO: OpenAI Backend (GPT-4o-mini)")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    try:
        backend = OpenAIBackend(model="gpt-4o-mini")
        canvas = Canvas(llm_backend=backend)

        # Test conversation with multiple extractable objects
        result = canvas.extract(
            user="We need to implement authentication. The API has a rate limit of 1000 requests per hour. Remember, I prefer using TypeScript.",
            assistant="I'll help implement auth. Given the rate limit, we should add request caching. TypeScript noted!",
        )

        print(f"\nExtracted {len(result.objects)} objects:")
        for obj in result.objects:
            print(f"  [{obj.type.value}] {obj.content}")
            print(f"    Context: {obj.context}")
            print(f"    Confidence: {obj.confidence}")

        print(f"\nExtraction took: {result.extraction_time:.3f}s")
        print(f"Model used: {result.model_used}")

    except Exception as e:
        print(f"Error: {e}")





def demo_factory_function():
    """Demo using the get_backend factory function."""
    print("\n" + "=" * 70)
    print("DEMO: Factory Function (get_backend)")
    print("=" * 70)

    # Create backends using factory
    backends = {
        "mock": get_backend("mock"),
    }

    # Add OpenAI if key available
    if os.getenv("OPENAI_API_KEY"):
        backends["openai"] = get_backend("openai", model="gpt-4o-mini")



    print(f"\nAvailable backends: {list(backends.keys())}")

    # Test each backend
    for name, backend in backends.items():
        print(f"\nTesting {name} backend...")
        canvas = Canvas(llm_backend=backend)
        result = canvas.extract(
            user="We decided to use Redis for caching.",
            assistant="Good call, Redis is fast and reliable.",
        )
        print(f"  Extracted {len(result.objects)} objects")


def demo_comparison():
    """Compare extraction quality across backends."""
    print("\n" + "=" * 70)
    print("DEMO: Backend Comparison")
    print("=" * 70)

    test_cases = [
        {
            "user": "Let's use PostgreSQL and add Redis caching. We need to implement auth by next week.",
            "assistant": "Good choices! I'll prioritize the auth implementation.",
        },
        {
            "user": "The API rate limit is 1000/hour. Remember, keep responses under 500ms.",
            "assistant": "Noted. We'll need efficient caching and query optimization.",
        },
    ]

    backends = {"mock": get_backend("mock")}

    if os.getenv("OPENAI_API_KEY"):
        backends["openai"] = get_backend("openai", model="gpt-4o-mini")



    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"User: {case['user'][:60]}...")
        print()

        for name, backend in backends.items():
            canvas = Canvas(llm_backend=backend)
            result = canvas.extract(case["user"], case["assistant"])
            print(f"{name:12} -> {len(result.objects)} objects extracted")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CogCanvas LLM Backends Demo")
    print("=" * 70)
    print("\nThis demo showcases different LLM backends for canvas object extraction.")
    print("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY to test real LLM backends.")

    # Run demos
    demo_mock_backend()
    demo_openai_backend()
    demo_factory_function()
    demo_comparison()

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)
