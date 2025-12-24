"""
Simple example of using different LLM backends with CogCanvas.
"""

import os
from cogcanvas import Canvas
from cogcanvas.llm import get_backend

# Example 1: Using Mock Backend (no API key needed)
print("=" * 60)
print("Example 1: Mock Backend")
print("=" * 60)

mock_backend = get_backend("mock")
canvas = Canvas(llm_backend=mock_backend)

result = canvas.extract(
    user="We need to implement authentication and use PostgreSQL",
    assistant="I'll start with the database setup and then add auth.",
)

print(f"\nExtracted {len(result.objects)} objects:")
for obj in result.objects:
    print(f"  [{obj.type.value}] {obj.content}")
    print(f"    Confidence: {obj.confidence}")


# Example 2: Using OpenAI Backend (requires OPENAI_API_KEY)
if os.getenv("OPENAI_API_KEY"):
    print("\n" + "=" * 60)
    print("Example 2: OpenAI Backend")
    print("=" * 60)

    openai_backend = get_backend("openai", model="gpt-4o-mini")
    canvas2 = Canvas(llm_backend=openai_backend)

    result2 = canvas2.extract(
        user="The API rate limit is 1000/hour. Remember to keep responses under 500ms.",
        assistant="Got it! I'll implement caching to stay within limits.",
    )

    print(f"\nExtracted {len(result2.objects)} objects:")
    for obj in result2.objects:
        print(f"  [{obj.type.value}] {obj.content}")
        print(f"    Context: {obj.context}")
        print(f"    Confidence: {obj.confidence}")
else:
    print("\n(Skipping OpenAI example - OPENAI_API_KEY not set)")





# Example 4: Using Canvas with automatic backend selection
print("\n" + "=" * 60)
print("Example 4: Automatic Backend Selection")
print("=" * 60)

# Canvas will use mock by default
auto_canvas = Canvas()
print(f"Backend type: {type(auto_canvas._backend).__name__}")

result4 = auto_canvas.extract(
    user="Let's decide on using TypeScript for the frontend",
    assistant="Great choice! TypeScript adds type safety.",
)

print(f"\nExtracted {len(result4.objects)} objects")
print(f"Total canvas objects: {auto_canvas.size}")
print(f"Turn count: {auto_canvas.turn_count}")

# Show canvas stats
print("\nCanvas statistics:")
stats = auto_canvas.stats()
for key, value in stats.items():
    print(f"  {key}: {value}")
