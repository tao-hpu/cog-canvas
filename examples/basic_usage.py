"""
Basic usage example for CogCanvas.

This demonstrates the core workflow:
1. Create a canvas
2. Extract objects from conversation turns
3. Retrieve relevant objects
4. Inject into prompts
"""

from cogcanvas import Canvas, ObjectType


def main():
    # Initialize canvas
    canvas = Canvas()

    # Simulate a multi-turn conversation
    turns = [
        {
            "user": "Let's use PostgreSQL for our database because we need good JSON support",
            "assistant": "Great choice! PostgreSQL's JSONB type is excellent for storing semi-structured data.",
        },
        {
            "user": "We should also add caching. Remember to use Redis.",
            "assistant": "Yes, Redis is perfect for caching. I'll note that as a TODO.",
        },
        {
            "user": "One important thing: the API rate limit is 100 requests per minute",
            "assistant": "Got it, I'll make sure to implement rate limiting accordingly.",
        },
    ]

    # Extract objects from each turn
    print("=" * 60)
    print("Extracting objects from conversation...")
    print("=" * 60)

    for i, turn in enumerate(turns, 1):
        result = canvas.extract(user=turn["user"], assistant=turn["assistant"])
        print(f"\nTurn {i}: Extracted {result.count} objects")
        for obj in result.objects:
            print(f"  - [{obj.type.value}] {obj.content[:60]}...")

    # Show canvas stats
    print("\n" + "=" * 60)
    print("Canvas Statistics")
    print("=" * 60)
    stats = canvas.stats()
    print(f"Total objects: {stats['total_objects']}")
    print(f"Turns processed: {stats['turn_count']}")
    print(f"By type: {stats['by_type']}")

    # Retrieve relevant objects
    print("\n" + "=" * 60)
    print("Retrieval Demo")
    print("=" * 60)

    query = "What database did we decide to use?"
    print(f"\nQuery: {query}")

    result = canvas.retrieve(query, top_k=3)
    print(f"Found {result.count} relevant objects:")
    for obj, score in zip(result.objects, result.scores):
        print(f"  - [{obj.type.value}] (score: {score:.2f}) {obj.content[:60]}...")

    # Inject into prompt
    print("\n" + "=" * 60)
    print("Injection Demo")
    print("=" * 60)

    injected = canvas.inject(result)
    print("\nInjected context:")
    print(injected)

    # List all objects
    print("\n" + "=" * 60)
    print("All Canvas Objects")
    print("=" * 60)

    for obj in canvas.list_objects():
        print(f"\n[{obj.type.value.upper()}] (turn {obj.turn_id})")
        print(f"  Content: {obj.content}")
        print(f"  Context: {obj.context}")


if __name__ == "__main__":
    main()
