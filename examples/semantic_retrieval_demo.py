"""
Demonstration of semantic retrieval with CogCanvas.

This example shows how embeddings enable semantic search,
finding conceptually similar objects even without exact keyword matches.
"""

from cogcanvas import Canvas, CanvasObject, ObjectType


def demo_mock_backend():
    """Demo using mock embedding backend (no dependencies)."""
    print("=" * 60)
    print("DEMO 1: Mock Embedding Backend")
    print("=" * 60)

    # Create canvas with mock embeddings
    canvas = Canvas(embedding_model="mock")

    # Add some objects manually
    objects = [
        CanvasObject(
            type=ObjectType.DECISION,
            content="Use PostgreSQL as our primary database",
            context="Team decided in architecture meeting",
        ),
        CanvasObject(
            type=ObjectType.TODO,
            content="Set up PostgreSQL cluster with replication",
            context="DevOps task for high availability",
        ),
        CanvasObject(
            type=ObjectType.KEY_FACT,
            content="Current user base is 10,000 active users",
            context="From analytics dashboard",
        ),
        CanvasObject(
            type=ObjectType.REMINDER,
            content="Always use prepared statements to prevent SQL injection",
            context="Security best practice",
        ),
        CanvasObject(
            type=ObjectType.INSIGHT,
            content="Caching reduces database load by 60%",
            context="Performance analysis finding",
        ),
    ]

    # Add objects to canvas (embeddings computed automatically)
    for obj in objects:
        canvas.add(obj, compute_embedding=True)

    print(f"\nAdded {canvas.size} objects to canvas")
    print("\nCanvas statistics:")
    stats = canvas.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Semantic retrieval examples
    queries = [
        "What database are we using?",
        "How many users do we have?",
        "Security considerations for database",
        "Performance optimization insights",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Semantic search
        result = canvas.retrieve(query, method="semantic", top_k=3)

        if result.count == 0:
            print("No results found")
        else:
            print(f"\nTop {result.count} results (semantic):")
            for i, (obj, score) in enumerate(zip(result.objects, result.scores), 1):
                print(f"\n  {i}. [{obj.type.value}] (score: {score:.3f})")
                print(f"     {obj.content}")


def demo_with_extraction():
    """Demo with extraction from conversation."""
    print("\n\n" + "=" * 60)
    print("DEMO 2: Extraction + Semantic Retrieval")
    print("=" * 60)

    canvas = Canvas(embedding_model="mock")

    # Simulate a conversation
    conversations = [
        (
            "Let's use PostgreSQL for our database and Redis for caching",
            "Great choices! PostgreSQL is reliable and Redis is fast.",
        ),
        (
            "We need to implement user authentication with JWT tokens",
            "Good idea. I'll add that to the TODO list.",
        ),
        (
            "Remember to always validate user input to prevent XSS attacks",
            "Absolutely, security is paramount.",
        ),
        (
            "Our initial target is 10,000 concurrent users",
            "That's a solid goal. We should design for that scale.",
        ),
    ]

    print("\nProcessing conversation turns...")
    for i, (user, assistant) in enumerate(conversations, 1):
        result = canvas.extract(user=user, assistant=assistant)
        print(f"  Turn {i}: Extracted {result.count} objects")

    print(f"\nTotal objects in canvas: {canvas.size}")

    # Query the conversation memory
    queries = [
        "What technology choices did we make?",
        "What are our security requirements?",
        "How many users are we targeting?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = canvas.retrieve(query, method="semantic", top_k=2)

        if result.count > 0:
            for i, (obj, score) in enumerate(zip(result.objects, result.scores), 1):
                print(f"\n  {i}. [{obj.type.value}] (similarity: {score:.3f})")
                print(f"     {obj.content}")
                print(f"     Context: {obj.context}")


def demo_keyword_vs_semantic():
    """Compare keyword vs semantic retrieval."""
    print("\n\n" + "=" * 60)
    print("DEMO 3: Keyword vs Semantic Comparison")
    print("=" * 60)

    canvas = Canvas(embedding_model="mock")

    # Add objects with varied phrasing
    objects = [
        CanvasObject(
            type=ObjectType.KEY_FACT,
            content="The database management system we selected is PostgreSQL",
        ),
        CanvasObject(
            type=ObjectType.DECISION,
            content="Chose to implement Redis as our caching layer",
        ),
        CanvasObject(
            type=ObjectType.TODO,
            content="Configure the relational database cluster",
        ),
    ]

    for obj in objects:
        canvas.add(obj)

    query = "What database did we choose?"
    print(f"\nQuery: {query}")

    # Keyword retrieval
    print("\n--- Keyword Retrieval ---")
    result_kw = canvas.retrieve(query, method="keyword", top_k=3)
    for i, (obj, score) in enumerate(zip(result_kw.objects, result_kw.scores), 1):
        print(f"{i}. (score: {score:.3f}) {obj.content}")

    # Semantic retrieval
    print("\n--- Semantic Retrieval ---")
    result_sem = canvas.retrieve(query, method="semantic", top_k=3)
    for i, (obj, score) in enumerate(zip(result_sem.objects, result_sem.scores), 1):
        print(f"{i}. (score: {score:.3f}) {obj.content}")

    print("\nNote: Semantic retrieval can find conceptually similar content")
    print("even without exact keyword matches!")


def demo_persistence():
    """Demo saving and loading canvas with embeddings."""
    print("\n\n" + "=" * 60)
    print("DEMO 4: Persistence with Embeddings")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "canvas.json"

        # Create and populate canvas
        print("\nCreating canvas and adding objects...")
        canvas1 = Canvas(embedding_model="mock", storage_path=str(storage_path))

        canvas1.add(
            CanvasObject(
                type=ObjectType.DECISION,
                content="Use microservices architecture",
            )
        )
        canvas1.add(
            CanvasObject(
                type=ObjectType.KEY_FACT,
                content="Expected traffic: 1M requests/day",
            )
        )

        print(f"Canvas 1 size: {canvas1.size}")

        # Load from storage
        print(f"\nLoading canvas from {storage_path}...")
        canvas2 = Canvas(embedding_model="mock", storage_path=str(storage_path))

        print(f"Canvas 2 size: {canvas2.size}")
        print("\nObjects loaded:")
        for obj in canvas2.list_objects():
            has_embedding = "✓" if obj.embedding else "✗"
            print(f"  [{obj.type.value}] {obj.content[:50]}... (embedding: {has_embedding})")

        # Test retrieval on loaded canvas
        print("\nTesting retrieval on loaded canvas...")
        result = canvas2.retrieve("architecture decisions", method="semantic")
        if result.count > 0:
            print(f"Found: {result.objects[0].content}")


if __name__ == "__main__":
    demo_mock_backend()
    demo_with_extraction()
    demo_keyword_vs_semantic()
    demo_persistence()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
    print("\nTo use real embeddings, install sentence-transformers:")
    print("  pip install cogcanvas[embeddings]")
    print("\nThen initialize with:")
    print('  canvas = Canvas(embedding_model="all-MiniLM-L6-v2")')
