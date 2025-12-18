#!/usr/bin/env python3
"""
Example demonstrating Phase 3 - Time-Aware Retrieval in CogCanvas.

This example shows how temporal information enhances retrieval accuracy
for time-related queries.
"""

from cogcanvas import Canvas
from cogcanvas.models import CanvasObject, ObjectType
import numpy as np


def create_example_canvas():
    """Create a canvas with example temporal objects."""
    print("Creating example canvas with temporal information...\n")

    # Initialize canvas
    canvas = Canvas(extractor_model="mock", embedding_model="mock")

    # Create objects with temporal information
    # (In real usage, these would be extracted by LLM)

    # Object 1: Event with normalized time
    obj1 = CanvasObject(
        id="obj1",
        type=ObjectType.KEY_FACT,
        content="Caroline went to the support group meeting",
        event_time="2023-05-07",  # Normalized ISO date
        event_time_raw="yesterday",
        quote="Yesterday, Caroline went to the support group.",
        turn_id=1,
    )

    # Object 2: Event without time information
    obj2 = CanvasObject(
        id="obj2",
        type=ObjectType.KEY_FACT,
        content="The team discussed quarterly goals",
        # No temporal information
        quote="We discussed our quarterly goals in the meeting.",
        turn_id=2,
    )

    # Object 3: Recent event with time
    obj3 = CanvasObject(
        id="obj3",
        type=ObjectType.DECISION,
        content="Decided to use PostgreSQL database",
        event_time="2023-05-08",
        event_time_raw="today",
        quote="Today we decided to go with PostgreSQL.",
        turn_id=3,
    )

    # Object 4: Older event with time
    obj4 = CanvasObject(
        id="obj4",
        type=ObjectType.KEY_FACT,
        content="Initial budget was set at $50,000",
        event_time="2023-04-15",
        event_time_raw="last month",
        quote="Last month, we set the initial budget at $50,000.",
        turn_id=4,
    )

    # Add mock embeddings (in real usage, computed automatically)
    np.random.seed(42)
    for obj in [obj1, obj2, obj3, obj4]:
        obj.embedding = np.random.randn(384).tolist()

    # Add objects to canvas
    canvas.add(obj1, compute_embedding=False)
    canvas.add(obj2, compute_embedding=False)
    canvas.add(obj3, compute_embedding=False)
    canvas.add(obj4, compute_embedding=False)

    print(f"Added {canvas.size} objects to canvas")
    print("-" * 70)

    return canvas


def demonstrate_temporal_query(canvas):
    """Demonstrate retrieval with temporal query."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Temporal Query")
    print("=" * 70)

    query = "When did Caroline go to the support group?"
    print(f"\nQuery: '{query}'")
    print(f"Type: Temporal query (contains 'when')")
    print()

    # Retrieve with time-aware boosting
    results = canvas.retrieve(query, top_k=4, method="semantic")

    print("Results (with temporal boosting):")
    print()
    for i, (obj, score) in enumerate(zip(results.objects, results.scores), 1):
        has_time = "✓" if (obj.event_time or obj.event_time_raw) else "✗"
        time_info = f" (Event: {obj.event_time})" if obj.event_time else " (no time)"

        print(f"{i}. [{obj.id}] Score: {score:.3f} | Time: {has_time}{time_info}")
        print(f"   Content: {obj.content}")
        print()

    print("Analysis:")
    print("- Objects with temporal information receive 30% score boost")
    print("- obj1 (Caroline/support group + time) ranks highest")
    print("- Temporal boost helps prioritize time-relevant objects")


def demonstrate_non_temporal_query(canvas):
    """Demonstrate retrieval with non-temporal query."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Non-Temporal Query")
    print("=" * 70)

    query = "What database did we choose?"
    print(f"\nQuery: '{query}'")
    print(f"Type: Non-temporal query (no temporal keywords)")
    print()

    # Retrieve without temporal boost
    results = canvas.retrieve(query, top_k=4, method="semantic")

    print("Results (no temporal boosting applied):")
    print()
    for i, (obj, score) in enumerate(zip(results.objects, results.scores), 1):
        has_time = "✓" if (obj.event_time or obj.event_time_raw) else "✗"
        time_info = f" (Event: {obj.event_time})" if obj.event_time else " (no time)"

        print(f"{i}. [{obj.id}] Score: {score:.3f} | Time: {has_time}{time_info}")
        print(f"   Content: {obj.content}")
        print()

    print("Analysis:")
    print("- No temporal keywords detected, so no boost applied")
    print("- Ranking based purely on semantic/keyword similarity")
    print("- obj3 (PostgreSQL decision) should rank high")


def demonstrate_time_normalization():
    """Demonstrate time expression normalization."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Time Normalization")
    print("=" * 70)

    from cogcanvas.temporal import (
        normalize_time_expression,
        parse_session_datetime,
    )

    # Parse session datetime
    session_str = "1:56 pm on 8 May, 2023"
    reference_dt = parse_session_datetime(session_str)

    print(f"\nSession Datetime: {session_str}")
    print(f"Parsed Reference: {reference_dt}")
    print()

    # Test various temporal expressions
    expressions = [
        "yesterday",
        "today",
        "last Tuesday",
        "2 weeks ago",
        "May 7, 2023",
    ]

    print("Time Expression Normalization:")
    print()
    for expr in expressions:
        result = normalize_time_expression(expr, reference_dt)
        print(f"Input:      '{expr}'")
        print(f"Normalized: {result.normalized}")
        print(f"Type:       {result.temporal_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print()


def demonstrate_filter_with_time():
    """Demonstrate LLM filter with temporal information."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: LLM Filter with Temporal Info")
    print("=" * 70)

    from cogcanvas.filtering import RetrievalFilter

    # Create some example objects
    objects = [
        CanvasObject(
            id="obj1",
            type=ObjectType.KEY_FACT,
            content="Caroline went to the support group meeting",
            event_time="2023-05-07",
            turn_id=1,
            quote="Yesterday, Caroline went to the support group.",
        ),
        CanvasObject(
            id="obj2",
            type=ObjectType.DECISION,
            content="Decided to use PostgreSQL",
            event_time_raw="today",
            turn_id=3,
            quote="Today we decided to go with PostgreSQL.",
        ),
    ]

    # Initialize filter
    filter = RetrievalFilter(model="mock")

    # Format candidates
    formatted = filter._format_candidates(objects)

    print("\nFormatted Candidates (as seen by LLM filter):")
    print("-" * 70)
    print(formatted)
    print("-" * 70)
    print()

    print("Observations:")
    print("- Temporal information clearly marked with [Event: ...] or [Time: ...]")
    print("- LLM filter can assess temporal relevance")
    print("- Helps filter out temporally incorrect but semantically similar objects")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Phase 3 - Time-Aware Retrieval: Usage Examples")
    print("=" * 70)

    # Create example canvas
    canvas = create_example_canvas()

    # Demonstrate different query types
    demonstrate_temporal_query(canvas)
    demonstrate_non_temporal_query(canvas)

    # Show time normalization
    demonstrate_time_normalization()

    # Show filter formatting
    demonstrate_filter_with_time()

    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. Temporal queries automatically boost objects with time information")
    print("2. Non-temporal queries are unaffected (backward compatible)")
    print("3. Time expressions are normalized to ISO dates for consistency")
    print("4. LLM filter sees temporal information for better relevance assessment")
    print()


if __name__ == "__main__":
    import sys

    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
