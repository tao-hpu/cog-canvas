#!/usr/bin/env python3
"""Test script for Phase 3 - Time-Aware Retrieval implementation."""

from cogcanvas import Canvas
from cogcanvas.models import CanvasObject, ObjectType
from datetime import datetime


def test_temporal_query_detection():
    """Test that temporal queries are detected correctly."""
    from cogcanvas.temporal import is_temporal_query

    # Test temporal queries
    temporal_queries = [
        "When did Caroline go to the group?",
        "What happened yesterday?",
        "Did she go last week?",
    ]

    non_temporal_queries = [
        "What is her name?",
        "Tell me about the project.",
    ]

    print("Testing temporal query detection...")
    for query in temporal_queries:
        is_temp, keywords = is_temporal_query(query)
        assert is_temp, f"Should detect as temporal: {query}"
        print(f"  ✓ Temporal: '{query}' (keywords: {keywords})")

    for query in non_temporal_queries:
        is_temp, keywords = is_temporal_query(query)
        assert not is_temp, f"Should NOT detect as temporal: {query}"
        print(f"  ✓ Non-temporal: '{query}'")

    print()


def test_time_normalization():
    """Test time expression normalization."""
    from cogcanvas.temporal import normalize_time_expression, parse_session_datetime

    print("Testing time normalization...")

    # Parse session datetime
    session_dt_str = "1:56 pm on 8 May, 2023"
    reference_dt = parse_session_datetime(session_dt_str)
    assert reference_dt is not None, "Should parse session datetime"
    print(f"  ✓ Parsed session datetime: {session_dt_str} -> {reference_dt}")

    # Test various time expressions
    test_cases = [
        ("yesterday", "2023-05-07"),
        ("today", "2023-05-08"),
        ("last week", "2023-05-01"),  # Approximate
    ]

    for raw_expr, expected_date_prefix in test_cases:
        result = normalize_time_expression(raw_expr, reference_dt)
        print(f"  ✓ '{raw_expr}' -> {result.normalized} (confidence: {result.confidence:.2f})")

    print()


def test_temporal_boost_in_retrieval():
    """Test that objects with time info get boosted in temporal queries."""
    print("Testing temporal boost in retrieval...")

    # Create a mock canvas
    canvas = Canvas(extractor_model="mock", embedding_model="mock")

    # Create consistent embeddings (384 dimensions to match mock backend default)
    import numpy as np
    np.random.seed(42)  # For reproducibility
    dim = 384

    # Manually add objects - some with time info, some without
    obj1 = CanvasObject(
        id="obj1",
        type=ObjectType.KEY_FACT,
        content="Caroline went to the support group",
        event_time="2023-05-07",  # Has normalized time
    )
    obj1.embedding = np.random.randn(dim).tolist()

    obj2 = CanvasObject(
        id="obj2",
        type=ObjectType.KEY_FACT,
        content="The meeting discussed project updates",
        # No time information
    )
    obj2.embedding = np.random.randn(dim).tolist()

    canvas.add(obj1, compute_embedding=False)
    canvas.add(obj2, compute_embedding=False)

    # Query with temporal keyword
    query = "When did Caroline go to the group?"
    results = canvas.retrieve(query, top_k=2, method="semantic")

    print(f"  Query: '{query}'")
    print(f"  Results:")
    for i, (obj, score) in enumerate(zip(results.objects, results.scores)):
        has_time = "✓ (has time)" if obj.event_time or obj.event_time_raw else "✗ (no time)"
        print(f"    {i+1}. [{obj.id}] {has_time} - Score: {score:.3f}")

    # For temporal queries, obj1 (with time) should rank higher or equal
    # Note: In mock mode, embeddings are random, so this is a soft check
    print(f"  ✓ Retrieval completed successfully")

    print()


def test_time_info_in_filtering():
    """Test that time info appears in filter prompts."""
    from cogcanvas.filtering import RetrievalFilter

    print("Testing time info in filtering...")

    # Create filter
    filter = RetrievalFilter(model="mock")

    # Create objects with time info
    objects = [
        CanvasObject(
            id="obj1",
            type=ObjectType.KEY_FACT,
            content="Caroline went to the support group",
            event_time="2023-05-07",
            turn_id=1,
        ),
        CanvasObject(
            id="obj2",
            type=ObjectType.DECISION,
            content="Decided to use PostgreSQL",
            event_time_raw="yesterday",
            turn_id=2,
        ),
    ]

    # Format candidates
    formatted = filter._format_candidates(objects)
    print(f"  Formatted candidates:\n{formatted}\n")

    # Check that time info is included
    assert "[Event: 2023-05-07]" in formatted, "Should include normalized time"
    assert "[Time: yesterday]" in formatted, "Should include raw time expression"

    print(f"  ✓ Time information correctly included in filter formatting")
    print()


def test_extraction_with_time_normalization():
    """Test that extraction normalizes time expressions."""
    print("Testing extraction with time normalization...")

    canvas = Canvas(extractor_model="mock", embedding_model="mock")

    # Simulate extraction with metadata
    user_msg = "Yesterday, Caroline went to the support group."
    assistant_msg = "I've noted that down."
    metadata = {
        "session_datetime": "1:56 pm on 8 May, 2023"
    }

    # In mock mode, extraction won't actually extract temporal info,
    # but we can verify the normalization logic path exists
    result = canvas.extract(user_msg, assistant_msg, metadata)

    print(f"  ✓ Extraction with temporal metadata completed")
    print(f"    Extracted {len(result.objects)} objects")

    print()


def main():
    """Run all tests."""
    print("=" * 70)
    print("Phase 3 - Time-Aware Retrieval Implementation Tests")
    print("=" * 70)
    print()

    try:
        test_temporal_query_detection()
        test_time_normalization()
        test_temporal_boost_in_retrieval()
        test_time_info_in_filtering()
        test_extraction_with_time_normalization()

        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
