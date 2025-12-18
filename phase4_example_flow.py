"""
Phase 4 Example: Demonstrating Temporal-Aware Answer Generation

This example shows how the CogCanvas agent uses specialized temporal prompts
to answer time-related questions more accurately.
"""

from cogcanvas.temporal import is_temporal_query


def demonstrate_temporal_detection():
    """Demonstrate how temporal queries are detected."""

    print("=" * 70)
    print("PHASE 4 EXAMPLE: Temporal-Aware Answer Generation")
    print("=" * 70)
    print()

    print("Step 1: Query Analysis")
    print("-" * 70)

    queries = [
        "When did Caroline go to the fitness group?",  # Temporal
        "What is her favorite hobby?",                 # Non-temporal
        "Did she visit yesterday?",                     # Temporal
        "Where does she live?",                         # Non-temporal
    ]

    for query in queries:
        is_temporal, keywords = is_temporal_query(query)

        print(f"\nQuery: '{query}'")
        print(f"  Type: {'TEMPORAL' if is_temporal else 'REGULAR'}")

        if is_temporal:
            print(f"  Keywords detected: {keywords}")
            print(f"  Action: Use TEMPORAL PROMPT with date-focused instructions")
        else:
            print(f"  Action: Use STANDARD PROMPT (CoT or Direct)")

    print()


def show_prompt_comparison():
    """Show the difference between temporal and non-temporal prompts."""

    print("\nStep 2: Prompt Selection")
    print("-" * 70)

    question_temporal = "When did I book the hotel?"
    question_regular = "What hotel did I book?"
    context = """
    [Object 1] KeyFact: "Booked Grand Hotel"
      - Source: "I booked the Grand Hotel yesterday"
      - Session: 1:56 pm on 8 May, 2023
      - Time expression: "yesterday"
      - Normalized time: "2023-05-07"
    """

    print("\n--- Temporal Question ---")
    print(f"Question: '{question_temporal}'")
    print("\nPrompt Used:")
    print("""
    You are answering a TEMPORAL question about WHEN something happened.

    ## Instructions for Temporal Reasoning
    1. Look for explicit dates (e.g., "May 7, 2023", "2023-05-08")
    2. Look for relative time expressions and their session context
    3. If a memory says "yesterday" and session was "8 May 2023", answer is "7 May 2023"
    4. Use normalized time if available

    ## Memory Context
    [Includes temporal metadata and normalized dates]

    ## Your Answer (provide the specific time/date):
    """)

    print("\n--- Non-Temporal Question ---")
    print(f"Question: '{question_regular}'")
    print("\nPrompt Used:")
    print("""
    You are an expert reasoning agent with access to a structured memory graph.

    ## Instructions
    1. Analyze the retrieved nodes
    2. Connect facts and infer relationships
    3. Synthesize a complete answer

    ## Memory Context
    [Standard context without temporal emphasis]

    ## Answer
    """)


def show_expected_improvements():
    """Show expected improvements from Phase 4."""

    print("\nStep 3: Expected Improvements")
    print("-" * 70)

    print("\nWithout Phase 4 (Standard Prompt):")
    print("  Question: 'When did I book the hotel?'")
    print("  Answer: 'Yesterday' or 'I booked the hotel yesterday'")
    print("  Issue: Relative time not resolved to absolute date")

    print("\nWith Phase 4 (Temporal Prompt):")
    print("  Question: 'When did I book the hotel?'")
    print("  Answer: 'May 7, 2023' or '7 May 2023'")
    print("  Benefit: Absolute date extracted using session context")

    print("\n\nKey Improvements:")
    improvements = [
        "1. Absolute dates instead of relative expressions",
        "2. Better use of session datetime for normalization",
        "3. Prioritization of normalized time fields",
        "4. Explicit instruction to extract temporal information",
        "5. No degradation on non-temporal questions",
    ]

    for improvement in improvements:
        print(f"  {improvement}")


def show_integration_flow():
    """Show how Phase 4 integrates with other phases."""

    print("\n\nStep 4: Integration with Other Phases")
    print("-" * 70)

    print("""
    Phase 1: Temporal Extraction
      - Extracts time expressions from conversation
      - Normalizes "yesterday" → "2023-05-07"
      - Stores in canvas object metadata
      ↓

    Phase 2: Graph Construction
      - Creates temporal causal edges
      - Orders events chronologically
      - Links related temporal events
      ↓

    Phase 3: Retrieval
      - Finds temporally-relevant objects
      - Can boost by temporal keywords
      - Returns objects with temporal metadata
      ↓

    Phase 4: Answer Generation ← YOU ARE HERE
      - Detects if query is temporal
      - Uses specialized temporal prompt
      - Emphasizes date extraction and normalization
      - Returns absolute dates when possible
    """)


def main():
    """Run the demonstration."""
    demonstrate_temporal_detection()
    show_prompt_comparison()
    show_expected_improvements()
    show_integration_flow()

    print("\n" + "=" * 70)
    print("Phase 4 implementation enables context-aware temporal reasoning!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
