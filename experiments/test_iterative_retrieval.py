"""
Quick test: Iterative Retrieval vs Single-round Retrieval
"""

import json
import sys

sys.path.insert(0, "/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas-all/cog-canvas")

from dotenv import load_dotenv

load_dotenv()

from cogcanvas import Canvas
from experiments.locomo_adapter import load_locomo, convert_to_eval_format


def test_iterative_retrieval(num_samples=20):
    """Compare single-round vs iterative retrieval on LoCoMo questions."""

    # Load one conversation
    raw_data = load_locomo("experiments/data/locomo10.json")
    conversations = convert_to_eval_format(raw_data)
    conv = conversations[0]  # Use first conversation

    print(f"Testing on conversation: {conv.id}")
    print(f"Turns: {len(conv.turns)}, Questions: {len(conv.qa_pairs)}")

    # Initialize canvas and process all turns (use models from .env)
    import os

    canvas = Canvas(
        extractor_model=os.getenv("MODEL_DEFAULT", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "bge-large-zh-v1.5"),
    )

    print("\nProcessing turns...")
    for i, turn in enumerate(conv.turns):
        canvas.extract(user=turn.user, assistant=turn.assistant)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(conv.turns)} turns")

    print(f"\nCanvas stats: {canvas.stats()}")

    # Test on sample questions
    questions = conv.qa_pairs[:num_samples]

    single_scores = []
    iterative_scores = []

    print(f"\nTesting {len(questions)} questions...")

    for i, qa in enumerate(questions):
        query = qa.question
        expected = qa.answer

        # Single-round retrieval (top_k=15)
        result_single = canvas.retrieve(
            query, top_k=15, method="hybrid", include_related=True
        )
        single_context = canvas.inject(result_single, format="compact", max_tokens=2000)

        # Iterative retrieval: Round 1 + Round 2
        result_r1 = canvas.retrieve(
            query, top_k=10, method="hybrid", include_related=True
        )
        if result_r1.objects:
            # Expand query with top object's content
            expanded_query = f"{query} {result_r1.objects[0].content}"
            result_r2 = canvas.retrieve(
                expanded_query, top_k=10, method="hybrid", include_related=True
            )

            # Merge and dedupe
            seen_ids = set()
            merged_objects = []
            for obj in result_r1.objects + result_r2.objects:
                if obj.id not in seen_ids:
                    seen_ids.add(obj.id)
                    merged_objects.append(obj)

            # Create merged result (take top 15)
            from cogcanvas.models import RetrievalResult

            result_iterative = RetrievalResult(
                objects=merged_objects[:15],
                scores=[1.0] * min(15, len(merged_objects)),
                query=query,
            )
        else:
            result_iterative = result_r1

        iterative_context = canvas.inject(
            result_iterative, format="compact", max_tokens=2000
        )

        # Simple scoring: check if expected keywords appear in context
        expected_keywords = set(expected.lower().split())

        single_keywords = set(single_context.lower().split())
        iterative_keywords = set(iterative_context.lower().split())

        single_overlap = len(expected_keywords & single_keywords) / max(
            len(expected_keywords), 1
        )
        iterative_overlap = len(expected_keywords & iterative_keywords) / max(
            len(expected_keywords), 1
        )

        single_scores.append(single_overlap)
        iterative_scores.append(iterative_overlap)

        # Show comparison for interesting cases
        if abs(single_overlap - iterative_overlap) > 0.1:
            print(f"\n[{i+1}] Q: {query[:60]}...")
            print(
                f"    Single: {single_overlap:.1%} | Iterative: {iterative_overlap:.1%}"
            )
            if iterative_overlap > single_overlap:
                print(
                    f"    ✅ Iterative better by {(iterative_overlap - single_overlap)*100:.1f}pp"
                )
            else:
                print(
                    f"    ❌ Single better by {(single_overlap - iterative_overlap)*100:.1f}pp"
                )

    # Summary
    avg_single = sum(single_scores) / len(single_scores)
    avg_iterative = sum(iterative_scores) / len(iterative_scores)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Single-round avg keyword overlap:    {avg_single:.1%}")
    print(f"Iterative avg keyword overlap:       {avg_iterative:.1%}")
    print(
        f"Difference:                          {(avg_iterative - avg_single)*100:+.1f}pp"
    )

    # Count wins
    single_wins = sum(1 for s, i in zip(single_scores, iterative_scores) if s > i)
    iterative_wins = sum(1 for s, i in zip(single_scores, iterative_scores) if i > s)
    ties = sum(1 for s, i in zip(single_scores, iterative_scores) if s == i)

    print(
        f"\nWin/Loss/Tie: Single={single_wins}, Iterative={iterative_wins}, Tie={ties}"
    )

    if avg_iterative > avg_single + 0.02:
        print("\n✅ CONCLUSION: Iterative retrieval shows improvement!")
    elif avg_iterative < avg_single - 0.02:
        print("\n❌ CONCLUSION: Single-round is better, skip iterative.")
    else:
        print("\n➖ CONCLUSION: No significant difference, skip iterative.")


if __name__ == "__main__":
    test_iterative_retrieval(num_samples=30)
