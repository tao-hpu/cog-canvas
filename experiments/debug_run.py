"""
Debug script for CogCanvas evaluation.

Runs a small batch and prints detailed failure analysis for each failed question.
"""

import json
import time
from pathlib import Path

from experiments.data_gen import EvaluationDataset
from experiments.runner import score_answer
from experiments.agents.cogcanvas_agent import CogCanvasAgent


def debug_run(num_samples: int = 5):
    """Run evaluation with detailed debugging."""

    # Load dataset
    print("Loading dataset...")
    dataset = EvaluationDataset.load("experiments/data/eval_set.json")
    conversations = dataset.conversations[:num_samples]
    print(f"Loaded {len(conversations)} conversations for debugging\n")

    # Create agent
    agent = CogCanvasAgent()
    print(f"Agent: {agent.name}\n")

    compression_turn = 40
    retain_recent = 5

    total_passed = 0
    total_failed = 0

    for conv_idx, conv in enumerate(conversations):
        print(f"\n{'='*80}")
        print(f"[Conversation {conv_idx + 1}/{len(conversations)}] ID: {conv.id}")
        print(f"{'='*80}")

        agent.reset()

        # Phase 1: Process turns up to compression
        print(f"\nProcessing {compression_turn} turns...")
        for turn in conv.turns:
            if turn.turn_id <= compression_turn:
                agent.process_turn(turn)

        # Print canvas stats after extraction
        stats = agent.get_canvas_stats()
        print(f"\nCanvas after extraction:")
        print(f"  Total objects: {stats.get('total_objects', 0)}")
        print(f"  By type: {stats.get('by_type', {})}")

        # Phase 2: Simulate compression
        retained_turns = [
            t for t in conv.turns
            if t.turn_id > compression_turn - retain_recent
            and t.turn_id <= compression_turn
        ]
        agent.on_compression(retained_turns)

        # Phase 3: Ask recall questions
        print(f"\n--- Testing {len(conv.planted_facts)} planted facts ---")

        for fact in conv.planted_facts:
            print(f"\n[FACT] Type: {fact.fact_type}, Planted at turn: {fact.turn_id}")
            print(f"  Question: {fact.test_question}")
            print(f"  Expected: {fact.ground_truth}")

            # Get detailed retrieval info
            retrieval_result = agent._canvas.retrieve(
                query=fact.test_question,
                top_k=5,
                method="semantic",
                include_related=True,
            )

            # Show what was retrieved
            print(f"\n  Retrieved {len(retrieval_result.objects)} objects:")
            for i, obj in enumerate(retrieval_result.objects[:3]):  # Top 3
                score = retrieval_result.scores[i] if i < len(retrieval_result.scores) else 0
                print(f"    [{i+1}] Score: {score:.3f} | Type: {obj.type.value}")
                print(f"        Content: {obj.content[:100]}...")
                if obj.quote:
                    print(f"        Quote: {obj.quote[:100]}...")

            # Get answer
            response = agent.answer_question(fact.test_question)

            # Score
            score_result = score_answer(response.answer, fact.ground_truth)

            print(f"\n  Answer: {response.answer[:200]}...")
            print(f"  Exact Match: {score_result.exact_match}")
            print(f"  Fuzzy Score: {score_result.fuzzy_score:.1f}")
            print(f"  PASSED: {score_result.passed}")

            if score_result.passed:
                total_passed += 1
                print("  ✓ SUCCESS")
            else:
                total_failed += 1
                print("  ✗ FAILED")
                print("\n  === FAILURE ANALYSIS ===")

                # Check if the expected answer exists in any canvas object
                found_in_canvas = False
                found_in_retrieved = False

                # Check all canvas objects
                for obj in agent._canvas._objects.values():
                    if fact.ground_truth.lower() in obj.content.lower() or \
                       (obj.quote and fact.ground_truth.lower() in obj.quote.lower()):
                        found_in_canvas = True
                        print(f"  Ground truth found in canvas object: {obj.type.value}")
                        print(f"    Content: {obj.content[:150]}...")
                        break

                # Check if it's in retrieved objects
                for i, obj in enumerate(retrieval_result.objects):
                    if fact.ground_truth.lower() in obj.content.lower() or \
                       (obj.quote and fact.ground_truth.lower() in obj.quote.lower()):
                        found_in_retrieved = True
                        score = retrieval_result.scores[i] if i < len(retrieval_result.scores) else 0
                        print(f"  Ground truth in RETRIEVED object #{i+1} (Score: {score:.3f})")
                        break

                if not found_in_canvas:
                    print("  ⚠️  Ground truth NOT FOUND in any canvas object!")
                    print("  → This is an EXTRACTION failure")
                elif found_in_retrieved:
                    print("  ⚠️  Ground truth is retrieved but not in top-1")
                    print("  → This is a RANKING failure - need to use top-k not just top-1")
                else:
                    print("  ⚠️  Ground truth exists in canvas but not retrieved")
                    print("  → This is a RETRIEVAL failure")

        # Summary for this conversation
        conv_passed = sum(1 for f in conv.planted_facts
                        if score_answer(agent.answer_question(f.test_question).answer, f.ground_truth).passed)
        print(f"\n--- Conversation Summary: {conv_passed}/{len(conv.planted_facts)} passed ---")

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Recall Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")


if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    debug_run(num)
