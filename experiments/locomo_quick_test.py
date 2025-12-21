"""
Quick test script for LoCoMo adapter and runner.

Usage:
    python -m experiments.test_locomo
"""

from experiments.locomo_adapter import load_locomo, convert_to_eval_format, verify_evidence_mapping
from experiments.runner_locomo import LoCoMoExperimentRunner
from experiments.agents.native_agent import NativeAgent


def test_adapter():
    """Test LoCoMo adapter."""
    print("="*60)
    print("Testing LoCoMo Adapter")
    print("="*60)

    # Load data
    raw_data = load_locomo("experiments/data/locomo10.json")
    print(f"\nLoaded {len(raw_data)} raw conversations")

    # Convert
    conversations = convert_to_eval_format(raw_data)
    print(f"Converted {len(conversations)} conversations")

    # Show first conversation stats
    conv = conversations[0]
    print(f"\nFirst conversation ({conv.id}):")
    print(f"  Speakers: {conv.speaker_a} and {conv.speaker_b}")
    print(f"  Turns: {len(conv.turns)}")
    print(f"  QA pairs: {len(conv.qa_pairs)}")
    print(f"  Compression point: {conv.get_compression_point()}")

    # Show category breakdown
    print(f"\n  Category breakdown:")
    for cat in [1, 2, 3]:
        qa_list = conv.get_qa_by_category(cat)
        cat_name = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop"}[cat]
        print(f"    {cat_name}: {len(qa_list)} questions")

    # Verify evidence mapping
    print(f"\n  Evidence mapping:")
    stats = verify_evidence_mapping(conv)
    print(f"    Mapped: {stats['mapped_refs']}/{stats['total_evidence_refs']} ({stats['mapping_rate']:.1%})")

    # Show sample QA
    print(f"\n  Sample QA pairs:")
    for i, qa in enumerate(conv.qa_pairs[:3]):
        print(f"    Q{i+1}: {qa.question[:50]}...")
        print(f"        A: {qa.answer}")
        print(f"        Category: {qa.category_name}")
        print(f"        Evidence: {qa.evidence}")

    return conversations


def test_runner(conversations):
    """Test LoCoMo runner."""
    print("\n" + "="*60)
    print("Testing LoCoMo Runner")
    print("="*60)

    # Create runner
    runner = LoCoMoExperimentRunner(
        dataset_path="experiments/data/locomo10.json",
        retain_recent=5
    )

    # Create agent
    agent = NativeAgent(retain_recent=5)

    # Run on 1 conversation with 5 questions
    result = runner.run(
        agent=agent,
        num_samples=1,
        max_questions_per_conv=5,
        verbose=True
    )

    print(f"\nTest Results:")
    print(f"  Conversations tested: {len(result.conversation_results)}")
    print(f"  Questions per conversation: {len(result.conversation_results[0].question_results)}")
    print(f"  Accuracy: {result.overall_accuracy:.1%}")

    return result


def main():
    """Run all tests."""
    print("LoCoMo Benchmark Test Suite\n")

    # Test adapter
    conversations = test_adapter()

    # Test runner
    result = test_runner(conversations)

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Convert dataset: python -m experiments.locomo_adapter")
    print("2. Run full evaluation: python -m experiments.runner_locomo --agent cogcanvas")
    print("3. Run with all agents: bash experiments/scripts/run_locomo_all.sh")


if __name__ == "__main__":
    main()
