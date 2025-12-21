"""
Test the answer phase with different prompt styles.
Tests CogCanvas multi-artifact fusion and CoT reasoning.

Usage:
    python experiments/test_answer_phase.py [conv_id] [max_turns]

Examples:
    python experiments/test_answer_phase.py locomo_000 30
    python experiments/test_answer_phase.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from cogcanvas import Canvas
from experiments.locomo_adapter import load_locomo, convert_to_eval_format
from experiments.agents.cogcanvas_agent import CogCanvasAgent


def load_conversation(conv_id: str = "locomo_000") -> Dict[str, Any]:
    """Load a specific conversation from LoCoMo."""
    raw_data = load_locomo("experiments/data/locomo10.json")
    conversations = convert_to_eval_format(raw_data)
    for conv in conversations:
        if conv.id == conv_id:
            return {
                "id": conv.id,
                "turns": conv.turns,
                "questions": [
                    {
                        "question": q.question,
                        "answer": q.answer,
                        "category": q.category_name
                    }
                    for q in conv.qa_pairs
                ],
            }
    raise ValueError(f"Conversation {conv_id} not found")


def build_agent(conv: Dict[str, Any], max_turns: int = 30, prompt_style: str = "cot_fusion") -> CogCanvasAgent:
    """Build a CogCanvas agent from conversation."""
    agent = CogCanvasAgent(
        use_real_llm_for_answer=True,
        retrieval_top_k=5,
        enable_graph_expansion=True,
        graph_hops=2,
        use_reranker=True,
        prompt_style=prompt_style,
    )

    print(f"Processing {min(max_turns, len(conv['turns']))} turns...")
    for i, turn in enumerate(conv["turns"][:max_turns]):
        agent.process_turn(turn)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} turns...")

    return agent


def test_single_question(
    agent: CogCanvasAgent,
    question: str,
    ground_truth: str,
    category: str = "unknown",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test a single question and return results."""
    response = agent.answer_question(question)

    result = {
        "question": question,
        "ground_truth": ground_truth,
        "category": category,
        "answer": response.answer,
        "metadata": response.metadata,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{category.upper()}] {question}")
        print(f"{'='*60}")
        print(f"Ground Truth: {ground_truth}")
        print(f"\nModel Answer:\n{response.answer}")
        print(f"\nMetadata: {response.metadata}")

    return result


def run_tests(
    conv_id: str = "locomo_000",
    max_turns: int = 30,
    num_questions: int = 10,
    prompt_styles: List[str] = None,
    custom_questions: List[Dict] = None,
):
    """
    Run answer phase tests.

    Args:
        conv_id: Conversation ID to test
        max_turns: Number of turns to process
        num_questions: Number of questions to test from LoCoMo
        prompt_styles: List of prompt styles to compare (default: all)
        custom_questions: Optional custom questions to test
    """
    if prompt_styles is None:
        prompt_styles = ["cot_fusion", "cot_v2", "cot", "direct"]

    print(f"{'='*60}")
    print(f"ANSWER PHASE TEST: {conv_id}")
    print(f"{'='*60}")

    # Load conversation
    conv = load_conversation(conv_id)
    print(f"Loaded {len(conv['turns'])} turns, {len(conv['questions'])} questions")

    # Get test questions
    test_questions = custom_questions or conv["questions"][:num_questions]

    # Test each prompt style
    for style in prompt_styles:
        print(f"\n{'#'*60}")
        print(f"PROMPT STYLE: {style.upper()}")
        print(f"{'#'*60}")

        # Build agent with this prompt style
        agent = build_agent(conv, max_turns, prompt_style=style)
        print(f"Agent built with {agent._canvas.size} artifacts")

        # Test each question
        results = []
        for q in test_questions:
            result = test_single_question(
                agent,
                q["question"],
                q["answer"],
                q.get("category", "unknown"),
            )
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY ({style})")
        print(f"{'='*60}")
        for r in results:
            gt_words = set(r["ground_truth"].lower().split())
            ans_words = set(r["answer"].lower().split())
            overlap = len(gt_words & ans_words) / len(gt_words) if gt_words else 0
            status = "✓" if overlap >= 0.5 else "✗"
            print(f"{status} [{r['category'][:8]:8s}] {r['question'][:50]}... ({overlap:.0%})")


def interactive_mode(conv_id: str = "locomo_000", max_turns: int = 30):
    """Interactive mode for testing custom questions."""
    print(f"{'='*60}")
    print(f"INTERACTIVE MODE: {conv_id}")
    print(f"{'='*60}")

    conv = load_conversation(conv_id)
    agent = build_agent(conv, max_turns, prompt_style="cot_fusion")
    print(f"Agent ready with {agent._canvas.size} artifacts")

    print("\nEnter questions to test (type 'quit' to exit):\n")

    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            response = agent.answer_question(question)
            print(f"\nAnswer:\n{response.answer}\n")
            print(f"Metadata: {response.metadata}\n")
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


# =============================================================================
# Custom Test Cases - Add your tests here!
# =============================================================================

CUSTOM_TESTS = [
    # Example multi-hop questions
    {
        "question": "Why did Caroline join the LGBTQ support group?",
        "answer": "Because she is transgender and wanted support",
        "category": "multi-hop",
    },
    {
        "question": "When did Caroline attend the support group?",
        "answer": "7 May 2023",
        "category": "temporal",
    },
    {
        "question": "What is Caroline's identity?",
        "answer": "Transgender woman",
        "category": "single-hop",
    },
    # Add more custom tests here...
]


if __name__ == "__main__":
    conv_id = sys.argv[1] if len(sys.argv) > 1 else "locomo_000"
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    # Choose mode
    mode = sys.argv[3] if len(sys.argv) > 3 else "test"

    if mode == "interactive":
        interactive_mode(conv_id, max_turns)
    elif mode == "custom":
        run_tests(
            conv_id=conv_id,
            max_turns=max_turns,
            custom_questions=CUSTOM_TESTS,
            prompt_styles=["cot_fusion"],  # Only test cot_fusion for custom
        )
    else:
        # Default: test with LoCoMo questions, compare all styles
        run_tests(
            conv_id=conv_id,
            max_turns=max_turns,
            num_questions=5,
            prompt_styles=["cot_fusion", "direct"],  # Compare fusion vs direct
        )
