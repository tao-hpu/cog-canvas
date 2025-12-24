"""
Analyze multi-hop failures: cases where CogCanvas fails but RAG succeeds.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_results(filepath):
    """Load evaluation results."""
    with open(filepath) as f:
        data = json.load(f)
    return data['conversations']

def extract_multihop_cases(cogcanvas_results, rag_results):
    """
    Extract multi-hop cases where CogCanvas fails but RAG succeeds.
    Returns: list of failure cases
    """
    failures = []

    # Build a lookup for RAG results
    rag_lookup = {}
    for conv in rag_results:
        conv_id = conv['id']
        for q_idx, question in enumerate(conv['questions']):
            # Check if this is a multi-hop question (category 3)
            if question.get('category') != 3:
                continue

            key = (conv_id, q_idx)
            rag_lookup[key] = {
                'question': question['question'],
                'answer': question.get('answer', ''),
                'passed': question.get('passed', False),
                'ground_truth': question.get('ground_truth', '')
            }

    # Compare with CogCanvas results
    for conv in cogcanvas_results:
        conv_id = conv['id']
        for q_idx, question in enumerate(conv['questions']):
            # Check if this is a multi-hop question (category 3)
            if question.get('category') != 3:
                continue

            key = (conv_id, q_idx)
            if key not in rag_lookup:
                continue

            cog_passed = question.get('passed', False)
            rag_passed = rag_lookup[key]['passed']

            # We want cases where CogCanvas fails but RAG succeeds
            if not cog_passed and rag_passed:
                failures.append({
                    'conversation_id': conv_id,
                    'question_idx': q_idx,
                    'question': question['question'],
                    'cog_answer': question.get('answer', ''),
                    'rag_answer': rag_lookup[key]['answer'],
                    'ground_truth': question.get('ground_truth', ''),
                    'cog_passed': cog_passed,
                    'rag_passed': rag_passed
                })

    return failures

def analyze_failures(failures):
    """Print detailed analysis of failure cases."""
    print(f"\n{'='*80}")
    print(f"Found {len(failures)} multi-hop cases where CogCanvas failed but RAG succeeded")
    print(f"{'='*80}\n")

    for i, case in enumerate(failures[:10], 1):  # Show first 10 cases
        print(f"\n{'='*80}")
        print(f"Case {i}/{min(10, len(failures))}")
        print(f"{'='*80}")
        print(f"Conversation ID: {case['conversation_id']}")
        print(f"Question Index: {case['question_idx']}")
        print(f"\nQuestion:")
        print(f"  {case['question']}")
        print(f"\nGround Truth:")
        print(f"  {case['ground_truth']}")
        print(f"\nRAG Answer (✓):")
        print(f"  {case['rag_answer'][:200]}{'...' if len(case['rag_answer']) > 200 else ''}")
        print(f"\nCogCanvas Answer (✗):")
        print(f"  {case['cog_answer'][:200]}{'...' if len(case['cog_answer']) > 200 else ''}")
        print()

    if len(failures) > 10:
        print(f"\n... and {len(failures) - 10} more cases\n")

    # Statistics
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    print(f"Total multi-hop failures (CogCanvas wrong, RAG right): {len(failures)}")

    # Count conversations affected
    conv_ids = set(case['conversation_id'] for case in failures)
    print(f"Number of conversations affected: {len(conv_ids)}")
    if len(conv_ids) > 0:
        print(f"Average failures per affected conversation: {len(failures) / len(conv_ids):.2f}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent / "results"

    print("Loading results...")
    cogcanvas_results = load_results(base_dir / "rolling_locomo_cogcanvas.json")
    rag_results = load_results(base_dir / "rolling_locomo_rag.json")

    print(f"CogCanvas: {len(cogcanvas_results)} conversations")
    print(f"RAG: {len(rag_results)} conversations")

    failures = extract_multihop_cases(cogcanvas_results, rag_results)
    analyze_failures(failures)

    # Save detailed failures to JSON for further analysis
    output_file = base_dir / "multihop_failure_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed failures saved to: {output_file}")
