"""
Debug multi-hop performance drop in CogCanvas v3.

Compare v2 (baseline) vs v3 (BM25) to identify what went wrong.
"""

import json
from pathlib import Path

def load_results(filepath):
    """Load evaluation results."""
    with open(filepath) as f:
        data = json.load(f)
    return data['conversations']

def extract_multihop_failures(results, version_name):
    """Extract failed multi-hop questions."""
    failures = []

    for conv in results:
        conv_id = conv['id']
        for q_idx, question in enumerate(conv['questions']):
            # Only multi-hop questions (category 3)
            if question.get('category') != 3:
                continue

            if not question.get('passed', False):
                failures.append({
                    'conversation_id': conv_id,
                    'question_idx': q_idx,
                    'question': question['question'],
                    'answer': question.get('answer', ''),
                    'ground_truth': question.get('ground_truth', ''),
                    'version': version_name
                })

    return failures

def compare_versions(v2_file, v3_file):
    """Compare v2 vs v3 multi-hop performance."""
    print("=" * 80)
    print("Multi-Hop Performance Diagnosis: v2 (baseline) vs v3 (BM25)")
    print("=" * 80)

    # Load results
    v2_results = load_results(v2_file)
    v3_results = load_results(v3_file)

    # Extract failures
    v2_failures = extract_multihop_failures(v2_results, "v2_baseline")
    v3_failures = extract_multihop_failures(v3_results, "v3_bm25")

    print(f"\nv2 (baseline) multi-hop failures: {len(v2_failures)}")
    print(f"v3 (BM25) multi-hop failures: {len(v3_failures)}")
    print(f"Difference: {len(v3_failures) - len(v2_failures)} more failures in v3")

    # Build lookup by (conv_id, q_idx)
    v2_failed_keys = {(f['conversation_id'], f['question_idx']) for f in v2_failures}
    v3_failed_keys = {(f['conversation_id'], f['question_idx']) for f in v3_failures}

    # Find new failures (v3 failed but v2 passed)
    new_failures = v3_failed_keys - v2_failed_keys
    # Find fixed cases (v2 failed but v3 passed)
    fixed_cases = v2_failed_keys - v3_failed_keys

    print(f"\n{'='*80}")
    print(f"New Failures (v3 broke): {len(new_failures)}")
    print(f"Fixed Cases (v3 fixed): {len(fixed_cases)}")
    print(f"{'='*80}")

    # Show new failures in detail
    if new_failures:
        print("\n=== NEW FAILURES (v3 broke these) ===\n")
        for i, (conv_id, q_idx) in enumerate(list(new_failures)[:5], 1):
            # Find the failure in v3
            failure = next(f for f in v3_failures if f['conversation_id'] == conv_id and f['question_idx'] == q_idx)
            print(f"Case {i}: {conv_id} Q{q_idx}")
            print(f"  Question: {failure['question']}")
            print(f"  Ground Truth: {failure['ground_truth']}")
            print(f"  v3 Answer: {failure['answer'][:200]}...")
            print()

    # Show fixed cases
    if fixed_cases:
        print("\n=== FIXED CASES (v3 fixed these) ===\n")
        for i, (conv_id, q_idx) in enumerate(list(fixed_cases)[:5], 1):
            # Find the failure in v2
            failure = next(f for f in v2_failures if f['conversation_id'] == conv_id and f['question_idx'] == q_idx)
            print(f"Case {i}: {conv_id} Q{q_idx}")
            print(f"  Question: {failure['question']}")
            print(f"  Ground Truth: {failure['ground_truth']}")
            print()

    # Hypothesis testing
    print("\n" + "=" * 80)
    print("HYPOTHESIS ANALYSIS")
    print("=" * 80)

    net_change = len(fixed_cases) - len(new_failures)
    if net_change > 0:
        print(f"✅ Net positive: v3 fixed {len(fixed_cases)} but broke {len(new_failures)}")
        print("   → BM25 is helping overall, but needs refinement for multi-hop")
    elif net_change < 0:
        print(f"❌ Net negative: v3 fixed {len(fixed_cases)} but broke {len(new_failures)}")
        print("   → BM25 is hurting multi-hop performance")
        print("\nPossible causes:")
        print("   1. BM25 weight too high (0.3) → introduces keyword noise")
        print("   2. 5-hop expansion too aggressive → retrieves irrelevant nodes")
        print("   3. Multi-hop queries need more semantic signal, less keyword matching")
        print("\nRecommended fixes:")
        print("   1. Reduce BM25 weight: 0.3 → 0.15 or 0.2")
        print("   2. Reduce graph hops: 5 → 3")
        print("   3. Add category-aware retrieval: use pure semantic for multi-hop")
    else:
        print("⚖️ Net neutral: same number of fixes and breaks")
        print("   → Need larger sample size to determine impact")

if __name__ == "__main__":
    base_dir = Path(__file__).parent / "results"

    # Compare with baseline
    # You need to specify the baseline file (v2) and new file (v3)
    v2_file = base_dir / "rolling_locomo_cogcanvas.json"  # Baseline
    v3_file = base_dir / "sample_rolling_cogcanvas_3.json"  # New v3 with BM25

    if not v2_file.exists():
        print(f"Error: Baseline file not found: {v2_file}")
        print("Please specify the correct baseline file path")
    elif not v3_file.exists():
        print(f"Error: v3 file not found: {v3_file}")
        print("Please specify the correct v3 result file path")
    else:
        compare_versions(v2_file, v3_file)
