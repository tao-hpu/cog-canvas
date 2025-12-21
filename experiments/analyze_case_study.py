#!/usr/bin/env python3
"""
Analyze LoCoMo results for Case Study and Token Efficiency.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_results(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def find_case_studies(results: dict):
    """Find good case studies from results."""

    success_cases = defaultdict(list)
    failure_cases = defaultdict(list)

    for conv in results.get("conversations", []):
        for q in conv.get("questions", []):
            category = q.get("category_name", "unknown")
            passed = q.get("passed", False)

            case = {
                "question": q.get("question"),
                "ground_truth": q.get("ground_truth"),
                "answer": q.get("answer", "")[:200] + "..." if len(q.get("answer", "")) > 200 else q.get("answer", ""),
                "found_keywords": q.get("found_keywords", []),
                "missing_keywords": q.get("missing_keywords", []),
                "keyword_overlap": q.get("keyword_overlap", 0),
            }

            if passed:
                success_cases[category].append(case)
            else:
                failure_cases[category].append(case)

    return dict(success_cases), dict(failure_cases)


def estimate_token_efficiency():
    """Estimate token consumption for different methods."""

    # Based on typical LoCoMo conversation: ~200 turns, ~50 tokens per turn
    total_context_tokens = 200 * 50  # ~10,000 tokens

    estimates = {
        "Full Context": {
            "strategy": "No compression",
            "tokens_per_query": total_context_tokens,
            "notes": "Infeasible for most LLMs"
        },
        "Summarization": {
            "strategy": "Compress history to summary",
            "tokens_per_query": 2000,  # ~20% of original
            "notes": "Loses specific details"
        },
        "RAG (top-5)": {
            "strategy": "Retrieve top-5 chunks",
            "tokens_per_query": 5 * 300,  # 5 chunks × 300 tokens
            "notes": "No structural relations"
        },
        "GraphRAG": {
            "strategy": "Community summaries",
            "tokens_per_query": 3000,  # Multiple community summaries
            "notes": "Lossy summarization"
        },
        "CogCanvas": {
            "strategy": "Artifacts + Graph expansion",
            "tokens_per_query": 15 * 50 + 10 * 50,  # 15 primary + 10 related
            "notes": "Verbatim grounding preserved"
        },
    }

    return estimates


def main():
    results_path = Path("experiments/results/locomo_boost.json")

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    results = load_results(results_path)

    print("="*60)
    print("CASE STUDY ANALYSIS")
    print("="*60)

    success, failure = find_case_studies(results)

    # Show successful cases
    print("\n### SUCCESSFUL CASES ###\n")
    for category in ["single-hop", "multi-hop", "temporal"]:
        cases = success.get(category, [])
        print(f"\n{category.upper()} ({len(cases)} cases):")
        if cases:
            case = cases[0]  # Show first example
            print(f"  Q: {case['question']}")
            print(f"  Expected: {case['ground_truth']}")
            print(f"  Found: {case['found_keywords']}")
            print(f"  Answer: {case['answer'][:150]}...")

    # Show failure cases
    print("\n\n### FAILURE CASES ###\n")
    for category in ["temporal", "multi-hop", "single-hop"]:
        cases = failure.get(category, [])
        print(f"\n{category.upper()} ({len(cases)} cases):")
        if cases:
            case = cases[0]  # Show first example
            print(f"  Q: {case['question']}")
            print(f"  Expected: {case['ground_truth']}")
            print(f"  Missing: {case['missing_keywords']}")
            print(f"  Answer: {case['answer'][:150]}...")

    # Token efficiency
    print("\n\n" + "="*60)
    print("TOKEN EFFICIENCY ESTIMATES")
    print("="*60)

    estimates = estimate_token_efficiency()

    print(f"\n{'Method':<20} {'Tokens/Query':<15} {'Strategy':<30}")
    print("-"*65)
    for method, data in estimates.items():
        print(f"{method:<20} {data['tokens_per_query']:<15} {data['strategy']:<30}")

    # Save for paper
    output = {
        "case_studies": {
            "success_examples": {k: v[:2] for k, v in success.items()},
            "failure_examples": {k: v[:2] for k, v in failure.items()},
        },
        "token_efficiency": estimates,
        "summary": {
            "total_success": sum(len(v) for v in success.values()),
            "total_failure": sum(len(v) for v in failure.values()),
            "temporal_success_rate": len(success["temporal"]) / (len(success["temporal"]) + len(failure["temporal"]) + 0.001),
        }
    }

    output_path = Path("experiments/results/case_study_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
