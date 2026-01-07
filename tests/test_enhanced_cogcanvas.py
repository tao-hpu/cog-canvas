#!/usr/bin/env python3
"""
Quick test script for enhanced CogCanvas with:
1. CoT Extraction Prompt
2. Two-stage Retrieval (Hybrid + Reranking)

Tests on 10 LoCoMo samples and compares with baseline.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Configure API
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

from experiments.runner_locomo import LoCoMoExperimentRunner
from experiments.agents.cogcanvas_agent import CogCanvasAgent


def test_baseline():
    """Test baseline CogCanvas (current SOTA)."""
    print("\n" + "="*60)
    print("Testing BASELINE CogCanvas")
    print("="*60)

    config = {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": 10,
        "graph_hops": 3,
    }

    agent = CogCanvasAgent(**config)
    runner = LoCoMoExperimentRunner(
        dataset_path="experiments/data/locomo10.json",
        retain_recent=5,
        rolling_interval=40,
    )

    result = runner.run(
        agent=agent,
        num_samples=10,
        verbose=1,
    )

    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    return result


def test_enhanced():
    """Test enhanced CogCanvas with CoT Extraction + Two-stage Retrieval."""
    print("\n" + "="*60)
    print("Testing ENHANCED CogCanvas")
    print("  - CoT Extraction Prompt (WHO/WHEN/WHAT/WHY)")
    print("  - Two-stage Retrieval (retrieve 20 → rerank to 5)")
    print("="*60)

    config = {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",  # BM25 + Semantic
        "prompt_style": "cot",
        "retrieval_top_k": 5,  # Final top-k after reranking
        "graph_hops": 3,
        "use_reranker": True,  # Enable two-stage retrieval
        "filter_candidate_k": 20,  # Coarse retrieval
    }

    agent = CogCanvasAgent(**config)
    runner = LoCoMoExperimentRunner(
        dataset_path="experiments/data/locomo10.json",
        retain_recent=5,
        rolling_interval=40,
    )

    result = runner.run(
        agent=agent,
        num_samples=10,
        verbose=1,
    )

    print("\n" + "="*60)
    print("ENHANCED RESULTS")
    print("="*60)
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    return result


def main():
    """Run comparison test."""
    print("\n" + "="*70)
    print(" CogCanvas Optimization Test: CoT Extraction + Retrieval Reranking ")
    print("="*70)

    # Test baseline
    baseline_result = test_baseline()

    # Test enhanced
    enhanced_result = test_enhanced()

    # Compare
    print("\n" + "="*70)
    print(" COMPARISON: Baseline vs Enhanced ")
    print("="*70)

    baseline_acc = baseline_result.overall_accuracy
    enhanced_acc = enhanced_result.overall_accuracy

    print(f"\nOverall Accuracy:")
    print(f"  Baseline:  {baseline_acc:.1%}")
    print(f"  Enhanced:  {enhanced_acc:.1%}")
    print(f"  Delta:     {(enhanced_acc - baseline_acc)*100:+.1f}pp")

    print(f"\nBy Category:")
    for cat, name in [(1, "Single-hop"), (2, "Temporal"), (3, "Multi-hop")]:
        baseline_cat = baseline_result.accuracy_by_category(cat)
        enhanced_cat = enhanced_result.accuracy_by_category(cat)
        delta = (enhanced_cat - baseline_cat) * 100
        print(f"  {name:12s}: {baseline_cat:.1%} → {enhanced_cat:.1%} ({delta:+.1f}pp)")

    print("\n" + "="*70)

    # Verdict
    if enhanced_acc > baseline_acc:
        improvement = (enhanced_acc - baseline_acc) * 100
        print(f"✓ SUCCESS: Enhanced version improved by {improvement:.1f}pp!")
    elif enhanced_acc == baseline_acc:
        print("= NEUTRAL: Performance unchanged")
    else:
        decline = (baseline_acc - enhanced_acc) * 100
        print(f"✗ DECLINE: Performance decreased by {decline:.1f}pp")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
