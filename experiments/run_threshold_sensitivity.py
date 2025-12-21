#!/usr/bin/env python3
"""
Threshold Sensitivity Analysis for CogCanvas

Runs Multi-hop experiments with different threshold configurations.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env file from the project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY', '')
os.environ['OPENAI_API_BASE'] = os.getenv('API_BASE', '')

# Threshold configurations to test
THRESHOLD_CONFIGS = [
    {"name": "low", "reference_threshold": 0.3, "causal_threshold": 0.25},
    {"name": "default", "reference_threshold": 0.5, "causal_threshold": 0.45},
    {"name": "high", "reference_threshold": 0.7, "causal_threshold": 0.6},
    {"name": "very_high", "reference_threshold": 0.8, "causal_threshold": 0.7},
]


def run_threshold_experiment(config: dict, dataset_path: str, samples: int = 10, workers: int = 5):
    """Run a single threshold configuration experiment."""
    from experiments.runner_multihop import MultiHopExperimentRunner
    from experiments.agents.cogcanvas_agent import CogCanvasAgent

    name = config["name"]
    ref_thresh = config["reference_threshold"]
    causal_thresh = config["causal_threshold"]

    print(f"\n{'='*60}")
    print(f"Running threshold config: {name}")
    print(f"  reference_threshold: {ref_thresh}")
    print(f"  causal_threshold: {causal_thresh}")
    print(f"{'='*60}\n")

    # Create agent factory with custom thresholds
    agent_config = {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "use_llm_filter": False,
        "retrieval_top_k": 10,
        "reference_threshold": ref_thresh,
        "causal_threshold": causal_thresh,
    }

    agent_factory = lambda: CogCanvasAgent(**agent_config)
    agent = agent_factory()

    # Create runner with dataset
    runner = MultiHopExperimentRunner(
        dataset_path=dataset_path,
        compression_turn=40,
    )

    # Run experiment
    result = runner.run(
        agent=agent,
        max_workers=workers,
        agent_factory=agent_factory,
        verbose=True,
        num_samples=samples,
    )

    # Save result
    output_path = Path(f"experiments/results/threshold_sensitivity_{name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    summary = result.summary()
    print(f"Pass Rate: {summary.get('overall_pass_rate', 'N/A')}")

    return {
        "config": name,
        "reference_threshold": ref_thresh,
        "causal_threshold": causal_thresh,
        "pass_rate": summary.get("overall_pass_rate", "N/A"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Threshold Sensitivity Analysis")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per config")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--config", type=str, default=None, help="Run specific config (low/default/high/very_high)")
    parser.add_argument("--dataset", type=str, default="experiments/data/multihop_eval.json", help="Dataset path")
    args = parser.parse_args()

    results = []

    if args.config:
        # Run single config
        config = next((c for c in THRESHOLD_CONFIGS if c["name"] == args.config), None)
        if config is None:
            print(f"Unknown config: {args.config}")
            print(f"Available: {[c['name'] for c in THRESHOLD_CONFIGS]}")
            sys.exit(1)
        result = run_threshold_experiment(config, args.dataset, args.samples, args.workers)
        results.append(result)
    else:
        # Run all configs
        for config in THRESHOLD_CONFIGS:
            result = run_threshold_experiment(config, args.dataset, args.samples, args.workers)
            results.append(result)

    # Print summary
    print("\n" + "="*60)
    print("THRESHOLD SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Config':<12} {'Ref Thresh':<12} {'Causal Thresh':<14} {'Pass Rate':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['config']:<12} {r['reference_threshold']:<12} {r['causal_threshold']:<14} {r['pass_rate']:<10}")

    # Save summary
    summary_path = Path("experiments/results/threshold_sensitivity_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "samples": args.samples,
            "results": results,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
