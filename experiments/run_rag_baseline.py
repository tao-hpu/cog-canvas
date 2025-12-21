#!/usr/bin/env python3
"""
RAG Baseline Experiments with Different Configurations

Tests different RAG hyperparameters to prove CogCanvas advantage is not due to weak RAG baseline.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env file from the project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY', '')
os.environ['OPENAI_API_BASE'] = os.getenv('API_BASE', '')

# RAG configurations to test
RAG_CONFIGS = [
    {"name": "rag_small", "chunk_size": 256, "top_k": 5, "overlap": 50},
    {"name": "rag_default", "chunk_size": 512, "top_k": 5, "overlap": 100},
    {"name": "rag_large", "chunk_size": 1024, "top_k": 5, "overlap": 200},
    {"name": "rag_topk10", "chunk_size": 512, "top_k": 10, "overlap": 100},
]


def run_rag_experiment(config: dict, dataset_path: str, samples: int = 10, workers: int = 5):
    """Run a single RAG configuration experiment."""
    from experiments.runner_multihop import MultiHopExperimentRunner
    from experiments.agents.rag_agent import RagAgent

    name = config["name"]
    chunk_size = config["chunk_size"]
    top_k = config["top_k"]
    overlap = config["overlap"]

    print(f"\n{'='*60}")
    print(f"Running RAG config: {name}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  top_k: {top_k}")
    print(f"  overlap: {overlap}")
    print(f"{'='*60}\n")

    # Create agent factory with custom RAG params
    agent_factory = lambda: RagAgent(
        chunk_size=chunk_size,
        top_k=top_k,
        overlap=overlap,
        retain_recent=5,
    )
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
    output_path = Path(f"experiments/results/rag_baseline_{name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    summary = result.summary()
    print(f"Pass Rate: {summary.get('overall_pass_rate', 'N/A')}")

    return {
        "config": name,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "overlap": overlap,
        "pass_rate": summary.get("overall_pass_rate", "N/A"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Baseline Experiments")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per config")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--config", type=str, default=None, help="Run specific config")
    parser.add_argument("--dataset", type=str, default="experiments/data/multihop_eval.json", help="Dataset path")
    args = parser.parse_args()

    results = []

    if args.config:
        # Run single config
        config = next((c for c in RAG_CONFIGS if c["name"] == args.config), None)
        if config is None:
            print(f"Unknown config: {args.config}")
            print(f"Available: {[c['name'] for c in RAG_CONFIGS]}")
            return
        result = run_rag_experiment(config, args.dataset, args.samples, args.workers)
        results.append(result)
    else:
        # Run all configs
        for config in RAG_CONFIGS:
            result = run_rag_experiment(config, args.dataset, args.samples, args.workers)
            results.append(result)

    # Print summary
    print("\n" + "="*60)
    print("RAG BASELINE EXPERIMENTS SUMMARY")
    print("="*60)
    print(f"{'Config':<15} {'Chunk':<8} {'Top-K':<8} {'Overlap':<10} {'Pass Rate':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['config']:<15} {r['chunk_size']:<8} {r['top_k']:<8} {r['overlap']:<10} {r['pass_rate']:<10}")

    # Compare with CogCanvas default (64.0%)
    print("\n" + "-"*60)
    print("CogCanvas Default: 64.0% Pass Rate")
    print("-"*60)

    # Save summary
    summary_path = Path("experiments/results/rag_baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "samples": args.samples,
            "results": results,
            "cogcanvas_reference": "64.0%",
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
