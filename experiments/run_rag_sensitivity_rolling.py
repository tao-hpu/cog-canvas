#!/usr/bin/env python3
"""
RAG hyperparameter sensitivity under the MAIN rolling protocol.

Replaces the retired Dec-20 pilot sweep (run_rag_baseline.py), which used a
pre-rolling protocol (10 conversations, no rolling_interval) whose scores are
not comparable to the paper's main tables. This sweep runs the same grid under
the exact protocol of rolling_multihop_rag.json: 50 conversations,
compression_turn=40, retain_recent=5, rolling_interval=40.

The 512/k10 cell reproduces the main-table RAG configuration as a harness check.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY', '')
os.environ['OPENAI_API_BASE'] = os.getenv('API_BASE', '')

RAG_CONFIGS = [
    {"name": "256_k5",   "chunk_size": 256,  "top_k": 5,  "overlap": 50},
    {"name": "512_k5",   "chunk_size": 512,  "top_k": 5,  "overlap": 100},
    {"name": "1024_k5",  "chunk_size": 1024, "top_k": 5,  "overlap": 200},
    {"name": "256_k10",  "chunk_size": 256,  "top_k": 10, "overlap": 50},
    {"name": "512_k10",  "chunk_size": 512,  "top_k": 10, "overlap": 100},  # main-table config
    {"name": "1024_k10", "chunk_size": 1024, "top_k": 10, "overlap": 200},
]


def run_config(config: dict, dataset_path: str, samples: int, workers: int):
    from experiments.runner_multihop import MultiHopExperimentRunner
    from experiments.agents.rag_agent import RagAgent

    print(f"\n{'='*60}\nRAG rolling sweep cell: {config['name']} "
          f"(chunk={config['chunk_size']}, k={config['top_k']}, overlap={config['overlap']})\n{'='*60}")

    agent_factory = lambda: RagAgent(
        chunk_size=config["chunk_size"],
        top_k=config["top_k"],
        overlap=config["overlap"],
        retain_recent=5,
    )
    runner = MultiHopExperimentRunner(
        dataset_path=dataset_path,
        compression_turn=40,
        rolling_interval=40,
    )
    result = runner.run(
        agent=agent_factory(),
        max_workers=workers,
        agent_factory=agent_factory,
        verbose=True,
        num_samples=samples,
    )

    output_path = Path(f"experiments/results/rag_sensitivity_rolling_{config['name']}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    summary = result.summary()
    print(f"[{config['name']}] pass rate: {summary.get('overall_pass_rate', 'N/A')}  -> {output_path}")
    return {"name": config['name'], **{k: config[k] for k in ('chunk_size', 'top_k', 'overlap')},
            "pass_rate": summary.get("overall_pass_rate")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--only", default=None, help="comma-separated cell names to run")
    parser.add_argument("--dataset", default="experiments/data/multihop_eval.json")
    args = parser.parse_args()

    configs = RAG_CONFIGS
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in RAG_CONFIGS if c["name"] in wanted]

    rows = []
    for config in configs:
        try:
            rows.append(run_config(config, args.dataset, args.samples, args.workers))
        except Exception as e:
            print(f"[{config['name']}] FAILED: {e}")
            rows.append({"name": config['name'], "error": str(e)})

    summary_path = Path("experiments/results/rag_sensitivity_rolling_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "protocol": {"samples": args.samples, "compression_turn": 40,
                                "retain_recent": 5, "rolling_interval": 40},
                   "cells": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}\nSweep summary -> {summary_path}")
    for r in rows:
        print(f"  {r.get('name'):>9}: {r.get('pass_rate', r.get('error'))}")


if __name__ == "__main__":
    main()
