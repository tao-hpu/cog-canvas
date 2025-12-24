#!/usr/bin/env python3
"""
Super quick test - 3 samples only to verify improvements work.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

from experiments.runner_locomo import LoCoMoExperimentRunner
from experiments.agents.cogcanvas_agent import CogCanvasAgent


def quick_test():
    """Quick sanity check on 3 samples."""
    print("\n" + "="*60)
    print("QUICK TEST: Enhanced CogCanvas (3 samples)")
    print("="*60)

    config = {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": 5,
        "graph_hops": 3,
        "use_reranker": True,
        "filter_candidate_k": 20,
    }

    agent = CogCanvasAgent(**config)
    runner = LoCoMoExperimentRunner(
        dataset_path="experiments/data/locomo10.json",
        retain_recent=5,
        rolling_interval=40,
    )

    result = runner.run(
        agent=agent,
        num_samples=3,
        verbose=2,
    )

    print("\n" + "="*60)
    print("RESULTS (3 samples)")
    print("="*60)
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    quick_test()
