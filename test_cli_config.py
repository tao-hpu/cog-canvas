#!/usr/bin/env python3
"""Test script to verify CLI argument parsing."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from experiments.agents.cogcanvas_agent import CogCanvasAgent

print("=" * 60)
print("CLI Argument Simulation Test")
print("=" * 60)

# Simulate the exact logic from runner_locomo.py
test_cases = [
    ("cogcanvas", "Full SOTA"),
    ("cogcanvas-no-temporal", "No Temporal"),
    ("cogcanvas-no-rerank", "No Rerank"),
]

for agent_arg, description in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing: --agent {agent_arg}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    # Replicate exact logic from runner_locomo.py lines 1636-1671
    config = {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": 10,
        "graph_hops": 3,
        "use_reranker": True,
        "reranker_candidate_k": 20,
    }

    # Apply modifications based on agent type
    if agent_arg in ("cogcanvas-nograph", "cogcanvas-no-graph"):
        config["enable_graph_expansion"] = False
    elif agent_arg == "cogcanvas-no-cot":
        config["prompt_style"] = "direct"
    elif agent_arg == "cogcanvas-no-temporal":
        config["enable_temporal_heuristic"] = False
    elif agent_arg == "cogcanvas-no-hybrid":
        config["retrieval_method"] = "semantic"
    elif agent_arg == "cogcanvas-no-rerank":
        config["use_reranker"] = False

    print(f"\nConfig after modification:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create agent
    try:
        agent = CogCanvasAgent(**config)
        print(f"\nAgent name: {agent.name}")
        print(f"✅ Match expected: ", end="")

        # Verify expectations
        if agent_arg == "cogcanvas":
            expected = "CogCanvas(Graph3hop+Time+Hybrid+CoT+Rerank)"
            if agent.name == expected:
                print(f"YES - {expected}")
            else:
                print(f"NO - Expected {expected}, got {agent.name}")
        elif agent_arg == "cogcanvas-no-temporal":
            expected = "CogCanvas(Graph3hop+Hybrid+CoT+Rerank)"
            if agent.name == expected:
                print(f"YES - {expected}")
            else:
                print(f"NO - Expected {expected}, got {agent.name}")
        elif agent_arg == "cogcanvas-no-rerank":
            expected = "CogCanvas(Graph3hop+Time+Hybrid+CoT)"
            if agent.name == expected:
                print(f"YES - {expected}")
            else:
                print(f"NO - Expected {expected}, got {agent.name}")

    except Exception as e:
        print(f"❌ ERROR creating agent: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
