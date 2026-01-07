#!/usr/bin/env python3
"""Test script to verify agent configurations."""

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
print("Agent Configuration Verification")
print("=" * 60)

# Test configurations
configs = {
    "cogcanvas (Full SOTA)": {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": 10,
        "graph_hops": 3,
        "use_reranker": True,
        "reranker_candidate_k": 20,
    },
    "cogcanvas-no-temporal": {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": False,  # DISABLED
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": 10,
        "graph_hops": 3,
        "use_reranker": True,
        "reranker_candidate_k": 20,
    },
    "cogcanvas-no-rerank": {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": 10,
        "graph_hops": 3,
        "use_reranker": False,  # DISABLED
    },
    "cogcanvas-no-graph": {
        "enable_graph_expansion": False,  # DISABLED
        "enable_temporal_heuristic": True,
        "retrieval_method": "hybrid",
        "prompt_style": "cot",
        "retrieval_top_k": 10,
        "graph_hops": 3,
        "use_reranker": True,
    },
    "cogcanvas-minimal": {
        "enable_graph_expansion": True,
        "enable_temporal_heuristic": False,
        "retrieval_method": "semantic",
        "prompt_style": "direct",
        "retrieval_top_k": 10,
        "graph_hops": 1,
        "use_reranker": False,
    },
}

for config_name, config in configs.items():
    try:
        agent = CogCanvasAgent(**config)
        print(f"\n{config_name:30s} -> {agent.name}")
        print(f"  enable_temporal_heuristic: {agent.enable_temporal_heuristic}")
        print(f"  use_reranker: {agent.use_reranker}")
        print(f"  enable_graph_expansion: {agent.enable_graph_expansion}")
        print(f"  retrieval_method: {agent.retrieval_method}")
        print(f"  prompt_style: {agent.prompt_style}")
        print(f"  graph_hops: {agent.graph_hops}")
    except Exception as e:
        print(f"\n{config_name:30s} -> ERROR: {e}")

print("\n" + "=" * 60)
