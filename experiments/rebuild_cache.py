#!/usr/bin/env python
"""
Rebuild Extraction Cache for LoCoMo Experiments.

This script pre-computes and caches the extraction phase for all LoCoMo
conversations, so that subsequent experiments can skip the expensive
LLM extraction calls.

Usage:
    # Rebuild cache for all configs
    python -m experiments.rebuild_cache

    # Rebuild cache for specific config
    python -m experiments.rebuild_cache --config no_vage

    # Force rebuild (clear existing cache first)
    python -m experiments.rebuild_cache --force

    # Rebuild with custom dataset
    python -m experiments.rebuild_cache -d experiments/data/locomo10.json

Cache Groups:
    - no_vage: For cogcanvas-3hop, cogcanvas-cot-fusion, cogcanvas-cot-v2, etc.
    - vage_rule: For cogcanvas-vage
    - vage_learned: For cogcanvas-vage-learned
"""

import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild extraction cache for LoCoMo experiments"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="experiments/data/locomo10.json",
        help="Path to LoCoMo dataset",
    )
    parser.add_argument(
        "--config",
        "-c",
        choices=["no_vage", "vage_rule", "vage_learned", "only_vage", "all"],
        default="all",
        help="Which config to rebuild: no_vage, vage_rule, vage_learned, only_vage (=vage_rule+vage_learned), all (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        default="experiments/cache/extraction",
        help="Cache directory",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=5,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rebuild (clear existing cache)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=None,
        help="Number of conversations to process (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load environment
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
    os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

    # Import after env is loaded
    from experiments.extraction_cache import (
        ExtractionCache,
        ExtractionConfig,
        EXTRACTION_CONFIGS,
        get_canvas_state_dict,
    )
    from experiments.locomo_adapter import load_locomo, convert_to_eval_format
    from experiments.agents.cogcanvas_agent import CogCanvasAgent

    # Load dataset
    print(f"Loading LoCoMo dataset from {args.dataset}...")
    raw_data = load_locomo(args.dataset)
    conversations = convert_to_eval_format(raw_data)
    print(f"Loaded {len(conversations)} conversations")

    if args.samples:
        conversations = conversations[: args.samples]
        print(f"Processing {len(conversations)} conversations")

    # Initialize cache
    cache = ExtractionCache(args.cache_dir)

    # Determine configs to rebuild
    if args.config == "all":
        configs_to_rebuild = list(EXTRACTION_CONFIGS.keys())
    elif args.config == "only_vage":
        configs_to_rebuild = ["vage_rule", "vage_learned"]
    else:
        configs_to_rebuild = [args.config]

    print(f"\nConfigs to rebuild: {configs_to_rebuild}")

    # Force clear if requested
    if args.force:
        for config_name in configs_to_rebuild:
            config = EXTRACTION_CONFIGS[config_name]
            count = cache.clear(config)
            print(f"  Cleared {count} cached entries for {config_name}")

    # Process each config
    for config_name in configs_to_rebuild:
        base_config = EXTRACTION_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Rebuilding cache for: {config_name}")
        print(f"Base config hash: {base_config.to_hash()}")
        print(f"{'='*60}")

        # Create agent config based on extraction config
        agent_config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": base_config.enable_temporal_heuristic,
            "retrieval_method": "hybrid",
            "prompt_style": "cot",
            "retrieval_top_k": 10,
            "graph_hops": 3,
            "reference_threshold": base_config.reference_threshold,
            "causal_threshold": base_config.causal_threshold,
            "enable_vage": base_config.enable_vage,
            "use_learned_vage": base_config.use_learned_vage,
            "vage_budget_k": base_config.vage_budget_k,
        }

        # Get actual config from a test agent to ensure hash matches
        test_agent = CogCanvasAgent(**agent_config)
        config = test_agent.get_extraction_config()
        print(f"Actual config hash from agent: {config.to_hash()}")

        # Check existing cache with actual config
        existing = cache.list_cached(config)
        existing_set = set(existing)
        to_process = [c for c in conversations if c.id not in existing_set]

        if not to_process:
            print(f"All {len(conversations)} conversations already cached. Skipping.")
            continue

        print(f"Already cached: {len(existing)}, to process: {len(to_process)}")

        # Process conversations
        completed = [0]
        lock = threading.Lock()
        start_time = time.time()

        def process_conv(conv):
            try:
                conv_start = time.time()

                # Create fresh agent for each conversation
                agent = CogCanvasAgent(**agent_config)
                agent.reset()

                # Get compression point
                compression_turn = conv.get_compression_point()
                compression_turn = min(compression_turn, len(conv.turns))

                if args.verbose >= 2:
                    with lock:
                        print(f"  [Starting] {conv.id} ({len(conv.turns)} turns, compression@{compression_turn})")

                # Phase 1: Pre-compression turns
                pre_turns = [t for t in conv.turns if t.turn_id <= compression_turn]
                for i, turn in enumerate(pre_turns):
                    agent.process_turn(turn)
                    if args.verbose >= 2 and (i + 1) % 50 == 0:
                        with lock:
                            print(f"    [{conv.id}] Phase 1: {i+1}/{len(pre_turns)} turns processed")

                # Phase 2: Compression
                retain_recent = 5
                retained_turns = [
                    t
                    for t in conv.turns
                    if t.turn_id > compression_turn - retain_recent
                    and t.turn_id <= compression_turn
                ]
                agent.on_compression(retained_turns)

                # Phase 3: Post-compression turns
                post_turns = [t for t in conv.turns if t.turn_id > compression_turn]
                for i, turn in enumerate(post_turns):
                    agent.process_turn(turn)
                    if args.verbose >= 2 and (i + 1) % 50 == 0:
                        with lock:
                            print(f"    [{conv.id}] Phase 3: {i+1}/{len(post_turns)} turns processed")

                # Get canvas state
                canvas_state = get_canvas_state_dict(agent._canvas)

                # Save to cache
                cache.save(
                    conv.id,
                    config,
                    canvas_state,
                    metadata={
                        "num_objects": len(canvas_state.get("objects", [])),
                        "num_turns": len(conv.turns),
                        "compression_turn": compression_turn,
                    },
                )

                elapsed = time.time() - conv_start
                with lock:
                    completed[0] += 1
                    if args.verbose >= 1:
                        print(
                            f"  [{completed[0]}/{len(to_process)}] {conv.id} - "
                            f"{len(canvas_state.get('objects', []))} objects ({elapsed:.1f}s)"
                        )

                return conv.id, True, None

            except Exception as e:
                with lock:
                    completed[0] += 1
                    print(f"  [{completed[0]}/{len(to_process)}] {conv.id} - ERROR: {e}")
                import traceback
                traceback.print_exc()
                return conv.id, False, str(e)

        # Run in parallel
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_conv, c): c.id for c in to_process}
            for future in as_completed(futures):
                results.append(future.result())

        # Summary
        elapsed = time.time() - start_time
        success = sum(1 for _, ok, _ in results if ok)
        failed = sum(1 for _, ok, _ in results if not ok)

        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"  Success: {success}, Failed: {failed}")

    # Final cache stats
    print(f"\n{'='*60}")
    print("CACHE STATISTICS")
    print(f"{'='*60}")
    stats = cache.stats()
    print(f"Total cached conversations: {stats['total_conversations']}")
    for config_hash, count in stats["by_config"].items():
        print(f"  {config_hash}: {count} conversations")


if __name__ == "__main__":
    main()
