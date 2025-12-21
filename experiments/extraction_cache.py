"""
Extraction Cache Manager for LoCoMo Experiments.

Caches Canvas extraction results to avoid redundant LLM calls.
Agents with the same extraction config can share cached results.

Cache Key is based on:
- conversation_id
- extraction config hash (extractor_model, embedding_model, etc.)

Usage:
    cache = ExtractionCache("experiments/cache/extraction")

    # Check if cached
    if cache.has(conv_id, config):
        canvas_state = cache.load(conv_id, config)
    else:
        # Run extraction...
        cache.save(conv_id, config, canvas_state)
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class ExtractionConfig:
    """Configuration that affects extraction results."""

    extractor_model: str = "gpt-4o-mini"
    embedding_model: str = "bge-large-zh-v1.5"
    enable_temporal_heuristic: bool = True
    reference_threshold: float = 0.5
    causal_threshold: float = 0.45
    enable_vage: bool = False
    use_learned_vage: bool = False
    vage_budget_k: int = 10
    vage_mode: str = "off"  # "off" | "standard" | "chain"

    def to_hash(self) -> str:
        """Generate a short hash for this config."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    @classmethod
    def from_agent_config(cls, agent_config: Dict[str, Any]) -> "ExtractionConfig":
        """Create from CogCanvasAgent config dict."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        return cls(
            extractor_model=agent_config.get("extractor_model") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini"),
            embedding_model=agent_config.get("embedding_model") or os.getenv("EMBEDDING_MODEL", "bge-large-zh-v1.5"),
            enable_temporal_heuristic=agent_config.get("enable_temporal_heuristic", True),
            reference_threshold=agent_config.get("reference_threshold", 0.5),
            causal_threshold=agent_config.get("causal_threshold", 0.45),
            enable_vage=agent_config.get("enable_vage", False),
            use_learned_vage=agent_config.get("use_learned_vage", False),
            vage_budget_k=agent_config.get("vage_budget_k", 10),
        )


class ExtractionCache:
    """
    Manages extraction cache for LoCoMo experiments.

    Cache Structure:
        cache_dir/
            {config_hash}/
                {conv_id}.json
    """

    def __init__(self, cache_dir: str = "experiments/cache/extraction"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, conv_id: str, config: ExtractionConfig) -> Path:
        """Get cache file path for a conversation."""
        config_hash = config.to_hash()
        config_dir = self.cache_dir / config_hash
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / f"{conv_id}.json"

    def has(self, conv_id: str, config: ExtractionConfig) -> bool:
        """Check if extraction cache exists."""
        return self._get_cache_path(conv_id, config).exists()

    def save(
        self,
        conv_id: str,
        config: ExtractionConfig,
        canvas_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save canvas state to cache.

        Args:
            conv_id: Conversation ID
            config: Extraction configuration
            canvas_state: Canvas state dict (from Canvas._save format)
            metadata: Optional metadata (e.g., extraction time, num_objects)
        """
        cache_path = self._get_cache_path(conv_id, config)

        cache_data = {
            "conv_id": conv_id,
            "config": asdict(config),
            "config_hash": config.to_hash(),
            "canvas_state": canvas_state,
            "metadata": metadata or {},
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    def load(self, conv_id: str, config: ExtractionConfig) -> Optional[Dict[str, Any]]:
        """
        Load cached canvas state.

        Returns:
            Canvas state dict, or None if not cached
        """
        cache_path = self._get_cache_path(conv_id, config)

        if not cache_path.exists():
            return None

        with open(cache_path) as f:
            cache_data = json.load(f)

        return cache_data.get("canvas_state")

    def list_cached(self, config: Optional[ExtractionConfig] = None) -> List[str]:
        """
        List all cached conversation IDs.

        Args:
            config: If provided, filter by this config's hash

        Returns:
            List of conversation IDs
        """
        if config:
            config_dir = self.cache_dir / config.to_hash()
            if not config_dir.exists():
                return []
            return [p.stem for p in config_dir.glob("*.json")]

        # List all
        conv_ids = set()
        for config_dir in self.cache_dir.iterdir():
            if config_dir.is_dir():
                for cache_file in config_dir.glob("*.json"):
                    conv_ids.add(cache_file.stem)
        return list(conv_ids)

    def clear(self, config: Optional[ExtractionConfig] = None) -> int:
        """
        Clear cache.

        Args:
            config: If provided, only clear for this config

        Returns:
            Number of files deleted
        """
        import shutil

        count = 0
        if config:
            config_dir = self.cache_dir / config.to_hash()
            if config_dir.exists():
                count = len(list(config_dir.glob("*.json")))
                shutil.rmtree(config_dir)
        else:
            for config_dir in self.cache_dir.iterdir():
                if config_dir.is_dir():
                    count += len(list(config_dir.glob("*.json")))
                    shutil.rmtree(config_dir)
        return count

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_conversations": 0,
            "by_config": {},
        }

        for config_dir in self.cache_dir.iterdir():
            if config_dir.is_dir():
                config_hash = config_dir.name
                conv_count = len(list(config_dir.glob("*.json")))
                stats["by_config"][config_hash] = conv_count
                stats["total_conversations"] += conv_count

        return stats


def get_canvas_state_dict(canvas) -> Dict[str, Any]:
    """
    Extract canvas state as a serializable dict.

    Args:
        canvas: Canvas instance

    Returns:
        Dict that can be used to restore canvas state
    """
    return {
        "turn_counter": canvas._turn_counter,
        "objects": [obj.to_dict() for obj in canvas._objects.values()],
        "graph": canvas._graph.to_dict(),
    }


def restore_canvas_state(canvas, state: Dict[str, Any]) -> None:
    """
    Restore canvas state from a cached dict.

    Args:
        canvas: Canvas instance to restore into
        state: State dict from cache
    """
    from cogcanvas.models import CanvasObject
    from cogcanvas.graph import CanvasGraph

    canvas._turn_counter = state.get("turn_counter", 0)
    canvas._objects = {
        obj_data["id"]: CanvasObject.from_dict(obj_data)
        for obj_data in state.get("objects", [])
    }

    # Restore graph
    graph_data = state.get("graph")
    if graph_data:
        canvas._graph = CanvasGraph.from_dict(graph_data)
    else:
        # Rebuild from objects
        canvas._graph = CanvasGraph()
        for obj in canvas._objects.values():
            canvas._graph.add_node(obj)


# Pre-defined configs for common agent variants
EXTRACTION_CONFIGS = {
    # Group 1: No VAGE (cogcanvas-3hop, cogcanvas-cot-fusion, cogcanvas-cot-v2)
    "no_vage": ExtractionConfig(
        enable_vage=False,
    ),

    # Group 2: VAGE rule-based
    "vage_rule": ExtractionConfig(
        enable_vage=True,
        use_learned_vage=False,
        vage_budget_k=10,
    ),

    # Group 3: VAGE learned
    "vage_learned": ExtractionConfig(
        enable_vage=True,
        use_learned_vage=True,
        vage_budget_k=10,
    ),
}


def get_extraction_config_for_agent(agent_name: str) -> ExtractionConfig:
    """
    Get the extraction config for a given agent name.

    Agents that share the same config can reuse cached extractions.
    """
    if agent_name in ("cogcanvas-vage",):
        return EXTRACTION_CONFIGS["vage_rule"]
    elif agent_name in ("cogcanvas-vage-learned",):
        return EXTRACTION_CONFIGS["vage_learned"]
    else:
        # Most agents use no VAGE
        return EXTRACTION_CONFIGS["no_vage"]
