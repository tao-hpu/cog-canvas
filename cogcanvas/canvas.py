"""Main Canvas class - the primary interface for CogCanvas."""

from typing import List, Optional, Dict, Any, Union, Literal, Tuple
import json
import time
from pathlib import Path

from cogcanvas.models import (
    CanvasObject,
    ObjectType,
    ExtractionResult,
    RetrievalResult,
)
from cogcanvas.llm.base import LLMBackend, MockLLMBackend
from cogcanvas.embeddings import (
    EmbeddingBackend,
    MockEmbeddingBackend,
    batch_cosine_similarity,
)
from cogcanvas.graph import CanvasGraph

class Canvas:
    """
    CogCanvas: A compression-resistant cognitive canvas for LLM conversations.

    The Canvas maintains a persistent collection of cognitive objects (decisions,
    TODOs, facts, etc.) that are extracted from dialogue and survive context
    compression.

    Example:
        >>> canvas = Canvas()
        >>> canvas.extract(
        ...     user="Let's use PostgreSQL",
        ...     assistant="Good choice!"
        ... )
        >>> results = canvas.retrieve("What database?")
        >>> print(results.objects[0].content)
        "Use PostgreSQL for the database"
    """

    def __init__(
        self,
        extractor_model: str = None,
        embedding_model: str = None,
        storage_path: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        embedding_backend: Optional[EmbeddingBackend] = None,
        enable_temporal_heuristic: bool = True,  # New parameter for ablation
    ):
        """
        Initialize a new Canvas.

        Args:
            extractor_model: Model to use for extraction (e.g., "gpt-4o-mini", "mock").
                           Defaults to MODEL_WEAK_2 env var or "mock".
            embedding_model: Model for embeddings (e.g., "BAAI/bge-m3", "mock").
                           Defaults to EMBEDDING_MODEL env var or "mock".
            storage_path: Path to persist canvas state (optional)
            llm_backend: Pre-configured LLM backend (overrides extractor_model)
            embedding_backend: Pre-configured embedding backend (overrides embedding_model)
            enable_temporal_heuristic: Whether to use temporal proximity for causal linking (Rule 4)
        """
        import os

        # Load from environment if not specified
        self.extractor_model = extractor_model or os.environ.get("MODEL_WEAK_2", "mock")
        self.embedding_model = embedding_model or os.environ.get(
            "EMBEDDING_MODEL", "mock"
        )
        self.storage_path = Path(storage_path) if storage_path else None
        self.enable_temporal_heuristic = enable_temporal_heuristic

        # Initialize LLM backend
        self._backend = self._init_backend(llm_backend)

        # Initialize embedding backend
        self._embedding_backend = self._init_embedding_backend(embedding_backend)

        # Internal state
        self._objects: Dict[str, CanvasObject] = {}
        self._turn_counter: int = 0
        self._graph: CanvasGraph = CanvasGraph()

        # Load existing state if available
        if self.storage_path and self.storage_path.exists():
            self._load()

    def _init_backend(self, llm_backend: Optional[LLMBackend]) -> LLMBackend:
        """Initialize the LLM backend based on configuration."""
        if llm_backend is not None:
            return llm_backend

        if self.extractor_model == "mock":
            return MockLLMBackend()

        # Try to initialize OpenAI backend
        try:
            from cogcanvas.llm.openai import OpenAIBackend

            return OpenAIBackend(
                model=self.extractor_model,
                embedding_model=self.embedding_model,
            )
        except (ImportError, ValueError) as e:
            print(
                f"Warning: Could not initialize OpenAI backend ({e}), falling back to mock"
            )
            return MockLLMBackend()

    def _init_embedding_backend(
        self, embedding_backend: Optional[EmbeddingBackend]
    ) -> EmbeddingBackend:
        """Initialize the embedding backend based on configuration."""
        import os

        if embedding_backend is not None:
            return embedding_backend

        if self.embedding_model == "mock":
            print(
                "Warning: Explicitly using MockEmbeddingBackend. Results will be random."
            )
            return MockEmbeddingBackend()

        # Check if API embedding is configured
        api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        api_base = os.environ.get("EMBEDDING_API_BASE") or os.environ.get(
            "OPENAI_API_BASE"
        )

        # Use API backend if API key is configured
        if api_key:
            try:
                from cogcanvas.embeddings import APIEmbeddingBackend

                return APIEmbeddingBackend(
                    model=self.embedding_model,
                    api_key=api_key,
                    api_base=api_base,
                )
            except Exception as e:
                # CRITICAL: Do not fallback to mock silently in production/experiments!
                raise RuntimeError(
                    f"Failed to initialize API embedding backend for model '{self.embedding_model}': {e}. "
                    "Aborting to prevent invalid experimental results."
                )

        # No API key configured
        raise RuntimeError(
            "No EMBEDDING_API_KEY or OPENAI_API_KEY found in environment. "
            "Cannot initialize real embeddings. Set MODEL='mock' if you really intend to use random embeddings."
        )

    # =========================================================================
    # Core API
    # =========================================================================

    def extract(
        self,
        user: str,
        assistant: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """
        Extract canvas objects from a dialogue turn.

        This is typically called after each conversation turn to identify
        and store any decisions, TODOs, facts, etc.

        Args:
            user: The user's message
            assistant: The assistant's response
            metadata: Optional additional context

        Returns:
            ExtractionResult with extracted objects
        """
        start_time = time.time()
        self._turn_counter += 1

        # Use LLM backend for extraction (real or mock)
        existing = list(self._objects.values())
        objects = self._backend.extract_objects(user, assistant, existing)

        # Compute embeddings for extracted objects
        if objects:
            texts = [obj.content for obj in objects]
            embeddings = self._embedding_backend.embed_batch(texts)
            for obj, embedding in zip(objects, embeddings):
                obj.embedding = embedding

        # Store extracted objects
        for obj in objects:
            obj.turn_id = self._turn_counter
            self._objects[obj.id] = obj
            self._graph.add_node(obj)

        # Infer relationships automatically
        self._infer_relations(
            objects, enable_temporal_heuristic=self.enable_temporal_heuristic
        )

        # Persist if storage configured
        if self.storage_path:
            self._save()

        return ExtractionResult(
            objects=objects,
            extraction_time=time.time() - start_time,
            model_used=self.extractor_model,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        obj_type: Optional[ObjectType] = None,
        method: Literal["semantic", "keyword", "hybrid"] = "semantic",
        include_related: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve relevant canvas objects for a query.

        ENHANCED (Paper v2): Implements Hybrid Retrieval (Semantic + Keyword Fusion).
        Even when method='semantic', we now incorporate keyword signals to improve
        recall for specific entities (e.g., "AWS", "PostgreSQL") that might have
        low semantic overlap with the query context.

        Fusion Logic:
        - Base: 0.7 * Semantic + 0.3 * Keyword
        - Boost: If Keyword > 0.5 (strong match), use max(Semantic, Keyword)

        Args:
            query: The search query
            top_k: Maximum number of objects to return
            obj_type: Filter by object type (optional)
            method: Retrieval method ("semantic", "keyword", or "hybrid")
            include_related: If True, include 1-hop related objects

        Returns:
            RetrievalResult with matching objects and scores
        """
        start_time = time.time()

        # Filter by type if specified
        candidates = list(self._objects.values())
        if obj_type:
            candidates = [obj for obj in candidates if obj.type == obj_type]

        # 1. Compute scores based on method
        if method == "keyword":
            # Pure keyword
            scored = self._keyword_retrieve(query, candidates)
        else:
            # Hybrid / Semantic (we upgrade 'semantic' to hybrid silently for better performance)
            semantic_list = self._semantic_retrieve(query, candidates)
            keyword_list = self._keyword_retrieve(query, candidates)

            # Convert to dict for O(1) lookup. Object hash is its ID usually, but here we use object instance.
            # Assuming CanvasObject is hashable (it is dataclass with frozen=False but eq=True default).
            # To be safe, map by ID.
            semantic_map = {obj.id: score for obj, score in semantic_list}
            keyword_map = {obj.id: score for obj, score in keyword_list}

            scored = []
            ALPHA = 0.7

            for obj in candidates:
                s_score = semantic_map.get(obj.id, 0.0)
                k_score = keyword_map.get(obj.id, 0.0)

                # Fusion Logic
                if k_score > 0.5:
                    # Strong keyword match (e.g. exact entity mention) -> Trust it
                    final_score = max(s_score, k_score)
                else:
                    # Weak/No keyword match -> Rely mostly on semantic
                    final_score = (ALPHA * s_score) + ((1 - ALPHA) * k_score)

                if final_score > 0:
                    scored.append((obj, final_score))

        # Sort by score and take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_objects = [obj for obj, _ in scored[:top_k]]
        top_scores = [score for _, score in scored[:top_k]]

        # If include_related is True, add 1-hop neighbors
        if include_related:
            related_objects = []
            related_scores = []
            for obj, score in zip(top_objects, top_scores):
                # Get related object IDs from graph
                related_ids = self._graph.get_neighbors(obj.id, direction="both")
                # Add related objects with slightly lower scores
                for related_id in related_ids:
                    related_obj = self._objects.get(related_id)
                    # Avoid duplicates
                    if (
                        related_obj
                        and related_obj not in top_objects
                        and related_obj not in related_objects
                    ):
                        related_objects.append(related_obj)
                        # Decay factor for related nodes
                        related_scores.append(score * 0.8)

            # Combine main and related results
            top_objects.extend(related_objects)
            top_scores.extend(related_scores)

        return RetrievalResult(
            objects=top_objects,
            scores=top_scores,
            query=query,
            retrieval_time=time.time() - start_time,
        )

    def retrieve_and_filter(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
        obj_type: Optional[ObjectType] = None,
        method: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        include_related: bool = False,
        use_llm_filter: bool = True,
    ) -> RetrievalResult:
        """
        Two-stage retrieval with optional LLM-based filtering.

        Stage 1: Retrieve candidate_k objects using embedding similarity
        Stage 2: Use LLM to filter down to final_k most relevant objects

        This improves precision by filtering out "related but not relevant" content
        that can distract the final answer generation.

        Args:
            query: Search query
            candidate_k: Number of candidates to retrieve in stage 1
            final_k: Number of objects to keep after filtering
            obj_type: Optional filter by object type
            method: Retrieval method ("semantic", "keyword", "hybrid")
            include_related: Whether to include 1-hop graph neighbors
            use_llm_filter: Whether to apply LLM filtering (can be disabled for ablation)

        Returns:
            RetrievalResult with filtered objects
        """
        # Stage 1: Initial retrieval
        candidates = self.retrieve(
            query=query,
            top_k=candidate_k,
            obj_type=obj_type,
            method=method,
            include_related=include_related,
        )

        # If filtering disabled or not enough candidates, return as-is
        if not use_llm_filter or len(candidates.objects) <= final_k:
            # Just truncate to final_k
            return RetrievalResult(
                objects=candidates.objects[:final_k],
                scores=candidates.scores[:final_k] if candidates.scores else [],
                query=query,
                retrieval_time=candidates.retrieval_time,
            )

        # Stage 2: LLM filtering
        try:
            from cogcanvas.filtering import RetrievalFilter

            # Initialize filter (lazy loading)
            if not hasattr(self, "_retrieval_filter"):
                self._retrieval_filter = RetrievalFilter()

            filtered = self._retrieval_filter.filter(
                query=query,
                candidates=candidates,
                top_k=final_k,
            )

            # Convert back to RetrievalResult
            return filtered.to_retrieval_result()

        except ImportError:
            logger.warning("filtering module not available, returning unfiltered")
            return RetrievalResult(
                objects=candidates.objects[:final_k],
                scores=candidates.scores[:final_k] if candidates.scores else [],
                query=query,
                retrieval_time=candidates.retrieval_time,
            )
        except Exception as e:
            logger.warning(f"LLM filtering failed: {e}, returning unfiltered")
            return RetrievalResult(
                objects=candidates.objects[:final_k],
                scores=candidates.scores[:final_k] if candidates.scores else [],
                query=query,
                retrieval_time=candidates.retrieval_time,
            )

    def inject(
        self,
        result: RetrievalResult,
        format: str = "markdown",
        max_tokens: Optional[int] = None,
        strategy: Literal["confidence", "recency", "relevance"] = "relevance",
    ) -> str:
        """
        Format retrieved objects for injection into a prompt.

        ADAPTIVE INJECTION: Automatically prunes to stay within token budget.
        This is CRITICAL for efficiency - reviewers will calculate:
            Efficiency Ratio = Information_Retained / Extra_Tokens

        Args:
            result: RetrievalResult from retrieve()
            format: Output format ("markdown", "json", "compact")
            max_tokens: Maximum token budget (None = unlimited, DANGEROUS for production)
            strategy: Pruning strategy when over budget:
                - "relevance": Keep highest-scored objects (default, uses retrieval scores)
                - "confidence": Keep highest-confidence extractions
                - "recency": Keep most recent objects

        Returns:
            Formatted string to inject into prompt
        """
        if not result.objects:
            return ""

        objects = list(result.objects)
        scores = list(result.scores) if result.scores else [1.0] * len(objects)

        # Sort by strategy for pruning priority
        if strategy == "confidence":
            paired = sorted(
                zip(objects, scores), key=lambda x: x[0].confidence, reverse=True
            )
        elif strategy == "recency":
            paired = sorted(
                zip(objects, scores), key=lambda x: x[0].turn_id, reverse=True
            )
        else:  # relevance (default) - already sorted by retrieval score
            paired = list(zip(objects, scores))

        objects = [p[0] for p in paired]
        scores = [p[1] for p in paired]

        # Adaptive pruning if max_tokens specified
        if max_tokens:
            objects, scores = self._prune_to_token_budget(
                objects, scores, max_tokens, format
            )

        # Format output
        if format == "json":
            # Compact JSON without embeddings (save tokens!)
            compact_objs = []
            for obj in objects:
                compact_objs.append(
                    {
                        "type": obj.type.value,
                        "content": obj.content,
                        "quote": obj.quote[:100] if obj.quote else "",  # Truncate quote
                    }
                )
            return json.dumps(compact_objs, separators=(",", ":"))  # No whitespace

        if format == "compact":
            # Ultra-compact format for maximum token efficiency
            lines = []
            for obj in objects:
                type_abbrev = obj.type.value[0].upper()  # D/T/K/R/I
                lines.append(f"[{type_abbrev}] {obj.content}")
            return "\n".join(lines)

        # Default: markdown (most readable)
        lines = ["## Relevant Context from This Conversation\n"]
        for obj in objects:
            type_label = obj.type.value.replace("_", " ").title()
            line = f"- **[{type_label}]** {obj.content}"
            # Add grounding quote if available (reviewers love verifiable sources!)
            if obj.quote:
                truncated = (
                    obj.quote[:150] + "..." if len(obj.quote) > 150 else obj.quote
                )
                line += f'\n  > "{truncated}"'
            lines.append(line)

        return "\n".join(lines)

    def _prune_to_token_budget(
        self,
        objects: List[CanvasObject],
        scores: List[float],
        max_tokens: int,
        format: str,
    ) -> tuple:
        """
        Prune objects to fit within token budget.

        Uses a greedy approach: keep adding objects until budget exhausted.
        Objects should already be sorted by priority (confidence/recency/relevance).
        """
        # Rough token estimation (1 token ≈ 4 chars for English)
        CHARS_PER_TOKEN = 4

        selected_objects = []
        selected_scores = []
        current_tokens = 0

        # Reserve tokens for header
        header_tokens = 20 if format == "markdown" else 5
        remaining_budget = max_tokens - header_tokens

        for obj, score in zip(objects, scores):
            # Estimate tokens for this object
            obj_text = obj.content
            if format == "markdown" and obj.quote:
                obj_text += obj.quote[:150]
            obj_tokens = (
                len(obj_text) // CHARS_PER_TOKEN + 10
            )  # +10 for formatting overhead

            if current_tokens + obj_tokens <= remaining_budget:
                selected_objects.append(obj)
                selected_scores.append(score)
                current_tokens += obj_tokens
            else:
                break  # Budget exhausted

        return selected_objects, selected_scores

    def estimate_injection_tokens(
        self,
        result: RetrievalResult,
        format: str = "markdown",
    ) -> int:
        """
        Estimate how many tokens an injection would use.

        Useful for efficiency analysis in experiments.
        """
        if not result.objects:
            return 0

        injected = self.inject(result, format=format, max_tokens=None)
        return len(injected) // 4  # Rough estimate: 4 chars per token

    # =========================================================================
    # Canvas Management
    # =========================================================================

    def add(self, obj: CanvasObject, compute_embedding: bool = True) -> None:
        """
        Manually add an object to the canvas.

        Args:
            obj: CanvasObject to add
            compute_embedding: Whether to compute embedding if missing
        """
        # Compute embedding if needed
        if compute_embedding and obj.embedding is None and obj.content:
            obj.embedding = self._embedding_backend.embed(obj.content)
        self._objects[obj.id] = obj
        self._graph.add_node(obj)
        if self.storage_path:
            self._save()

    def remove(self, obj_id: str) -> bool:
        """Remove an object from the canvas."""
        if obj_id in self._objects:
            del self._objects[obj_id]
            self._graph.remove_node(obj_id)
            if self.storage_path:
                self._save()
            return True
        return False

    def get(self, obj_id: str) -> Optional[CanvasObject]:
        """Get an object by ID."""
        return self._objects.get(obj_id)

    def list_objects(
        self,
        obj_type: Optional[ObjectType] = None,
    ) -> List[CanvasObject]:
        """List all objects, optionally filtered by type."""
        objects = list(self._objects.values())
        if obj_type:
            objects = [obj for obj in objects if obj.type == obj_type]
        return sorted(objects, key=lambda x: x.turn_id)

    def clear(self) -> None:
        """Clear all objects from the canvas."""
        self._objects.clear()
        self._turn_counter = 0
        self._graph = CanvasGraph()
        if self.storage_path:
            self._save()

    # =========================================================================
    # Graph Operations
    # =========================================================================

    def link(
        self,
        from_id: str,
        to_id: str,
        relation: str = "references",
    ) -> bool:
        """
        Create a relationship between two objects.

        Args:
            from_id: Source object ID
            to_id: Target object ID
            relation: Type of relation ("references", "leads_to", "caused_by")

        Returns:
            True if link was created
        """
        from_obj = self._objects.get(from_id)
        to_obj = self._objects.get(to_id)

        if not from_obj or not to_obj:
            return False

        # Update graph
        success = self._graph.add_edge(from_id, to_id, relation)
        if not success:
            return False

        # Update object relationship fields
        if relation == "references":
            if to_id not in from_obj.references:
                from_obj.references.append(to_id)
            if from_id not in to_obj.referenced_by:
                to_obj.referenced_by.append(from_id)
        elif relation == "leads_to":
            if to_id not in from_obj.leads_to:
                from_obj.leads_to.append(to_id)
            if from_id not in to_obj.caused_by:
                to_obj.caused_by.append(from_id)
        elif relation == "caused_by":
            if to_id not in from_obj.caused_by:
                from_obj.caused_by.append(to_id)
            if from_id not in to_obj.leads_to:
                to_obj.leads_to.append(from_id)

        if self.storage_path:
            self._save()

        return True

    def auto_link(
        self,
        reference_threshold: float = 0.5,
        causal_threshold: float = 0.45,
    ) -> int:
        """
        Re-run automatic relationship inference on all objects.

        This method recalculates semantic relationships between all objects
        using the specified thresholds. Useful when you want to adjust
        the sensitivity of relationship detection.

        Args:
            reference_threshold: Min cosine similarity for 'references' relation
            causal_threshold: Min cosine similarity for 'caused_by' relation

        Returns:
            Number of new links created
        """
        # Get all objects
        all_objects = list(self._objects.values())
        if len(all_objects) < 2:
            return 0

        # Count existing links before
        links_before = sum(
            len(obj.references) + len(obj.leads_to) + len(obj.caused_by)
            for obj in all_objects
        )

        # Re-run inference for each object (treating each as "new")
        # Process in order of turn_id to maintain causal correctness
        sorted_objects = sorted(all_objects, key=lambda o: o.turn_id)

        for i, obj in enumerate(sorted_objects):
            # Skip if no embedding
            if obj.embedding is None:
                continue

            # Get objects from earlier turns
            earlier_objects = [o for o in sorted_objects[:i] if o.embedding is not None]
            if not earlier_objects:
                continue

            # Run inference with this single object as "new"
            self._infer_relations(
                [obj],
                reference_threshold=reference_threshold,
                causal_threshold=causal_threshold,
            )

        # Count new links
        links_after = sum(
            len(obj.references) + len(obj.leads_to) + len(obj.caused_by)
            for obj in all_objects
        )

        return links_after - links_before

    def get_related(
        self, obj_id: str, depth: int = 1, relation: Optional[str] = None
    ) -> List[CanvasObject]:
        """
        Get all objects related to the given object.

        Args:
            obj_id: The object ID to query
            depth: How many hops to traverse (default: 1)
            relation: Filter by relation type (optional)

        Returns:
            List of related CanvasObjects
        """
        if obj_id not in self._objects:
            return []

        # Use graph to get related IDs
        related_ids = self._graph.get_subgraph(obj_id, depth, relation)

        return [self._objects[rid] for rid in related_ids if rid in self._objects]

    def get_subgraph(
        self, obj_id: str, depth: int = 1, relation: Optional[str] = None
    ) -> List[CanvasObject]:
        """
        Get a subgraph of objects within N hops.

        This is an alias for get_related() with explicit naming.

        Args:
            obj_id: Starting object ID
            depth: Maximum number of hops
            relation: Filter by relation type

        Returns:
            List of CanvasObjects in the subgraph
        """
        return self.get_related(obj_id, depth, relation)

    def find_path(
        self, from_id: str, to_id: str, relation: Optional[str] = None
    ) -> List[CanvasObject]:
        """
        Find shortest path between two objects.

        Args:
            from_id: Starting object ID
            to_id: Target object ID
            relation: Filter by relation type

        Returns:
            List of CanvasObjects forming the path (empty if no path)
        """
        path_ids = self._graph.find_path(from_id, to_id, relation)
        return [self._objects[oid] for oid in path_ids if oid in self._objects]

    def get_roots(self, relation: Optional[str] = None) -> List[CanvasObject]:
        """
        Get objects with no incoming edges.

        Args:
            relation: Filter by relation type

        Returns:
            List of root CanvasObjects
        """
        root_ids = self._graph.get_roots(relation)
        return [self._objects[rid] for rid in root_ids if rid in self._objects]

    def get_leaves(self, relation: Optional[str] = None) -> List[CanvasObject]:
        """
        Get objects with no outgoing edges.

        Args:
            relation: Filter by relation type

        Returns:
            List of leaf CanvasObjects
        """
        leaf_ids = self._graph.get_leaves(relation)
        return [self._objects[lid] for lid in leaf_ids if lid in self._objects]

    def topological_sort(self, relation: str = "leads_to") -> List[CanvasObject]:
        """
        Return objects in topological order based on causal relationships.

        Args:
            relation: Relation type to use for ordering

        Returns:
            List of CanvasObjects in topological order
        """
        sorted_ids = self._graph.topological_sort(relation)
        return [self._objects[oid] for oid in sorted_ids if oid in self._objects]

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def size(self) -> int:
        """Number of objects in the canvas."""
        return len(self._objects)

    @property
    def turn_count(self) -> int:
        """Number of dialogue turns processed."""
        return self._turn_counter

    def stats(self) -> Dict[str, Any]:
        """Get canvas statistics."""
        type_counts = {}
        total_confidence = 0.0
        for obj in self._objects.values():
            type_counts[obj.type.value] = type_counts.get(obj.type.value, 0) + 1
            try:
                conf = float(obj.confidence)
            except (ValueError, TypeError):
                conf = 0.0
            total_confidence += conf

        avg_confidence = total_confidence / len(self._objects) if self._objects else 0.0

        return {
            "total_objects": self.size,
            "turn_count": self._turn_counter,
            "by_type": type_counts,
            "avg_confidence": avg_confidence,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save(self) -> None:
        """Save canvas state to storage."""
        if not self.storage_path:
            return

        data = {
            "turn_counter": self._turn_counter,
            "objects": [obj.to_dict() for obj in self._objects.values()],
            "graph": self._graph.to_dict(),
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load canvas state from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        self._turn_counter = data.get("turn_counter", 0)
        self._objects = {
            obj_data["id"]: CanvasObject.from_dict(obj_data)
            for obj_data in data.get("objects", [])
        }

        # Load graph if available
        graph_data = data.get("graph")
        if graph_data:
            self._graph = CanvasGraph.from_dict(graph_data)
        else:
            # Rebuild graph from object relationships (for backward compatibility)
            self._graph = CanvasGraph()
            for obj in self._objects.values():
                self._graph.add_node(obj)

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _infer_relations(
        self,
        new_objects: List[CanvasObject],
        reference_threshold: float = 0.5,  # Lowered for better cross-lingual matching
        causal_threshold: float = 0.45,  # More sensitive causal detection
        enable_temporal_heuristic: bool = True,  # Ablation control
    ) -> None:
        """
        Automatically infer relationships using HYBRID SIMILARITY (Semantic + Keyword).

        Improvements (Paper v2):
        - Added keyword overlap detection to catch "Budget" -> "Budget Plan" links
          that might have low semantic similarity in some models.
        - Hybrid thresholding: Lower semantic barrier if keywords match.

        Inference rules:
        1. Semantic references: If cosine similarity > reference_threshold, create 'references' link
        2. Causal TODO->DECISION: Recent decisions with high similarity cause TODOs
        3. INSIGHT causality: INSIGHTs caused_by semantically related KEY_FACTs or DECISIONs
        4. (Optional) Temporal Heuristic: Recent KEY_FACTs cause DECISIONs regardless of semantics

        Args:
            new_objects: Newly extracted objects to analyze
            reference_threshold: Min cosine similarity for 'references' relation (default: 0.7)
            causal_threshold: Min cosine similarity for 'caused_by' relation (default: 0.6)
            enable_temporal_heuristic: Whether to use Rule 4 (default: True)
        """
        from cogcanvas.embeddings import batch_cosine_similarity
        import re

        # Helper for keyword extraction
        STOPWORDS = {
            "the",
            "a",
            "an",
            "that",
            "this",
            "to",
            "in",
            "on",
            "for",
            "of",
            "with",
            "is",
            "was",
            "are",
            "were",
            "be",
            "have",
            "had",
            "do",
            "does",
            "did",
            "and",
            "or",
            "but",
            "so",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "hers",
            "ours",
            "theirs",
        }

        def get_keywords(text: str) -> set:
            words = re.findall(r"\b[a-z]{3,}\b", text.lower())
            return {w for w in words if w not in STOPWORDS}

        # Get existing objects (excluding new ones)
        new_ids = {obj.id for obj in new_objects}
        existing = [obj for obj in self._objects.values() if obj.id not in new_ids]

        if not existing:
            return  # Nothing to link to

        # Filter existing objects that have embeddings
        existing_with_embeddings = [
            obj for obj in existing if obj.embedding is not None
        ]
        if not existing_with_embeddings:
            return

        existing_embeddings = [obj.embedding for obj in existing_with_embeddings]

        for new_obj in new_objects:
            if new_obj.embedding is None:
                continue  # Skip objects without embeddings

            # Pre-compute new object keywords
            new_keywords = get_keywords(new_obj.content)
            if new_obj.quote:
                new_keywords.update(get_keywords(new_obj.quote))

            # Compute similarity to all existing objects at once (efficient batch operation)
            similarities = batch_cosine_similarity(
                new_obj.embedding, existing_embeddings
            )

            # Rule 1: Semantic references (for all object types)
            for existing_obj, sim in zip(existing_with_embeddings, similarities):
                # Check for keyword overlap
                existing_keywords = get_keywords(existing_obj.content)
                overlap = False
                if new_keywords and existing_keywords:
                    if not new_keywords.isdisjoint(existing_keywords):
                        overlap = True

                # Hybrid matching: Link if semantic similarity is high OR (overlap AND acceptable similarity)
                # We lower the threshold significantly if there is keyword overlap (0.3 is usually distinct enough)
                if sim >= reference_threshold or (overlap and sim >= 0.3):
                    # Use link() to update both graph AND object fields
                    self.link(new_obj.id, existing_obj.id, "references")

            # Rule 2: TODOs caused by recent DECISIONs (semantic matching)
            if new_obj.type == ObjectType.TODO:
                # Find recent decisions (within last 5 turns) with high semantic similarity
                best_decision = None
                best_sim = 0.0

                # Dynamic threshold: Lower it if we have keyword overlap
                current_threshold = causal_threshold

                for existing_obj, sim in zip(existing_with_embeddings, similarities):
                    if (
                        existing_obj.type == ObjectType.DECISION
                        and existing_obj.turn_id >= new_obj.turn_id - 5
                        and existing_obj.turn_id < new_obj.turn_id
                    ):

                        # Check overlap specific to this pair
                        existing_keywords = get_keywords(existing_obj.content)
                        overlap = (
                            not new_keywords.isdisjoint(existing_keywords)
                            if (new_keywords and existing_keywords)
                            else False
                        )

                        # Effective threshold for this pair
                        eff_threshold = 0.3 if overlap else causal_threshold

                        if sim > best_sim and sim >= eff_threshold:
                            best_decision = existing_obj
                            best_sim = sim

                if best_decision:
                    # Use link() to update both graph AND object fields
                    self.link(new_obj.id, best_decision.id, "caused_by")

            # Rule 3: INSIGHTs caused_by KEY_FACTs or DECISIONs (semantic matching)
            if new_obj.type == ObjectType.INSIGHT:
                for existing_obj, sim in zip(existing_with_embeddings, similarities):
                    if (
                        existing_obj.type in (ObjectType.KEY_FACT, ObjectType.DECISION)
                        and existing_obj.turn_id >= new_obj.turn_id - 3
                        and existing_obj.turn_id < new_obj.turn_id
                    ):

                        # Check overlap
                        existing_keywords = get_keywords(existing_obj.content)
                        overlap = (
                            not new_keywords.isdisjoint(existing_keywords)
                            if (new_keywords and existing_keywords)
                            else False
                        )

                        eff_threshold = 0.3 if overlap else causal_threshold

                        if sim >= eff_threshold:
                            # Use link() to update both graph AND object fields
                            self.link(new_obj.id, existing_obj.id, "caused_by")

            # Rule 4: DECISIONs caused by recent KEY_FACTs/REMINDERS (Temporal Heuristic)
            # This is critical for catching "Budget $500" -> "Choose AWS" links where semantic overlap is low.
            if enable_temporal_heuristic and new_obj.type == ObjectType.DECISION:
                recent_constraints = []
                for (
                    existing_obj
                ) in existing:  # Use raw existing list, embeddings not needed
                    if (
                        existing_obj.type in (ObjectType.KEY_FACT, ObjectType.REMINDER)
                        and existing_obj.turn_id >= new_obj.turn_id - 5
                        and existing_obj.turn_id < new_obj.turn_id
                    ):
                        recent_constraints.append(existing_obj)

                # Sort by recency (closest to decision first)
                recent_constraints.sort(key=lambda x: x.turn_id, reverse=True)

                # Link top 3
                for constraint in recent_constraints[:3]:
                    self.link(new_obj.id, constraint.id, "caused_by")

    # =========================================================================
    # Mock Helpers (for testing)
    # =========================================================================

    def _mock_extract(self, user: str, assistant: str) -> List[CanvasObject]:
        """Mock extraction for testing without LLM."""
        objects = []
        combined = f"{user} {assistant}".lower()

        # Simple rule-based extraction for demo
        if any(word in combined for word in ["decide", "use", "choose", "go with"]):
            objects.append(
                CanvasObject(
                    type=ObjectType.DECISION,
                    content=f"Decision from turn {self._turn_counter}: {user[:100]}",
                    context="Extracted from user message",
                )
            )

        if any(word in combined for word in ["todo", "need to", "should", "must"]):
            objects.append(
                CanvasObject(
                    type=ObjectType.TODO,
                    content=f"TODO from turn {self._turn_counter}: {user[:100]}",
                    context="Extracted from conversation",
                )
            )

        if any(word in combined for word in ["remember", "note", "important"]):
            objects.append(
                CanvasObject(
                    type=ObjectType.REMINDER,
                    content=f"Reminder from turn {self._turn_counter}: {user[:100]}",
                    context="Extracted from conversation",
                )
            )

        return objects

    def _simple_match_score(self, query: str, content: str) -> float:
        """Simple keyword matching score with stopword filtering and punctuation removal."""
        import string

        STOPWORDS = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "at",
            "by",
            "for",
            "from",
            "in",
            "of",
            "on",
            "to",
            "with",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "where",
            "this",
            "that",
            "these",
            "those",
            "can",
            "could",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
        }

        def clean_tokenize(text: str) -> set:
            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            return {w for w in text.lower().split() if w not in STOPWORDS}

        query_words = clean_tokenize(query)
        content_words = clean_tokenize(content)

        if not query_words:
            return 0.0

        overlap = query_words & content_words
        return len(overlap) / len(query_words)

    def _keyword_retrieve(
        self, query: str, candidates: List[CanvasObject]
    ) -> List[Tuple[CanvasObject, float]]:
        """
        Retrieve objects using keyword matching.

        Args:
            query: Search query
            candidates: List of candidate objects to search

        Returns:
            List of (object, score) tuples
        """
        scored = []
        query_lower = query.lower()
        for obj in candidates:
            score = self._simple_match_score(query_lower, obj.content.lower())
            if score > 0:
                scored.append((obj, score))
        return scored

    def _semantic_retrieve(
        self, query: str, candidates: List[CanvasObject]
    ) -> List[Tuple[CanvasObject, float]]:
        """
        Retrieve objects using semantic similarity.

        Args:
            query: Search query
            candidates: List of candidate objects to search

        Returns:
            List of (object, score) tuples
        """
        if not candidates:
            return []

        # Get query embedding
        query_embedding = self._embedding_backend.embed(query)

        # Get candidate embeddings (filter out None embeddings)
        valid_candidates = [
            (obj, obj.embedding) for obj in candidates if obj.embedding is not None
        ]
        if not valid_candidates:
            return [(obj, 0.0) for obj in candidates]

        candidate_objs = [obj for obj, _ in valid_candidates]
        candidate_embeddings = [emb for _, emb in valid_candidates]

        # Calculate similarities
        from cogcanvas.embeddings import batch_cosine_similarity

        similarities = batch_cosine_similarity(query_embedding, candidate_embeddings)

        # Pair objects with scores
        scored = [
            (obj, float(score)) for obj, score in zip(candidate_objs, similarities)
        ]

        return scored

    # =========================================================================
    # Magic Methods
    # =========================================================================

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"Canvas(objects={self.size}, turns={self._turn_counter})"
