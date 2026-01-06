"""
CogCanvas Agent for Evaluation.

Wraps the Canvas class into the Agent interface for the experiment runner.

Key behavior:
- process_turn(): Extracts canvas objects from each turn
- on_compression(): History is truncated, but canvas objects SURVIVE
- answer_question(): Retrieves relevant objects and generates answer
"""

from typing import List, Optional
import time
import re

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from experiments.llm_utils import call_llm_with_retry

from cogcanvas import Canvas
from cogcanvas.models import ObjectType
from cogcanvas.reranker import Reranker


class CogCanvasAgent(Agent):
    """
    CogCanvas agent that uses the Canvas for compression-resistant memory.

    On compression:
    - Conversation history is truncated (like baseline)
    - Canvas objects are PRESERVED (the key differentiator!)

    On answer:
    - Retrieves relevant canvas objects
    - Combines with retained history to generate answer
    """

    def __init__(
        self,
        extractor_model: str = None,  # None = load from env (EXTRACTOR_MODEL)
        answer_model: str = None,  # None = load from env (ANSWER_MODEL)
        embedding_model: str = None,  # None = load from env
        retrieval_top_k: int = 15,  # Increased from 10 to 15 for better recall
        enable_graph_expansion: bool = True,  # New flag for ablation
        graph_hops: int = 1,  # Number of hops for graph expansion (default 1 for backward compat)
        use_reranker: bool = False,  # Disable reranking to test baseline speed
        reranker_type: str = "api",  # Use BGE reranker API (MUCH faster than LLM)
        reranker_candidate_k: int = 30,  # Increased from 20 to 30 for better recall
        use_real_llm_for_answer: bool = True,  # Default to True for fair comparison
        # Ablation Parameters
        enable_temporal_heuristic: bool = True,
        retrieval_method: str = "hybrid",  # "semantic", "keyword", "hybrid"
        # We all are cot now
        prompt_style: str = "cot",  # "direct", "cot", "cot_temporal"
        # LLM Filtering Parameters
        use_llm_filter: bool = False,  # Enable LLM-based filtering (new feature)
        filter_candidate_k: int = 20,  # Number of candidates before filtering
        # Graph Construction Parameters
        reference_threshold: float = 0.5,  # Threshold for reference links
        causal_threshold: float = 0.45,  # Threshold for causal links
        # Gleaning Parameter (for ablation)
        enable_gleaning: bool = False,  # Disabled by default (minimal benefit on LoCoMo)
        # VAGE Parameters
        enable_vage: bool = False,  # Enable Vulnerability-Aware Greedy Extraction
        use_learned_vage: bool = False,  # Use learned vulnerability model
        vage_budget_k: int = 10,  # Max objects to keep per turn
        vage_mode: str = "off",  # "off" | "standard" | "chain" (chain uses graph-aware selection)
        vage_verbose: bool = False,  # Print detailed VAGE logs
        # Query Expansion Parameters
        use_query_expansion: bool = False,  # Enable multi-query retrieval (Perplexity-style)
        query_expansion_n: int = 3,  # Number of sub-queries to generate
        # Multi-Round Retrieval Parameters
        use_multi_round: bool = False,  # Enable multi-round iterative retrieval
        max_retrieval_rounds: int = 3,  # Maximum number of retrieval rounds
        confidence_threshold: float = 0.7,  # Confidence threshold to stop iteration
        # Query Routing Parameters
        use_query_routing: bool = False,  # Enable query complexity routing (simple->single, complex->multi)
        # Smart Routing Parameters (retrieval-result-based routing)
        use_smart_routing: bool = False,  # Enable smart routing based on retrieval quality
        smart_routing_high_score: float = 0.85,  # Threshold for "high confidence" direct answer
        smart_routing_spread_threshold: float = 0.15,  # Score spread threshold for multi-entity detection
        smart_routing_low_score: float = 0.5,  # Threshold for "low confidence" query expansion
        # Cache Parameters
        storage_path: str = None,  # Path to cache Canvas state
    ):
        """
        Initialize CogCanvas agent.

        Args:
            extractor_model: Model for Canvas extraction (default: EXTRACTOR_MODEL)
            answer_model: Model for answering questions (default: ANSWER_MODEL)
            embedding_model: Model for embeddings
            retrieval_top_k: Number of objects to retrieve (default: 15)
            enable_graph_expansion: Whether to use N-hop graph expansion (Ablation)
            graph_hops: Number of hops for graph expansion (default: 1)
            use_reranker: Enable reranking after retrieval
            use_real_llm_for_answer: If True, use LLM for answer generation
            enable_temporal_heuristic: Enable temporal causality rule in graph construction
            retrieval_method: Retrieval strategy
            prompt_style: Prompting strategy
            use_llm_filter: Enable LLM-based filtering to improve precision
            filter_candidate_k: Number of candidates to retrieve before filtering
            reference_threshold: Min cosine similarity for 'references' relation
            causal_threshold: Min cosine similarity for 'caused_by' relation
            enable_gleaning: Enable second-pass extraction (LightRAG-inspired, +2x LLM calls)
            enable_vage: Enable VAGE for optimal artifact selection
            use_learned_vage: Use learned vulnerability model instead of heuristics
            vage_budget_k: Max objects to keep per turn when VAGE is enabled
            vage_mode: VAGE mode - "off", "standard" (original), or "chain" (graph-aware)
            vage_verbose: Print detailed VAGE progress logs
            use_query_expansion: Enable Perplexity-style multi-query retrieval
            query_expansion_n: Number of sub-queries to generate (default: 3)
            use_multi_round: Enable multi-round iterative retrieval
            max_retrieval_rounds: Maximum number of retrieval rounds (default: 3)
            confidence_threshold: Confidence threshold to stop iteration (default: 0.7)
            use_query_routing: Enable query complexity routing (simple->single-round, complex->multi-round)
            use_smart_routing: Enable smart routing based on retrieval result quality
            smart_routing_high_score: Score threshold for direct answer (default: 0.85)
            smart_routing_spread_threshold: Score spread threshold for multi-entity detection (default: 0.15)
            smart_routing_low_score: Score threshold for query expansion (default: 0.5)
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()

        # Load models from .env if not specified
        # Extractor: EXTRACTOR_MODEL > gpt-4o-mini
        self.extractor_model = extractor_model or os.getenv(
            "EXTRACTOR_MODEL", "gpt-4o-mini"
        )
        # Answer: ANSWER_MODEL > glm-4-flash
        self.answer_model = answer_model or os.getenv(
            "ANSWER_MODEL", "glm-4-flash"
        )
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "bge-large-zh-v1.5"
        )

        self.retrieval_top_k = retrieval_top_k
        self.enable_graph_expansion = enable_graph_expansion
        self.graph_hops = graph_hops
        self.use_reranker = use_reranker
        self.reranker_type = reranker_type
        self.reranker_candidate_k = reranker_candidate_k
        self.use_real_llm_for_answer = use_real_llm_for_answer

        # Ablation config
        self.enable_temporal_heuristic = enable_temporal_heuristic
        self.retrieval_method = retrieval_method
        self.prompt_style = prompt_style

        # LLM Filtering config
        self.use_llm_filter = use_llm_filter
        self.filter_candidate_k = filter_candidate_k

        # Graph construction thresholds
        self.reference_threshold = reference_threshold
        self.causal_threshold = causal_threshold

        # Gleaning config
        self.enable_gleaning = enable_gleaning

        # VAGE config
        self.enable_vage = enable_vage
        self.use_learned_vage = use_learned_vage
        self.vage_budget_k = vage_budget_k
        self.vage_mode = vage_mode
        self.vage_verbose = vage_verbose

        # Query Expansion config
        self.use_query_expansion = use_query_expansion
        self.query_expansion_n = query_expansion_n

        # Multi-Round Retrieval config
        self.use_multi_round = use_multi_round
        self.max_retrieval_rounds = max_retrieval_rounds
        self.confidence_threshold = confidence_threshold

        # Query Routing config
        self.use_query_routing = use_query_routing

        # Smart Routing config (retrieval-result-based routing)
        self.use_smart_routing = use_smart_routing
        self.smart_routing_high_score = smart_routing_high_score
        self.smart_routing_spread_threshold = smart_routing_spread_threshold
        self.smart_routing_low_score = smart_routing_low_score

        # Cache config
        self.storage_path = storage_path

        # Initialize LLM client for answering (uses ANSWER_API_* if available)
        self._client = None
        self._answer_client = None  # Separate client for answer model
        if self.use_real_llm_for_answer:
            self._init_client()
            self._init_answer_client()

        # Initialize reranker if enabled
        self._reranker = None
        if self.use_reranker:
            self._init_reranker()

        # Will be initialized in reset()
        self._canvas: Optional[Canvas] = None
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []

        # Initialize
        self.reset()

    def _init_client(self):
        """Initialize LLM client for extraction (uses EXTRACTOR_API_*)."""
        try:
            from openai import OpenAI
            import os

            api_key = os.getenv("EXTRACTOR_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("EXTRACTOR_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if api_key:
                self._client = OpenAI(api_key=api_key, base_url=api_base)
        except ImportError:
            pass

    def _init_answer_client(self):
        """Initialize separate LLM client for answering (supports different API)."""
        try:
            from openai import OpenAI
            import os

            # Use ANSWER_API_* if available, otherwise fall back to default API_*
            answer_api_key = os.getenv("ANSWER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            answer_api_base = os.getenv("ANSWER_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if answer_api_key:
                self._answer_client = OpenAI(api_key=answer_api_key, base_url=answer_api_base)
        except ImportError:
            pass

    def _init_reranker(self):
        """Initialize reranker based on reranker_type."""
        try:
            import os
            from cogcanvas.reranker import LLMRerankerBackend

            if self.reranker_type == "llm":
                # LLM-based reranking (uses GPT to score relevance)
                if not self._client:
                    self._init_client()  # Ensure LLM client is initialized
                self._reranker = Reranker(
                    backend=LLMRerankerBackend(
                        llm_client=self._client,
                        model=self.extractor_model
                    )
                )
                print(f"Initialized LLM reranker with model: {self.extractor_model}")
            elif self.reranker_type == "api":
                # API-based reranking (uses BGE reranker)
                api_key = os.getenv("EMBEDDING_API_KEY")
                if api_key:
                    self._reranker = Reranker(use_mock=False)
                    print("Initialized API reranker")
                else:
                    print("No API key found, falling back to mock reranker")
                    self._reranker = Reranker(use_mock=True)
            else:  # "mock"
                self._reranker = Reranker(use_mock=True)
                print("Initialized mock reranker")
        except Exception as e:
            print(f"Failed to initialize reranker: {e}, using mock reranker")
            self._reranker = Reranker(use_mock=True)

    @property
    def name(self) -> str:
        parts = []
        if self.enable_graph_expansion:
            if self.graph_hops > 1:
                parts.append(f"Graph{self.graph_hops}hop")
            else:
                parts.append("Graph")
        if self.enable_temporal_heuristic:
            parts.append("Time")
        if self.retrieval_method == "hybrid":
            parts.append("Hybrid")
        if self.prompt_style == "cot":
            parts.append("CoT")
        if self.use_llm_filter:
            parts.append("Filter")
        if self.use_reranker:
            parts.append("Rerank")
        if self.use_multi_round:
            parts.append(f"MultiRound{self.max_retrieval_rounds}")
        if self.use_query_expansion:
            parts.append("QueryExp")
        if self.use_query_routing:
            parts.append("Routed")
        if self.use_smart_routing:
            parts.append("SmartRoute")

        config_str = "+".join(parts) if parts else "Baseline"
        return f"CogCanvas({config_str})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self._canvas = Canvas(
            extractor_model=self.extractor_model,
            embedding_model=self.embedding_model,
            enable_temporal_heuristic=self.enable_temporal_heuristic,
            reference_threshold=self.reference_threshold,
            causal_threshold=self.causal_threshold,
            enable_gleaning=self.enable_gleaning,
            enable_vage=self.enable_vage,
            use_learned_vage=self.use_learned_vage,
            vage_budget_k=self.vage_budget_k,
            vage_mode=self.vage_mode,
            vage_verbose=self.vage_verbose,
        )
        self._history = []
        self._retained_history = []

    def process_turn(self, turn: ConversationTurn, verbose: int = 0) -> None:
        """
        Process a conversation turn.

        Extracts canvas objects and stores the turn in history.
        """
        start = time.time()
        # Store in history
        self._history.append(turn)

        # Extract canvas objects from this turn
        metadata = {"turn_id": turn.turn_id}
        # Get session_datetime if available
        session_datetime = getattr(turn, "session_datetime", None)

        self._canvas.extract(
            user=turn.user,
            assistant=turn.assistant,
            metadata=metadata,
            session_datetime=session_datetime,  # Pass as separate parameter for temporal resolution
        )
        elapsed = (time.time() - start) * 1000
        if verbose >= 3:
            print(f"      [Turn {turn.turn_id}] Extracted in {elapsed:.0f}ms")

    def batch_extract(self, turns: List[ConversationTurn], verbose: int = 0) -> None:
        """
        Batch extract canvas objects from multiple turns at once.

        This is the recommended extraction method:
        - Combines all turns into a single LLM call
        - Better context for entity resolution and relation extraction
        - Much faster than per-turn extraction (1 call vs N calls)

        Args:
            turns: List of conversation turns to extract from
            verbose: Verbosity level for logging
        """
        if not turns:
            return

        start = time.time()

        # Combine all dialogue into a single conversation block
        # CRITICAL FIX: Include session_datetime in the dialogue text
        # Each turn may come from a different session with different datetime
        # This allows temporal resolution to use the correct base date for each turn
        combined_user_parts = []
        combined_assistant_parts = []

        for turn in turns:
            session_dt = getattr(turn, "session_datetime", None)
            dt_marker = f" (Session: {session_dt})" if session_dt else ""
            combined_user_parts.append(f"[Turn {turn.turn_id}{dt_marker}] {turn.user}")
            combined_assistant_parts.append(f"[Turn {turn.turn_id}] {turn.assistant}")

        combined_user = "\n".join(combined_user_parts)
        combined_assistant = "\n".join(combined_assistant_parts)

        # Get session_datetime from the last turn for fallback temporal resolution
        # Note: Per-turn session_datetime is now embedded in the dialogue text
        session_datetime = None
        for turn in reversed(turns):
            session_datetime = getattr(turn, "session_datetime", None)
            if session_datetime:
                break

        # Metadata for batch extraction
        metadata = {
            "turn_ids": [t.turn_id for t in turns],
            "batch_size": len(turns),
            "first_turn": turns[0].turn_id,
            "last_turn": turns[-1].turn_id,
            # Store all session datetimes for post-processing if needed
            "session_datetimes": {
                t.turn_id: getattr(t, "session_datetime", None)
                for t in turns
            },
        }

        # Single LLM call for all turns
        self._canvas.extract(
            user=combined_user,
            assistant=combined_assistant,
            metadata=metadata,
            session_datetime=session_datetime,
        )

        # Update history
        self._history.extend(turns)

        elapsed = (time.time() - start) * 1000
        if verbose >= 1:
            num_objects = len(self._canvas._objects)
            print(f"      [Batch Extract] {len(turns)} turns in {elapsed:.0f}ms, Canvas: {num_objects} objects")

    def store_turns_only(self, turns: List[ConversationTurn]) -> None:
        """
        Store turns in history without extraction.
        Used with batch_extract() to accumulate turns before batch processing.
        """
        self._history.extend(turns)

    def should_compress(
        self,
        current_turn_index: int,
        total_turns: int,
        verbose: int = 0
    ) -> tuple[bool, str]:
        """
        Dynamic compression trigger (Letta-inspired).

        Returns (should_compress, reason) tuple.

        Triggers based on:
        1. Topic shift (cosine similarity between recent turns)
        2. Object density (if extraction rate drops, topic is stale)
        3. Turn limit (safety net: compress every 60 turns max)

        Args:
            current_turn_index: Current turn index (0-based)
            total_turns: Total number of turns in conversation
            verbose: Verbosity level for logging

        Returns:
            (should_compress: bool, reason: str)
        """
        import numpy as np

        # Safety net: never exceed max turns without compression
        MAX_TURNS_WITHOUT_COMPRESSION = 60
        MIN_TURNS_BEFORE_COMPRESSION = 20

        if current_turn_index < MIN_TURNS_BEFORE_COMPRESSION:
            return False, "too_soon"

        # Condition 1: Turn limit (safety net)
        if current_turn_index >= MAX_TURNS_WITHOUT_COMPRESSION:
            if verbose >= 2:
                print(f"      [Compress Trigger] Turn limit reached ({current_turn_index} >= {MAX_TURNS_WITHOUT_COMPRESSION})")
            return True, "turn_limit"

        # Condition 2: Topic shift detection
        # Compare recent 5 turns vs previous 5 turns using embeddings
        if len(self._history) >= 15:
            recent_turns = self._history[-5:]
            prev_turns = self._history[-10:-5]

            # Get embeddings for recent and previous turns
            recent_texts = [f"{t.user} {t.assistant}" for t in recent_turns]
            prev_texts = [f"{t.user} {t.assistant}" for t in prev_turns]

            try:
                recent_emb = self._canvas._embedding_backend.embed_batch(recent_texts)
                prev_emb = self._canvas._embedding_backend.embed_batch(prev_texts)

                # Average embeddings
                recent_avg = np.mean(recent_emb, axis=0)
                prev_avg = np.mean(prev_emb, axis=0)

                # Cosine similarity
                from cogcanvas.embeddings import batch_cosine_similarity
                similarity = batch_cosine_similarity([recent_avg], [prev_avg])[0]

                TOPIC_SHIFT_THRESHOLD = 0.55  # Below this = topic shifted
                if similarity < TOPIC_SHIFT_THRESHOLD:
                    if verbose >= 2:
                        print(f"      [Compress Trigger] Topic shift detected (similarity={similarity:.2f} < {TOPIC_SHIFT_THRESHOLD})")
                    return True, f"topic_shift(sim={similarity:.2f})"
            except Exception as e:
                if verbose >= 3:
                    print(f"      [Compress] Topic detection failed: {e}")

        # Condition 3: Object density check
        # If recent turns have low object extraction rate, topic might be stale
        if len(self._history) >= 20:
            recent_10_turns = list(range(max(0, current_turn_index - 10), current_turn_index + 1))

            # Count objects created in recent turns
            recent_object_count = 0
            for obj in self._canvas._objects.values():
                if hasattr(obj, 'turn_id') and obj.turn_id in recent_10_turns:
                    recent_object_count += 1

            # Objects per turn in recent window
            objects_per_turn = recent_object_count / 10
            LOW_DENSITY_THRESHOLD = 0.5  # Below 0.5 objects/turn = low density

            if objects_per_turn < LOW_DENSITY_THRESHOLD:
                if verbose >= 2:
                    print(f"      [Compress Trigger] Low object density ({objects_per_turn:.1f}/turn < {LOW_DENSITY_THRESHOLD})")
                return True, f"low_density({objects_per_turn:.1f}/turn)"

        return False, "none"

    def on_compression(self, retained_turns: List[ConversationTurn], verbose: int = 0, reason: str = "fixed") -> None:
        """
        Handle compression event.

        History is truncated to retained_turns, but CANVAS OBJECTS SURVIVE.
        This is the key advantage of CogCanvas!

        Args:
            retained_turns: Turns to retain after compression
            verbose: Verbosity level for logging
            reason: Reason for compression trigger (for logging)
        """
        # Get stats BEFORE compression for logging
        total_history_turns = len(self._history)
        total_objects = len(self._canvas._objects)
        # Count edges across all relation types
        total_edges = 0
        if hasattr(self._canvas, '_graph') and hasattr(self._canvas._graph, '_edges'):
            for rel_edges in self._canvas._graph._edges.values():
                for targets in rel_edges.values():
                    total_edges += len(targets)

        # Truncate history (simulating context compression)
        self._retained_history = retained_turns

        # Log compression event with details
        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"[COMPRESSION TRIGGERED] reason={reason}")
            print(f"{'='*60}")
            print(f"  History: {total_history_turns} turns → {len(retained_turns)} retained")
            print(f"  Canvas Objects: {total_objects} (PRESERVED)")
            print(f"  Graph Edges: {total_edges} (PRESERVED)")

            # Show retained turn IDs
            if verbose >= 2 and retained_turns:
                turn_ids = [t.turn_id for t in retained_turns]
                print(f"  Retained Turn IDs: {turn_ids}")

            # Show object type distribution
            if verbose >= 2 and total_objects > 0:
                type_counts = {}
                for obj in self._canvas._objects.values():
                    obj_type = getattr(obj, 'type', 'unknown')
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                print(f"  Object Types: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

            print(f"{'='*60}\n")

        # NOTE: Canvas objects are NOT cleared!
        # This is the whole point - they survive compression

    # =========================================================================
    # Cache Methods
    # =========================================================================

    def save_canvas_state(self, path: str) -> None:
        """
        Save the current Canvas state to a file.

        Args:
            path: Path to save the Canvas state
        """
        if self._canvas is not None:
            self._canvas.save(path)

    def load_canvas_state(self, path: str) -> bool:
        """
        Load Canvas state from a file.

        Args:
            path: Path to load the Canvas state from

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._canvas is not None:
            return self._canvas.load(path)
        return False

    def has_cached_state(self, path: str) -> bool:
        """Check if cached Canvas state exists."""
        from pathlib import Path
        return Path(path).exists()

    def _adaptive_top_k(self, question: str, base_k: int = 10) -> int:
        """
        Dynamically adjust retrieval top-k based on question type.

        Multi-hop questions need more context, simple facts need less.

        Args:
            question: The question to analyze
            base_k: Base top-k value (default from config)

        Returns:
            Adjusted top-k value
        """
        q_lower = question.lower()

        # Multi-hop indicators: need more context (15)
        multihop_keywords = [
            "after", "before", "because", "caused", "led to", "result",
            "why did", "how did", "what happened", "consequence",
            "influence", "affect", "impact", "relationship between"
        ]
        if any(kw in q_lower for kw in multihop_keywords):
            return max(base_k, 15)

        # Temporal indicators: medium context (12)
        temporal_keywords = [
            "when", "what time", "what date", "how long", "since",
            "during", "while", "until", "ago", "recently"
        ]
        if any(kw in q_lower for kw in temporal_keywords):
            return max(base_k, 12)

        # Simple fact questions: base context
        return base_k

    def _expand_query(self, question: str, n: int = 3, verbose: int = 0) -> List[str]:
        """
        Expand a complex question into multiple sub-queries (Perplexity-style).

        Generates 3 different types of queries:
        1. Direct factual lookup - Focuses on extracting key entities and facts
        2. Temporal/contextual query - Focuses on time, context, and circumstances
        3. Related entities query - Focuses on related people, places, and background

        This helps with:
        - Complex reasoning questions that need multiple perspectives
        - Questions requiring both factual recall and inference
        - Commonsense questions that need related context

        Args:
            question: Original question
            n: Number of sub-queries to generate (default: 3)
            verbose: Verbosity level

        Returns:
            List of queries (original + expanded). Returns [question] if expansion fails.
        """
        import json

        # Use answer client if available, otherwise extractor client
        client = self._answer_client or self._client
        if not client:
            if verbose >= 2:
                print("        [Query Expansion] No LLM client available, using original query only")
            return [question]

        prompt = f"""You are a search query expansion expert. Your task is to generate diverse search queries that will help retrieve relevant information to answer the original question.

Original question: {question}

Generate exactly {n} different search queries that would help answer this question. Focus on different aspects:

1. **Direct factual lookup**: Rephrase the question to focus on extracting specific facts, entities, or attributes mentioned directly in the conversation.

2. **Temporal/contextual query**: Focus on time-related information, when things happened, the sequence of events, or the context/circumstances around the topic.

3. **Related entities query**: Focus on finding related people, places, relationships, or background information that could help answer the question.

Return your response as a JSON array with exactly {n} queries. Example format:
["query about direct facts", "query about timing or context", "query about related entities"]

IMPORTANT:
- Each query should be concise (under 50 words)
- Each query should approach the question from a different angle
- Only return the JSON array, nothing else
"""

        try:
            response = call_llm_with_retry(
                client=client,
                model=self.answer_model,  # Use answer model for faster response
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,  # Slight variation for diversity
            )

            # Try to parse JSON response
            response_text = response.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                # Extract content between code blocks
                lines = response_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or (not line.startswith("```") and "[" in response_text):
                        json_lines.append(line)
                response_text = "\n".join(json_lines).strip()

            # Find JSON array in the response
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                sub_queries = json.loads(json_str)

                # Validate that we got a list of strings
                if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                    # Always include original query first, then add expanded queries
                    all_queries = [question] + [q.strip() for q in sub_queries if q.strip()]

                    if verbose >= 2:
                        print(f"        [Query Expansion] Generated {len(all_queries)} queries:")
                        for i, q in enumerate(all_queries):
                            print(f"          {i+1}. {q[:80]}{'...' if len(q) > 80 else ''}")

                    return all_queries

            # If JSON parsing failed, try line-by-line parsing as fallback
            if verbose >= 2:
                print("        [Query Expansion] JSON parsing failed, trying line parsing")

            sub_queries = []
            for line in response_text.split('\n'):
                line = line.strip()
                # Remove numbering, bullets, quotes
                line = line.lstrip('0123456789.-) "\'')
                line = line.rstrip('"\'')
                if line and len(line) > 10:  # Minimum query length
                    sub_queries.append(line)

            if sub_queries:
                all_queries = [question] + sub_queries[:n]
                if verbose >= 2:
                    print(f"        [Query Expansion] Parsed {len(all_queries)} queries from text")
                return all_queries

            # Fallback to original query
            if verbose >= 1:
                print("        [Query Expansion] Could not parse response, using original query")
            return [question]

        except json.JSONDecodeError as e:
            if verbose >= 1:
                print(f"        [Query Expansion] JSON decode error: {e}, using original query")
            return [question]
        except Exception as e:
            if verbose >= 1:
                print(f"        [Query Expansion] Failed: {e}, using original query")
            return [question]

    def _check_answer_confidence(self, question: str, context: str, answer: str, verbose: int = 0) -> float:
        """
        Let LLM judge if the current context is sufficient to answer the question.

        Returns a confidence score between 0 and 1.
        - High score (>=0.7): Context has enough information
        - Low score (<0.7): Missing key information, need more retrieval

        Args:
            question: The original question
            context: Current retrieved context
            answer: Draft answer generated from context
            verbose: Verbosity level

        Returns:
            Confidence score (0-1)
        """
        answer_client = self._answer_client or self._client
        if not answer_client:
            return 1.0  # No LLM, assume confident

        prompt = f"""You are a STRICT evaluator checking if retrieved context can DEFINITIVELY answer a question.

## Question
{question}

## Retrieved Context
{context}

## Draft Answer
{answer}

## Scoring Guide (BE STRICT)
- 0.0-0.2: Context is completely irrelevant or missing
- 0.2-0.4: Context is partially relevant but missing KEY details (dates, names, specifics)
- 0.4-0.6: Context has some relevant info but answer requires GUESSING or INFERENCE
- 0.6-0.8: Context supports the answer but minor details might be uncertain
- 0.8-1.0: Context DIRECTLY and EXPLICITLY contains the answer with clear evidence

## IMPORTANT
- If the answer contains "I think", "probably", "might", "could be" → score < 0.5
- If specific dates/numbers/names in the answer are NOT in the context → score < 0.4
- If the answer is a guess based on general knowledge → score < 0.3
- ONLY give score >= 0.7 if the context EXPLICITLY contains the exact answer

Return ONLY a number (e.g., 0.35), nothing else."""

        try:
            response = call_llm_with_retry(
                client=answer_client,
                model=self.answer_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )

            # Parse the confidence score
            score_str = response.strip()
            # Handle cases like "0.8" or "0.85" or just "0"
            confidence = float(score_str)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            if verbose >= 2:
                print(f"        [Confidence Check] Score: {confidence:.2f}")

            return confidence

        except (ValueError, Exception) as e:
            if verbose >= 1:
                print(f"        [Confidence Check] Failed to parse: {e}, defaulting to 0.5")
            return 0.5  # Default to uncertain if parsing fails

    def _generate_followup_query(self, question: str, context: str, answer: str, verbose: int = 0) -> str:
        """
        Generate a follow-up query to retrieve missing information.

        Based on the current context and draft answer, identify what information
        is still needed and generate a targeted search query.

        Args:
            question: The original question
            context: Current retrieved context
            answer: Draft answer (may be incomplete)
            verbose: Verbosity level

        Returns:
            A follow-up search query string
        """
        answer_client = self._answer_client or self._client
        if not answer_client:
            return question  # Fallback to original question

        prompt = f"""You are helping to find missing information to answer a question.

## Original Question
{question}

## Currently Retrieved Context
{context}

## Current Answer Attempt
{answer}

## Task
The current context may not have enough information to fully answer the question.
Analyze what specific information is MISSING and generate a NEW search query to find it.

Focus on:
- Missing dates, times, or temporal information
- Missing names, places, or specific entities
- Missing causal relationships or reasons
- Missing details that would complete the answer

Generate a search query that is DIFFERENT from the original question and targets the missing information.
Return ONLY the search query, nothing else."""

        try:
            response = call_llm_with_retry(
                client=answer_client,
                model=self.answer_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,  # Slight variation for diversity
            )

            followup_query = response.strip()

            if verbose >= 2:
                print(f"        [Follow-up Query] {followup_query}")

            return followup_query if followup_query else question

        except Exception as e:
            if verbose >= 1:
                print(f"        [Follow-up Query] Failed: {e}, using original question")
            return question

    def _retrieve_with_expansion(self, question: str, top_k: int, verbose: int = 0):
        """
        Retrieve using multiple expanded queries and merge results (Perplexity-style).

        For each expanded query:
        1. Perform retrieval with the query
        2. Merge results with deduplication
        3. When same object appears multiple times, keep the HIGHEST score

        Args:
            question: Original question
            top_k: Final number of objects to return
            verbose: Verbosity level

        Returns:
            Merged RetrievalResult with deduplicated objects sorted by score
        """
        from cogcanvas.models import RetrievalResult
        import time as _time

        retrieval_start = _time.time()

        # Generate expanded queries
        queries = self._expand_query(question, self.query_expansion_n, verbose)

        if verbose >= 2:
            print(f"        [Retrieve+Expansion] Using {len(queries)} queries")

        # Track objects by ID -> (object, max_score, hit_count)
        # This allows us to:
        # 1. Deduplicate by obj.id
        # 2. Keep the highest score when an object is retrieved by multiple queries
        # 3. Track how many queries hit this object (for potential boosting)
        object_map = {}  # obj.id -> {"obj": obj, "max_score": float, "hit_count": int}

        # Retrieve more per query to ensure good coverage
        per_query_k = max(top_k // len(queries) + 3, 8)

        for query_idx, query in enumerate(queries):
            result = self._canvas.retrieve(
                query=query,
                top_k=per_query_k,
                method=self.retrieval_method,
                include_related=self.enable_graph_expansion,
                max_hops=self.graph_hops,
            )

            # Merge results, keeping max score for duplicates
            scores = result.scores or [0.5] * len(result.objects)
            for obj, score in zip(result.objects, scores):
                if obj.id in object_map:
                    # Object already seen - update max score and hit count
                    if score > object_map[obj.id]["max_score"]:
                        object_map[obj.id]["max_score"] = score
                    object_map[obj.id]["hit_count"] += 1
                else:
                    # New object
                    object_map[obj.id] = {
                        "obj": obj,
                        "max_score": score,
                        "hit_count": 1,
                    }

            if verbose >= 3:
                print(f"          Query {query_idx + 1}: Retrieved {len(result.objects)} objects")

        # Convert map to lists and apply hit count boosting
        # Objects retrieved by multiple queries get a small score boost
        all_objects = []
        all_scores = []

        for obj_id, data in object_map.items():
            obj = data["obj"]
            score = data["max_score"]
            hit_count = data["hit_count"]

            # Apply mild boosting for objects hit by multiple queries
            # This rewards objects that are relevant to multiple aspects of the question
            if hit_count > 1:
                # Boost by 5% per additional hit (capped at 15%)
                boost_factor = 1.0 + min(0.05 * (hit_count - 1), 0.15)
                score = min(score * boost_factor, 1.0)  # Cap at 1.0

            all_objects.append(obj)
            all_scores.append(score)

        # Sort by score (descending) and truncate
        if all_scores:
            sorted_pairs = sorted(
                zip(all_objects, all_scores),
                key=lambda x: x[1],
                reverse=True
            )
            all_objects = [p[0] for p in sorted_pairs]
            all_scores = [p[1] for p in sorted_pairs]

        retrieval_ms = (_time.time() - retrieval_start) * 1000

        if verbose >= 2:
            multi_hit_count = sum(1 for data in object_map.values() if data["hit_count"] > 1)
            print(f"        [Retrieve+Expansion] {len(all_objects)} unique objects, {multi_hit_count} multi-hit, {retrieval_ms:.0f}ms")

        return RetrievalResult(
            objects=all_objects[:top_k * 2],  # Keep more for potential reranking
            scores=all_scores[:top_k * 2],
            query=question,
            retrieval_time=retrieval_ms / 1000,  # Convert to seconds
        )

    def _classify_query_complexity(self, question: str) -> str:
        """
        Classify question complexity using LLM for query routing.

        Returns 'simple' or 'complex' to determine retrieval strategy:
        - 'simple': Single-round retrieval (fast, for direct fact queries)
        - 'complex': Multi-round retrieval (thorough, for reasoning questions)
        """
        # Use LLM to classify
        prompt = f"""Classify this question as 'simple' or 'complex' for a memory retrieval system.

Question: {question}

Classification criteria:
- SIMPLE: Direct fact lookup requiring ONE piece of information
  Examples: "What is X's job?", "Where does X live?", "What is X's email?"

- COMPLEX: Requires reasoning, inference, or connecting MULTIPLE facts
  Examples: "Why did X decide to...?", "What would X prefer?", "When did X do Y after Z?"
  Also complex: temporal ordering, cause-effect, preferences, hypotheticals

Return ONLY one word: 'simple' or 'complex'"""

        try:
            client = self._answer_client or self._client
            if client:
                response = client.chat.completions.create(
                    model=self.answer_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                result = response.choices[0].message.content.strip().lower()
                if 'complex' in result:
                    return 'complex'
                elif 'simple' in result:
                    return 'simple'
        except Exception as e:
            pass  # Fall back to rule-based

        # Fallback: rule-based classification
        return self._classify_query_complexity_rules(question)

    def _classify_query_complexity_rules(self, question: str) -> str:
        """Rule-based fallback for query complexity classification."""
        question_lower = question.lower()

        # Complex indicators - patterns that suggest multi-hop reasoning
        complex_patterns = [
            r'\bwhy\b',                    # Causal questions
            r'\bhow did\b',                # Process questions
            r'\bbecause\b',                # Causal reasoning
            r'\bled to\b',                 # Causal chain
            r'\bresult(?:ed)? in\b',       # Consequence
            r'\bbefore\b',                 # Temporal ordering
            r'\bafter\b',                  # Temporal ordering
            r'\bwhen did.*(?:after|before)\b',  # Complex temporal
            r'\bwhat would\b',             # Hypothetical
            r'\bwould.*prefer\b',          # Preference inference
            r'\blikely to\b',              # Probability inference
            r'\brelationship between\b',   # Multi-entity relationship
            r'\bconnection\b',             # Linking entities
            r'\binfluence[ds]?\b',         # Impact/influence
            r'\baffect(?:ed|s)?\b',        # Impact
            r'\bimpact\b',                 # Consequence
            r'\bconsequence\b',            # Result
        ]

        # Check complex patterns
        for pattern in complex_patterns:
            if re.search(pattern, question_lower):
                return 'complex'

        # Long questions are typically complex (need more context)
        word_count = len(question.split())
        if word_count > 15:
            return 'complex'

        # Default to simple for short, direct queries
        return 'simple'

    def answer_question(self, question: str, verbose: int = 0) -> AgentResponse:
        """
        Answer a recall question using canvas objects + retained history.

        Supports four modes based on configuration:
        1. Single-round retrieval (use_multi_round=False): Original behavior
        2. Multi-round retrieval (use_multi_round=True): Iterative retrieval with confidence checking
        3. Query routing (use_query_routing=True + use_multi_round=True): Route based on complexity
        4. Smart routing (use_smart_routing=True): Route based on retrieval result quality

        Enhanced Two-Stage Retrieval:
        1. Coarse retrieval: Get top-K candidates (20-50)
        2. Fine-grained reranking: Rerank to top-N (5-10) using LLM or reranker
        """
        # Smart routing: analyze retrieval result first, then decide strategy
        if self.use_smart_routing:
            return self._answer_with_smart_routing(question, verbose)

        # Query routing: classify complexity and route accordingly
        if self.use_query_routing and self.use_multi_round:
            complexity = self._classify_query_complexity(question)

            if verbose >= 1:
                q_preview = question[:50] + '...' if len(question) > 50 else question
                print(f"        [Query Routing] '{q_preview}' -> {complexity}")

            if complexity == 'simple':
                # Simple questions: fast single-round retrieval
                return self._answer_question_single_round(question, verbose)
            else:
                # Complex questions: thorough multi-round retrieval
                return self._answer_question_multi_round(question, verbose)

        # Standard routing based on use_multi_round flag
        elif self.use_multi_round:
            return self._answer_question_multi_round(question, verbose)
        else:
            return self._answer_question_single_round(question, verbose)

    def _answer_question_single_round(self, question: str, verbose: int = 0) -> AgentResponse:
        """
        Original single-round retrieval logic.

        This is the backward-compatible path when use_multi_round=False.
        """
        start_time = time.time()

        # Adaptive top-k based on question type
        effective_top_k = self._adaptive_top_k(question, self.retrieval_top_k)

        # Step 1: Retrieve relevant canvas objects
        # Query Expansion + Reranking (new SOTA path)
        if self.use_query_expansion:
            retrieval_start = time.time()
            # Use multi-query retrieval
            retrieval_result = self._retrieve_with_expansion(
                question, self.reranker_candidate_k if self.use_reranker else effective_top_k, verbose
            )
            retrieval_ms = (time.time() - retrieval_start) * 1000

            # Apply reranking if enabled
            if self.use_reranker and self._reranker and len(retrieval_result.objects) > 0:
                rerank_start = time.time()
                retrieval_result = self._apply_reranking(retrieval_result, question)
                rerank_ms = (time.time() - rerank_start) * 1000
                # Truncate to final top-k
                retrieval_result.objects = retrieval_result.objects[:effective_top_k]
                if retrieval_result.scores:
                    retrieval_result.scores = retrieval_result.scores[:effective_top_k]
                if verbose >= 2:
                    print(f"        [QueryExpansion+Rerank] {retrieval_ms:.0f}ms + {rerank_ms:.0f}ms, Objects: {len(retrieval_result.objects)}")
            else:
                # Just truncate without reranking
                retrieval_result.objects = retrieval_result.objects[:effective_top_k]
                if retrieval_result.scores:
                    retrieval_result.scores = retrieval_result.scores[:effective_top_k]

        # Two-stage retrieval if reranker is enabled (without query expansion)
        elif self.use_reranker and self._reranker:
            # Stage 1: Coarse retrieval with larger K (use reranker_candidate_k)
            coarse_k = self.reranker_candidate_k
            retrieval_start = time.time()
            retrieval_result = self._canvas.retrieve(
                query=question,
                top_k=coarse_k,
                method=self.retrieval_method,
                include_related=self.enable_graph_expansion,
                max_hops=self.graph_hops,
            )
            retrieval_ms = (time.time() - retrieval_start) * 1000

            # Apply N-hop graph expansion if needed (before reranking)
            if self.enable_graph_expansion and self.graph_hops > 1:
                retrieval_result = self._apply_multihop_expansion(
                    retrieval_result, question
                )

            # Stage 2: Reranking to top-K
            if len(retrieval_result.objects) > 0:
                rerank_start = time.time()
                retrieval_result = self._apply_reranking(retrieval_result, question)
                rerank_ms = (time.time() - rerank_start) * 1000
                # Truncate to final top-k (using adaptive value)
                retrieval_result.objects = retrieval_result.objects[:effective_top_k]
                if retrieval_result.scores:
                    retrieval_result.scores = retrieval_result.scores[:effective_top_k]
                if verbose >= 3:
                    print(f"        [Retrieval] {retrieval_ms:.0f}ms, Rerank: {rerank_ms:.0f}ms, Objects: {len(retrieval_result.objects)}, top_k={effective_top_k}")

        elif self.use_llm_filter:
            # LLM filtering (alternative to reranking)
            retrieval_result = self._canvas.retrieve_and_filter(
                query=question,
                candidate_k=self.filter_candidate_k,
                final_k=effective_top_k,  # Use adaptive value
                method=self.retrieval_method,
                include_related=self.enable_graph_expansion,
                use_llm_filter=True,
            )
        else:
            # Original single-stage retrieval (baseline)
            retrieval_result = self._canvas.retrieve(
                query=question,
                top_k=effective_top_k,  # Use adaptive value
                method=self.retrieval_method,
                include_related=self.enable_graph_expansion,
                max_hops=self.graph_hops,
            )

            # Apply N-hop graph expansion if needed
            if self.enable_graph_expansion and self.graph_hops > 1:
                retrieval_result = self._apply_multihop_expansion(
                    retrieval_result, question
                )

        # Step 2: Build context from retrieved objects
        # Increased to 5000 to unleash full potential (RAG uses ~1500-2000, we go bigger)
        canvas_context = self._canvas.inject(
            retrieval_result,
            format="compact",
            max_tokens=5000,
        )

        # Step 3: Build answer
        answer = self._extract_answer_from_context(
            question, retrieval_result, canvas_context, verbose=verbose
        )

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "num_objects_retrieved": len(retrieval_result.objects),
                "graph_expansion": self.enable_graph_expansion,
                "graph_hops": self.graph_hops if self.enable_graph_expansion else 0,
                "reranking_applied": self.use_reranker,
                "retrieval_scores": (
                    retrieval_result.scores[:3] if retrieval_result.scores else []
                ),
            },
        )

    def _answer_question_multi_round(self, question: str, verbose: int = 0) -> AgentResponse:
        """
        Multi-round iterative retrieval logic.

        Iteratively retrieves information until:
        1. Confidence threshold is met, OR
        2. Maximum rounds is reached

        Each round:
        1. Retrieve with current query
        2. Generate draft answer
        3. Check confidence
        4. If not confident, generate follow-up query and continue
        """
        from cogcanvas.models import RetrievalResult

        start_time = time.time()

        # Adaptive top-k based on question type
        effective_top_k = self._adaptive_top_k(question, self.retrieval_top_k)

        # Track all retrieved objects across rounds (deduplicated)
        all_objects = []
        all_scores = []
        seen_ids = set()

        # Current query starts as the original question
        current_query = question

        # Track retrieval rounds
        retrieval_rounds = 0
        final_answer = None
        final_confidence = 0.0

        if verbose >= 1:
            print(f"        [Multi-Round] Starting iterative retrieval (max_rounds={self.max_retrieval_rounds}, threshold={self.confidence_threshold})")

        for round_num in range(self.max_retrieval_rounds):
            retrieval_rounds = round_num + 1

            if verbose >= 2:
                print(f"        [Round {retrieval_rounds}] Query: {current_query[:80]}...")

            # Step 1: Retrieve for current query
            round_result = self._retrieve_single_round(
                current_query, effective_top_k, exclude_ids=seen_ids, verbose=verbose
            )

            # Step 2: Merge results (deduplicated)
            new_objects_count = 0
            for obj in round_result.objects:
                if obj.id not in seen_ids:
                    all_objects.append(obj)
                    # Use retrieval score if available, otherwise default
                    score = 0.5
                    if round_result.scores and len(round_result.scores) > len(all_objects) - 1:
                        idx = round_result.objects.index(obj)
                        if idx < len(round_result.scores):
                            score = round_result.scores[idx]
                    all_scores.append(score)
                    seen_ids.add(obj.id)
                    new_objects_count += 1

            if verbose >= 2:
                print(f"        [Round {retrieval_rounds}] New objects: {new_objects_count}, Total: {len(all_objects)}")

            # Step 3: Build context from ALL retrieved objects so far
            merged_result = RetrievalResult(
                objects=all_objects,
                scores=all_scores,
                query=question,
                retrieval_time=0,
            )
            canvas_context = self._canvas.inject(
                merged_result,
                format="compact",
                max_tokens=5000,
            )

            # Step 4: Generate draft answer
            draft_answer = self._extract_answer_from_context(
                question, merged_result, canvas_context, verbose=0  # Suppress inner verbose
            )

            # Step 5: Check confidence
            confidence = self._check_answer_confidence(
                question, canvas_context, draft_answer, verbose=verbose
            )
            final_answer = draft_answer
            final_confidence = confidence

            if verbose >= 2:
                print(f"        [Round {retrieval_rounds}] Confidence: {confidence:.2f}")

            # Step 6: If confident enough, stop
            if confidence >= self.confidence_threshold:
                if verbose >= 1:
                    print(f"        [Multi-Round] Converged at round {retrieval_rounds} (confidence={confidence:.2f})")
                break

            # Step 7: If not confident and not last round, generate follow-up query
            if round_num < self.max_retrieval_rounds - 1:
                current_query = self._generate_followup_query(
                    question, canvas_context, draft_answer, verbose=verbose
                )

        # Build final response
        latency = (time.time() - start_time) * 1000

        if verbose >= 1:
            print(f"        [Multi-Round] Completed in {retrieval_rounds} rounds, {len(all_objects)} objects, {latency:.0f}ms")

        return AgentResponse(
            answer=final_answer,
            latency_ms=latency,
            metadata={
                "num_objects_retrieved": len(all_objects),
                "graph_expansion": self.enable_graph_expansion,
                "graph_hops": self.graph_hops if self.enable_graph_expansion else 0,
                "reranking_applied": self.use_reranker,
                "retrieval_scores": all_scores[:3] if all_scores else [],
                "multi_round": True,
                "retrieval_rounds": retrieval_rounds,
                "final_confidence": final_confidence,
            },
        )

    def _retrieve_single_round(
        self,
        query: str,
        top_k: int,
        exclude_ids: set = None,
        verbose: int = 0
    ):
        """
        Perform a single round of retrieval.

        This is a helper for multi-round retrieval that handles:
        - Query expansion (if enabled)
        - Reranking (if enabled)
        - Exclusion of already-retrieved objects

        Args:
            query: The query string
            top_k: Number of objects to retrieve
            exclude_ids: Set of object IDs to exclude (already retrieved)
            verbose: Verbosity level

        Returns:
            RetrievalResult with retrieved objects
        """
        from cogcanvas.models import RetrievalResult

        exclude_ids = exclude_ids or set()

        # Use query expansion if enabled
        if self.use_query_expansion:
            retrieval_result = self._retrieve_with_expansion(
                query, self.reranker_candidate_k if self.use_reranker else top_k, verbose
            )
        else:
            # Standard retrieval
            retrieval_result = self._canvas.retrieve(
                query=query,
                top_k=self.reranker_candidate_k if self.use_reranker else top_k,
                method=self.retrieval_method,
                include_related=self.enable_graph_expansion,
                max_hops=self.graph_hops,
            )

        # Apply N-hop graph expansion if needed
        if self.enable_graph_expansion and self.graph_hops > 1:
            retrieval_result = self._apply_multihop_expansion(retrieval_result, query)

        # Apply reranking if enabled
        if self.use_reranker and self._reranker and len(retrieval_result.objects) > 0:
            retrieval_result = self._apply_reranking(retrieval_result, query)

        # Filter out already-seen objects and truncate
        filtered_objects = []
        filtered_scores = []
        for i, obj in enumerate(retrieval_result.objects):
            if obj.id not in exclude_ids:
                filtered_objects.append(obj)
                if retrieval_result.scores and i < len(retrieval_result.scores):
                    filtered_scores.append(retrieval_result.scores[i])
                else:
                    filtered_scores.append(0.5)

        # Truncate to top_k
        filtered_objects = filtered_objects[:top_k]
        filtered_scores = filtered_scores[:top_k]

        return RetrievalResult(
            objects=filtered_objects,
            scores=filtered_scores,
            query=query,
            retrieval_time=retrieval_result.retrieval_time if hasattr(retrieval_result, 'retrieval_time') else 0,
        )

    def _apply_multihop_expansion(self, retrieval_result, query: str):
        """
        Apply N-hop graph expansion to retrieval results.

        Args:
            retrieval_result: Initial retrieval result
            query: The query string

        Returns:
            Updated RetrievalResult with expanded objects
        """
        from cogcanvas.models import RetrievalResult

        # Get all expanded objects using N-hop traversal with BFS to track distances
        expanded_objects = []
        expanded_scores = []
        seen_ids = set()

        # Process each initial object
        for obj, score in zip(retrieval_result.objects, retrieval_result.scores or []):
            if obj.id not in seen_ids:
                expanded_objects.append(obj)
                expanded_scores.append(score)
                seen_ids.add(obj.id)

            # Perform BFS to get N-hop neighbors with distance tracking
            visited = {obj.id}
            current_level = [(obj.id, 0)]  # (node_id, distance)
            hop_distances = {}  # Map node_id -> min_distance

            # BFS traversal
            while current_level:
                next_level = []
                for node_id, dist in current_level:
                    if dist >= self.graph_hops:
                        continue

                    neighbors = self._canvas._graph.get_neighbors(
                        node_id, direction="both"
                    )
                    for neighbor_id in neighbors:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            next_dist = dist + 1
                            hop_distances[neighbor_id] = next_dist
                            next_level.append((neighbor_id, next_dist))

                current_level = next_level

            # Add related objects with decayed scores based on hop distance
            for related_id, hop_distance in hop_distances.items():
                if related_id not in seen_ids:
                    related_obj = self._canvas._objects.get(related_id)
                    if related_obj:
                        expanded_objects.append(related_obj)
                        # Decay score based on hop distance: 0.8^hop
                        decay_factor = 0.8**hop_distance
                        expanded_scores.append(score * decay_factor)
                        seen_ids.add(related_id)

        return RetrievalResult(
            objects=expanded_objects,
            scores=expanded_scores,
            query=query,
            retrieval_time=retrieval_result.retrieval_time,
        )

    def _apply_reranking(self, retrieval_result, query: str):
        """
        Apply reranking to retrieval results.

        Args:
            retrieval_result: Retrieval result to rerank
            query: The query string

        Returns:
            Reranked RetrievalResult
        """
        from cogcanvas.models import RetrievalResult

        # Prepare documents for reranking (use content + quote)
        documents = []
        for obj in retrieval_result.objects:
            doc_text = obj.content or ""
            if obj.quote:
                doc_text = f"{doc_text} | {obj.quote}"
            documents.append(doc_text)

        # Rerank
        reranked_indices = self._reranker.rerank(
            query=query, documents=documents, top_k=None  # Keep all, just reorder
        )

        # Reorder objects and scores based on reranking
        reranked_objects = []
        reranked_scores = []
        for idx, rerank_score in reranked_indices:
            reranked_objects.append(retrieval_result.objects[idx])
            # Use reranking score
            reranked_scores.append(rerank_score)

        return RetrievalResult(
            objects=reranked_objects,
            scores=reranked_scores,
            query=query,
            retrieval_time=retrieval_result.retrieval_time,
        )

    def _extract_answer_from_context(
        self,
        question: str,
        retrieval_result,
        canvas_context: str,
        verbose: int = 0,
    ) -> str:
        """
        Extract answer from retrieved canvas objects.
        """
        # 1. Use LLM if enabled and available (prefer _answer_client for answer model)
        answer_client = self._answer_client or self._client
        if self.use_real_llm_for_answer and answer_client:

            if self.prompt_style == "cot_fusion":
                # Multi-Artifact Fusion Prompt - Connects related artifacts for complex reasoning
                prompt = f"""You are a reasoning agent with access to a memory graph containing 8 types of nodes:

**Task-oriented:**
- DECISION: Choices made (e.g., "Use AWS")
- KEY_FACT: Constraints/numbers (e.g., "Budget $500")
- REMINDER: Rules/preferences
- TODO: Action items
- INSIGHT: Conclusions

**Social/Personal:**
- PERSON_ATTRIBUTE: Personal traits/status (e.g., "Caroline is a counselor")
- EVENT: Activities with time (e.g., "Attended support group on May 7")
- RELATIONSHIP: Interpersonal connections (e.g., "Caroline and Melanie are friends")

## Retrieved Memory Nodes
{canvas_context}

## Question
{question}

## Multi-Artifact Fusion Protocol
1. **IDENTIFY**: List ALL relevant artifacts from the memory
2. **CONNECT**: Find relationships between artifacts:
   - EVENT ↔ PERSON_ATTRIBUTE (who did what)
   - RELATIONSHIP ↔ PERSON_ATTRIBUTE (who knows whom)
   - DECISION ↔ KEY_FACT (why this choice)
3. **CHAIN**: Build explicit reasoning: "Because [A], therefore [B]"
4. **ANSWER**: Synthesize a complete answer

## Response Format
**Artifacts Used:**
- [Type] Content

**Reasoning Chain:**
Because [Artifact 1] → [Artifact 2] → Therefore [Conclusion]

**Answer:**
[Direct answer]
"""
            elif self.prompt_style == "cot_v2":
                # Enhanced Chain-of-Thought Prompt (V2 - Explicit Causal Reasoning)
                prompt = f"""You are a causal reasoning agent. Your memory contains a knowledge graph with 8 types of nodes:
- DECISION: Choices made (e.g., "Use AWS")
- KEY_FACT: Constraints/numbers (e.g., "Budget $500")
- REMINDER: Rules/preferences
- TODO: Action items
- INSIGHT: Conclusions
- PERSON_ATTRIBUTE: Personal traits/status
- EVENT: Activities with time
- RELATIONSHIP: Interpersonal connections

## Retrieved Memory Nodes
{canvas_context}

## Task
Answer: {question}

## Reasoning Protocol
1. IDENTIFY: Which nodes are relevant? List them.
2. CHAIN: What causal links exist? (Constraint → Decision → Impact)
3. SYNTHESIZE: Combine into a complete answer.

## Your Response
**Reasoning:**
[Show your causal chain here]

**Answer:**
[Final answer]
"""
            elif self.prompt_style == "cot_temporal":
                # Enhanced CoT Prompt with Temporal Reasoning Focus
                prompt = f"""You are an expert reasoning agent with access to a structured memory graph (CogCanvas).
Your goal is to answer questions by connecting discrete facts and tracking changes over time.

## Memory Context (Retrieved Nodes)
{canvas_context}

## Instructions for Temporal Reasoning
1. **Identify Temporal Patterns**: Look for EVENT nodes with timestamps. Pay attention to:
   - When did something happen?
   - How did opinions/decisions/states change over time?
   - What is the timeline of events?

2. **Track State Changes**: Notice when a PERSON_ATTRIBUTE, DECISION, or status changes:
   - "Before X, they thought Y"
   - "After Z happened, they changed to W"

3. **Infer Causality**: Even if not explicitly linked, reason about cause-effect:
   - Events → Decisions (e.g., "After visiting museum → decided to learn art")
   - Constraints → Choices (e.g., "Budget $500 → chose AWS")

4. **Synthesize Answer**: Provide a complete answer that:
   - Answers WHEN if it's a temporal question
   - Explains WHY if there's a causal chain
   - Acknowledges uncertainty if information is incomplete

## Question
{question}

## Answer
"""
            elif self.prompt_style == "cot":
                # Chain-of-Thought Prompt - Optimized for Direct Answers
                prompt = f"""Answer this question based on the retrieved memory context.

## Memory Context
{canvas_context}

## Question
{question}

## RULES (MUST FOLLOW):
1. **DIRECT ANSWER FIRST**: Start with the answer in the first sentence.
   - "when" question -> Start with the date (e.g., "May 15, 2023")
   - "who" question -> Start with the name (e.g., "Sarah")
   - "what" question -> Start with the exact thing (e.g., "Adoption agencies")
   - "where" question -> Start with the place (e.g., "Central Park")
   - yes/no question -> Start with "Yes" or "No"

2. **NEVER SAY**: "no information", "not mentioned", "I don't know", "context does not"

3. **ALWAYS ANSWER**: Give your best answer based on context. For multi-hop, connect facts.

4. **USE ABSOLUTE DATES**: Use dates from [Date: ...] brackets, not "yesterday" or "last week".

5. **BE CONCISE**: 1-2 sentences max.

## Answer:
"""
            else:
                # Direct Prompt (The Baseline) - Optimized for Direct Answers
                prompt = f"""Answer the question based on the memory context below.

## Memory Context
{canvas_context}

## Question
{question}

## RULES:
1. Start with the direct answer (date for "when", name for "who", etc.)
2. Never say "no information found" - always give your best answer based on available context
3. Be concise: 1-3 sentences max

## Answer:
"""

            try:
                llm_start = time.time()
                response = call_llm_with_retry(
                    client=answer_client,
                    model=self.answer_model,  # Use answer model (glm-4-flash) for QA
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0,
                )
                llm_ms = (time.time() - llm_start) * 1000
                if verbose >= 3:
                    print(f"        [LLM Answer] {llm_ms:.0f}ms (model={self.answer_model})")
                return response
            except Exception as e:
                print(f"LLM generation failed: {e}")
                # Fallback to heuristic

        # 2. Fallback Heuristic
        if not retrieval_result.objects:
            return "I don't have information about that."

        # Combine all top-k objects' content and quotes
        answers = []
        for obj in retrieval_result.objects[: self.retrieval_top_k]:
            if obj.quote:
                answers.append(obj.quote)
            if obj.content:
                answers.append(obj.content)

        return " | ".join(answers)

    # =========================================================================
    # Smart Routing Methods (Retrieval-Result-Based Routing)
    # =========================================================================

    def _analyze_retrieval_quality(self, question: str, retrieval_result) -> dict:
        """
        Analyze retrieval result quality and return features for routing decision.

        Args:
            question: The original question
            retrieval_result: RetrievalResult from Canvas.retrieve()

        Returns:
            Dict with analysis features:
            - top_score: Highest retrieval score
            - avg_score: Average retrieval score
            - score_spread: Difference between max and min scores
            - is_temporal: Whether question contains temporal keywords
            - is_multi_entity: Whether question suggests multiple entities
            - num_results: Number of retrieved objects
        """
        scores = retrieval_result.scores or []

        if not scores:
            return {
                "quality": "poor",
                "strategy": "expand",
                "top_score": 0.0,
                "avg_score": 0.0,
                "score_spread": 0.0,
                "is_temporal": False,
                "is_multi_entity": False,
                "num_results": 0,
            }

        top_score = scores[0] if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        score_spread = max(scores) - min(scores) if len(scores) > 1 else 0.0

        # Check for temporal keywords
        time_words = [
            "when", "before", "after", "first", "last", "earlier", "later",
            "date", "time", "year", "month", "day", "recently", "ago",
            "during", "while", "until", "since"
        ]
        q_lower = question.lower()
        is_temporal = any(w in q_lower for w in time_words)

        # Check for multi-entity patterns
        multi_entity_words = [" and ", " both ", " between ", " with ", " or "]
        is_multi_entity = any(w in q_lower for w in multi_entity_words)

        return {
            "top_score": top_score,
            "avg_score": avg_score,
            "score_spread": score_spread,
            "is_temporal": is_temporal,
            "is_multi_entity": is_multi_entity,
            "num_results": len(scores),
        }

    def _smart_route(self, question: str, retrieval_result, analysis: dict) -> str:
        """
        Decide retrieval strategy based on retrieval result quality.

        Routing Logic:
        1. High confidence single match (top_score > 0.85, spread > 0.3) -> "direct"
        2. Temporal question with multiple results -> "temporal_sort"
        3. Multi-entity OR uniform scores (spread < 0.15) -> "multi_round"
        4. Low scores (top < 0.5) -> "expand"
        5. Default -> "direct"

        Args:
            question: The original question
            retrieval_result: RetrievalResult object
            analysis: Quality analysis from _analyze_retrieval_quality

        Returns:
            Strategy string: "direct", "temporal_sort", "multi_round", or "expand"
        """
        top_score = analysis["top_score"]
        score_spread = analysis["score_spread"]
        is_temporal = analysis["is_temporal"]
        is_multi_entity = analysis["is_multi_entity"]
        num_results = analysis["num_results"]

        # Strategy 1: High confidence single match -> direct answer
        if top_score > self.smart_routing_high_score and score_spread > 0.3:
            return "direct"

        # Strategy 2: Low scores -> query expansion (prioritize getting better results first)
        if top_score < self.smart_routing_low_score:
            return "expand"

        # Strategy 3: Temporal question with good results -> temporal sort
        # Only use temporal_sort if we have decent results (top_score >= 0.5)
        if is_temporal and num_results >= 3 and top_score >= self.smart_routing_low_score:
            return "temporal_sort"

        # Strategy 4: Multi-entity question OR uniform scores -> multi-round
        if is_multi_entity or score_spread < self.smart_routing_spread_threshold:
            return "multi_round"

        # Default: direct answer
        return "direct"

    def _answer_with_smart_routing(self, question: str, verbose: int = 0) -> AgentResponse:
        """
        Answer question using smart routing based on retrieval result quality.

        Flow:
        1. First-round retrieval
        2. Analyze retrieval quality
        3. Route to appropriate strategy:
           - direct: Answer immediately with current results
           - temporal_sort: Sort by time, then answer
           - multi_round: Use multi-round iterative retrieval
           - expand: Use query expansion for better coverage

        Args:
            question: The question to answer
            verbose: Verbosity level

        Returns:
            AgentResponse with answer and metadata
        """
        start_time = time.time()

        # Adaptive top-k based on question type
        effective_top_k = self._adaptive_top_k(question, self.retrieval_top_k)

        # First-round retrieval
        first_result = self._canvas.retrieve(
            query=question,
            top_k=self.reranker_candidate_k if self.use_reranker else effective_top_k,
            method=self.retrieval_method,
            include_related=self.enable_graph_expansion,
            max_hops=self.graph_hops,
        )

        # Apply reranking if enabled
        if self.use_reranker and self._reranker and len(first_result.objects) > 0:
            first_result = self._apply_reranking(first_result, question)

        # Analyze retrieval quality
        analysis = self._analyze_retrieval_quality(question, first_result)
        strategy = self._smart_route(question, first_result, analysis)

        if verbose >= 1:
            print(f"        [Smart Route] top={analysis['top_score']:.2f}, "
                  f"spread={analysis['score_spread']:.2f}, "
                  f"temporal={analysis['is_temporal']}, "
                  f"multi_entity={analysis['is_multi_entity']} -> {strategy}")

        # Route to appropriate strategy
        if strategy == "direct":
            return self._answer_direct(question, first_result, effective_top_k, verbose, start_time)
        elif strategy == "temporal_sort":
            return self._answer_with_temporal_sort(question, first_result, effective_top_k, verbose, start_time)
        elif strategy == "multi_round":
            # Delegate to existing multi-round method (it will start fresh)
            return self._answer_question_multi_round(question, verbose)
        elif strategy == "expand":
            return self._answer_with_expansion(question, effective_top_k, verbose, start_time)
        else:
            # Fallback to direct
            return self._answer_direct(question, first_result, effective_top_k, verbose, start_time)

    def _answer_direct(
        self,
        question: str,
        retrieval_result,
        effective_top_k: int,
        verbose: int = 0,
        start_time: float = None
    ) -> AgentResponse:
        """
        Answer directly using the provided retrieval result.

        Args:
            question: The question
            retrieval_result: Pre-computed retrieval result
            effective_top_k: Number of objects to use
            verbose: Verbosity level
            start_time: Start time for latency calculation

        Returns:
            AgentResponse
        """
        if start_time is None:
            start_time = time.time()

        # Truncate to effective top-k
        retrieval_result.objects = retrieval_result.objects[:effective_top_k]
        if retrieval_result.scores:
            retrieval_result.scores = retrieval_result.scores[:effective_top_k]

        # Build context and generate answer
        canvas_context = self._canvas.inject(
            retrieval_result,
            format="compact",
            max_tokens=5000,
        )

        answer = self._extract_answer_from_context(
            question, retrieval_result, canvas_context, verbose=verbose
        )

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "num_objects_retrieved": len(retrieval_result.objects),
                "graph_expansion": self.enable_graph_expansion,
                "graph_hops": self.graph_hops if self.enable_graph_expansion else 0,
                "reranking_applied": self.use_reranker,
                "retrieval_scores": (
                    retrieval_result.scores[:3] if retrieval_result.scores else []
                ),
                "smart_routing": True,
                "routing_strategy": "direct",
            },
        )

    def _answer_with_temporal_sort(
        self,
        question: str,
        retrieval_result,
        effective_top_k: int,
        verbose: int = 0,
        start_time: float = None
    ) -> AgentResponse:
        """
        Answer with temporal sorting of retrieval results.

        Sorts objects by turn_id (or event_time if available) to provide
        chronological context for temporal questions.

        Args:
            question: The question
            retrieval_result: Pre-computed retrieval result
            effective_top_k: Number of objects to use
            verbose: Verbosity level
            start_time: Start time for latency calculation

        Returns:
            AgentResponse
        """
        from cogcanvas.models import RetrievalResult

        if start_time is None:
            start_time = time.time()

        # Sort objects by turn_id (chronological order)
        objects_with_scores = list(zip(
            retrieval_result.objects,
            retrieval_result.scores or [0.5] * len(retrieval_result.objects)
        ))

        # Sort by turn_id (ascending for chronological order)
        sorted_pairs = sorted(
            objects_with_scores,
            key=lambda x: getattr(x[0], 'turn_id', 0) or 0
        )

        sorted_objects = [p[0] for p in sorted_pairs][:effective_top_k]
        sorted_scores = [p[1] for p in sorted_pairs][:effective_top_k]

        sorted_result = RetrievalResult(
            objects=sorted_objects,
            scores=sorted_scores,
            query=question,
            retrieval_time=retrieval_result.retrieval_time if hasattr(retrieval_result, 'retrieval_time') else 0,
        )

        if verbose >= 2:
            turn_ids = [getattr(obj, 'turn_id', '?') for obj in sorted_objects[:5]]
            print(f"        [Temporal Sort] Sorted by turn_id: {turn_ids}...")

        # Build context with temporal ordering emphasis
        canvas_context = self._canvas.inject(
            sorted_result,
            format="compact",
            max_tokens=5000,
        )

        answer = self._extract_answer_from_context(
            question, sorted_result, canvas_context, verbose=verbose
        )

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "num_objects_retrieved": len(sorted_result.objects),
                "graph_expansion": self.enable_graph_expansion,
                "graph_hops": self.graph_hops if self.enable_graph_expansion else 0,
                "reranking_applied": self.use_reranker,
                "retrieval_scores": sorted_scores[:3] if sorted_scores else [],
                "smart_routing": True,
                "routing_strategy": "temporal_sort",
            },
        )

    def _answer_with_expansion(
        self,
        question: str,
        effective_top_k: int,
        verbose: int = 0,
        start_time: float = None
    ) -> AgentResponse:
        """
        Answer using query expansion for better recall.

        Uses _retrieve_with_expansion to generate multiple query variants
        and merge their results.

        Args:
            question: The question
            effective_top_k: Number of objects to use
            verbose: Verbosity level
            start_time: Start time for latency calculation

        Returns:
            AgentResponse
        """
        if start_time is None:
            start_time = time.time()

        # Use query expansion retrieval
        retrieval_result = self._retrieve_with_expansion(
            question,
            self.reranker_candidate_k if self.use_reranker else effective_top_k,
            verbose
        )

        # Apply reranking if enabled
        if self.use_reranker and self._reranker and len(retrieval_result.objects) > 0:
            retrieval_result = self._apply_reranking(retrieval_result, question)

        # Truncate to effective top-k
        retrieval_result.objects = retrieval_result.objects[:effective_top_k]
        if retrieval_result.scores:
            retrieval_result.scores = retrieval_result.scores[:effective_top_k]

        # Build context and answer
        canvas_context = self._canvas.inject(
            retrieval_result,
            format="compact",
            max_tokens=5000,
        )

        answer = self._extract_answer_from_context(
            question, retrieval_result, canvas_context, verbose=verbose
        )

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "num_objects_retrieved": len(retrieval_result.objects),
                "graph_expansion": self.enable_graph_expansion,
                "graph_hops": self.graph_hops if self.enable_graph_expansion else 0,
                "reranking_applied": self.use_reranker,
                "retrieval_scores": (
                    retrieval_result.scores[:3] if retrieval_result.scores else []
                ),
                "smart_routing": True,
                "routing_strategy": "expand",
            },
        )

    def get_canvas_stats(self) -> dict:
        """Get statistics about the current canvas state."""
        if not self._canvas:
            return {}
        return self._canvas.stats()
