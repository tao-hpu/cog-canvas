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
        extractor_model: str = None,  # None = load from env
        embedding_model: str = None,  # None = load from env
        retrieval_top_k: int = 10,
        enable_graph_expansion: bool = True,  # New flag for ablation
        graph_hops: int = 1,  # Number of hops for graph expansion (default 1 for backward compat)
        use_reranker: bool = False,  # Disable reranking to test baseline speed
        reranker_type: str = "api",  # Use BGE reranker API (MUCH faster than LLM)
        reranker_candidate_k: int = 20,  # Retrieve top-20 before reranking to top-k
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
        # VAGE Parameters
        enable_vage: bool = False,  # Enable Vulnerability-Aware Greedy Extraction
        use_learned_vage: bool = False,  # Use learned vulnerability model
        vage_budget_k: int = 10,  # Max objects to keep per turn
        vage_mode: str = "off",  # "off" | "standard" | "chain" (chain uses graph-aware selection)
        vage_verbose: bool = False,  # Print detailed VAGE logs
    ):
        """
        Initialize CogCanvas agent.

        Args:
            extractor_model: Model for extraction
            embedding_model: Model for embeddings
            retrieval_top_k: Number of objects to retrieve (default: 10)
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
            enable_vage: Enable VAGE for optimal artifact selection
            use_learned_vage: Use learned vulnerability model instead of heuristics
            vage_budget_k: Max objects to keep per turn when VAGE is enabled
            vage_mode: VAGE mode - "off", "standard" (original), or "chain" (graph-aware)
            vage_verbose: Print detailed VAGE progress logs
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()

        # Load from .env if not specified
        self.extractor_model = extractor_model or os.getenv(
            "MODEL_DEFAULT", "gpt-4o-mini"
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

        # VAGE config
        self.enable_vage = enable_vage
        self.use_learned_vage = use_learned_vage
        self.vage_budget_k = vage_budget_k
        self.vage_mode = vage_mode
        self.vage_verbose = vage_verbose

        # Initialize LLM client for answering
        self._client = None
        if self.use_real_llm_for_answer:
            self._init_client()

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
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            import os

            api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if api_key:
                self._client = OpenAI(api_key=api_key, base_url=api_base)
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

    def answer_question(self, question: str, verbose: int = 0) -> AgentResponse:
        """
        Answer a recall question using canvas objects + retained history.

        Enhanced Two-Stage Retrieval:
        1. Coarse retrieval: Get top-K candidates (20-50)
        2. Fine-grained reranking: Rerank to top-N (5-10) using LLM or reranker
        """
        start_time = time.time()

        # Adaptive top-k based on question type
        effective_top_k = self._adaptive_top_k(question, self.retrieval_top_k)

        # Step 1: Retrieve relevant canvas objects
        # Two-stage retrieval if reranker is enabled
        if self.use_reranker and self._reranker:
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
        # 1. Use LLM if enabled and available
        if self.use_real_llm_for_answer and self._client:

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
                # Chain-of-Thought Prompt (The SOTA one)
                prompt = f"""You are an expert reasoning agent with access to a structured memory graph (CogCanvas).
Your goal is to answer the user's question by connecting discrete facts from the memory.

## Memory Context (Retrieved Nodes)
{canvas_context}

## Instructions
1. Analyze the retrieved nodes. Look for "Constraints" (KeyFacts/Reminders) and "Decisions".
2. Even if the nodes are not explicitly linked, use your reasoning to infer causal relationships (e.g., "Budget is $500" likely caused "Choose Cheap Hosting").
3. Synthesize a complete answer that explains WHY things happened.

## Question
{question}

## Answer
"""
            else:
                # Direct Prompt (The Baseline)
                prompt = f"""Answer the question based on the provided CogCanvas memory context.
If the information is not available, say "I don't have enough information."

## Memory Context
{canvas_context}

## Question
{question}

## Answer
Provide a concise, direct answer."""

            try:
                llm_start = time.time()
                response = call_llm_with_retry(
                    client=self._client,
                    model=self.extractor_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0,
                )
                llm_ms = (time.time() - llm_start) * 1000
                if verbose >= 3:
                    print(f"        [LLM Answer] {llm_ms:.0f}ms")
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

    def get_canvas_stats(self) -> dict:
        """Get statistics about the current canvas state."""
        if not self._canvas:
            return {}
        return self._canvas.stats()
