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

from cogcanvas import Canvas
from cogcanvas.models import ObjectType


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
        retrieval_top_k: int = 5,
        enable_graph_expansion: bool = True,  # New flag for ablation
        use_real_llm_for_answer: bool = True, # Default to True for fair comparison
        # Ablation Parameters
        enable_temporal_heuristic: bool = True,
        retrieval_method: str = "hybrid",  # "semantic", "keyword", "hybrid"
        prompt_style: str = "cot",         # "direct", "cot"
    ):
        """
        Initialize CogCanvas agent.

        Args:
            extractor_model: Model for extraction
            embedding_model: Model for embeddings
            retrieval_top_k: Number of objects to retrieve
            enable_graph_expansion: Whether to use 1-hop graph expansion (Ablation)
            use_real_llm_for_answer: If True, use LLM for answer generation
            enable_temporal_heuristic: Enable temporal causality rule in graph construction
            retrieval_method: Retrieval strategy
            prompt_style: Prompting strategy
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()

        # Load from .env if not specified
        self.extractor_model = extractor_model or os.getenv("MODEL_WEAK_2", "gpt-4o-mini")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-large-zh-v1.5")

        self.retrieval_top_k = retrieval_top_k
        self.enable_graph_expansion = enable_graph_expansion
        self.use_real_llm_for_answer = use_real_llm_for_answer
        
        # Ablation config
        self.enable_temporal_heuristic = enable_temporal_heuristic
        self.retrieval_method = retrieval_method
        self.prompt_style = prompt_style

        # Initialize LLM client for answering
        self._client = None
        if self.use_real_llm_for_answer:
            self._init_client()

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

    @property
    def name(self) -> str:
        parts = []
        if self.enable_graph_expansion: parts.append("Graph")
        if self.enable_temporal_heuristic: parts.append("Time")
        if self.retrieval_method == "hybrid": parts.append("Hybrid")
        if self.prompt_style == "cot": parts.append("CoT")
        
        config_str = "+".join(parts) if parts else "Baseline"
        return f"CogCanvas({config_str})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self._canvas = Canvas(
            extractor_model=self.extractor_model,
            embedding_model=self.embedding_model,
            enable_temporal_heuristic=self.enable_temporal_heuristic,
        )
        self._history = []
        self._retained_history = []

    def process_turn(self, turn: ConversationTurn) -> None:
        """
        Process a conversation turn.

        Extracts canvas objects and stores the turn in history.
        """
        # Store in history
        self._history.append(turn)

        # Extract canvas objects from this turn
        self._canvas.extract(
            user=turn.user,
            assistant=turn.assistant,
            metadata={"turn_id": turn.turn_id},
        )

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle compression event.

        History is truncated to retained_turns, but CANVAS OBJECTS SURVIVE.
        This is the key advantage of CogCanvas!
        """
        # Truncate history (simulating context compression)
        self._retained_history = retained_turns

        # NOTE: Canvas objects are NOT cleared!
        # This is the whole point - they survive compression

    def answer_question(self, question: str) -> AgentResponse:
        """
        Answer a recall question using canvas objects + retained history.
        """
        start_time = time.time()

        # Step 1: Retrieve relevant canvas objects
        # Controlled by enable_graph_expansion flag
        retrieval_result = self._canvas.retrieve(
            query=question,
            top_k=self.retrieval_top_k,
            method=self.retrieval_method,
            include_related=self.enable_graph_expansion,
        )

        # Step 2: Build context from retrieved objects
        canvas_context = self._canvas.inject(
            retrieval_result,
            format="compact",
            max_tokens=800,
        )

        # Step 3: Build answer
        answer = self._extract_answer_from_context(question, retrieval_result, canvas_context)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "num_objects_retrieved": len(retrieval_result.objects),
                "graph_expansion": self.enable_graph_expansion,
                "retrieval_scores": retrieval_result.scores[:3] if retrieval_result.scores else [],
            },
        )

    def _extract_answer_from_context(
        self,
        question: str,
        retrieval_result,
        canvas_context: str,
    ) -> str:
        """
        Extract answer from retrieved canvas objects.
        """
        # 1. Use LLM if enabled and available
        if self.use_real_llm_for_answer and self._client:
            
            if self.prompt_style == "cot":
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
                response = self._client.chat.completions.create(
                    model=self.extractor_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM generation failed: {e}")
                # Fallback to heuristic
        
        # 2. Fallback Heuristic
        if not retrieval_result.objects:
            return "I don't have information about that."

        # Combine all top-k objects' content and quotes
        answers = []
        for obj in retrieval_result.objects[:self.retrieval_top_k]:
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
