"""
Native Baseline Agent: Simple truncation without memory augmentation.

This is the most basic baseline that simulates standard context truncation.
After compression, only the most recent N turns are available - early facts
are completely lost.

Expected performance: ~0% recall on facts planted before compression window.
"""

from typing import List, Optional
import time
import os

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from experiments.llm_utils import call_llm_with_retry


class NativeAgent(Agent):
    """
    Native baseline agent - no memory augmentation.

    On compression:
    - History is truncated to only retained turns
    - ALL early context is lost (the problem we're solving)

    On answer:
    - Uses only retained history to answer
    - Facts from early turns are NOT recoverable
    """

    def __init__(
        self,
        model: str = None,
        retain_recent: int = 5,
    ):
        """
        Initialize NativeAgent.

        Args:
            model: Model name for answer generation (None = load from env MODEL_DEFAULT)
            retain_recent: Number of recent turns to keep (for reference, actual
                          truncation is controlled by runner's on_compression)
        """
        from dotenv import load_dotenv

        load_dotenv()

        self.retain_recent = retain_recent

        # Use MODEL_DEFAULT by default (same as CogCanvas for fair comparison)
        self.model = model or os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")

        # Initialize LLM client
        self._client = None
        self._init_client()

        # State
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []

    def _init_client(self):
        """Initialize LLM client using API_KEY and API_BASE from .env."""
        try:
            from openai import OpenAI

            # Use ANSWER_API_* for answering questions
            api_key = os.getenv("ANSWER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("ANSWER_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if not api_key:
                print("Warning: ANSWER_API_KEY/API_KEY not set, using mock responses")
                self._client = None
                return

            self._client = OpenAI(
                api_key=api_key,
                base_url=api_base,
            )
        except ImportError:
            print("Warning: openai not installed, using mock responses")
            self._client = None

    @property
    def name(self) -> str:
        return f"Native(model={self.model}, retain={self.retain_recent})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self._history = []
        self._retained_history = []

    def process_turn(self, turn: ConversationTurn) -> None:
        """
        Process a conversation turn.

        Simply stores the turn in history. No extraction or augmentation.
        """
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle compression event.

        This is where the critical information loss happens:
        - Only retained_turns survive
        - ALL earlier context (including planted facts) is LOST

        This is the fundamental limitation we're trying to solve with CogCanvas.
        """
        # Truncate to only retained turns
        self._retained_history = retained_turns

        # Clear full history (simulating actual context truncation)
        self._history = list(retained_turns)

    def answer_question(self, question: str) -> AgentResponse:
        """
        Answer a recall question using ONLY retained history.

        Since early facts are truncated, this will fail for questions
        about information from early turns.
        """
        start_time = time.time()

        # Build context from retained history only
        context = self._format_history(self._retained_history)

        # Generate answer
        answer = self._generate_answer(context, question)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "retained_turns": len(self._retained_history),
                "context_length": len(context),
            },
        )

    def _format_history(self, turns: List[ConversationTurn]) -> str:
        """Format conversation turns into a readable context string."""
        if not turns:
            return "[No conversation history available]"

        lines = []
        for turn in turns:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")
            lines.append("")

        return "\n".join(lines)

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer using LLM."""
        prompt = f"""You are an expert reasoning agent. Your goal is to answer the user's question by connecting discrete facts from the retrieved information.

## Retrieved Context
{context}

## Instructions
1. Analyze the retrieved information carefully
2. Even if pieces of information are not explicitly linked, use your reasoning to infer relationships
3. Synthesize a complete answer that explains the reasoning process

## Question
{question}

## Answer
"""

        if self._client is None:
            # Mock response for testing without API
            return "I don't have enough information to answer this question."

        try:
            return call_llm_with_retry(
                client=self._client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
        except Exception as e:
            return f"Error generating answer: {e}"
