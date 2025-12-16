"""Base class for all experiment methods (baselines + ours)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class MethodResult:
    """Result of running a method on a conversation."""
    method_name: str
    conversation_id: str

    # Per-fact results
    fact_results: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregate metrics
    retention_score: float = 0.0      # % of facts correctly recalled
    consistency_score: float = 0.0    # % of non-contradictory answers

    # Cost metrics
    total_tokens_used: int = 0
    injection_tokens: int = 0         # Extra tokens from method (e.g., canvas injection)

    # Timing
    total_time: float = 0.0

    # Raw data for debugging
    raw_answers: List[str] = field(default_factory=list)
    context_at_test: str = ""         # What context was available during testing


@dataclass
class FactTestResult:
    """Result of testing a single planted fact."""
    fact_id: str
    question: str
    ground_truth: str
    model_answer: str
    is_correct: bool
    confidence: float = 0.0           # Model's confidence if available
    grounding_verified: bool = False  # True if answer traces back to quote


class BaseMethod(ABC):
    """
    Abstract base class for all evaluation methods.

    Each method implements:
    1. process_conversation(): How to handle the conversation history
    2. answer_question(): How to answer test questions after compression
    """

    name: str = "base"

    def __init__(self, llm_client: Any = None, config: Dict = None):
        """
        Initialize the method.

        Args:
            llm_client: LLM client for generating answers (OpenAI/Anthropic)
            config: Method-specific configuration
        """
        self.llm_client = llm_client
        self.config = config or {}

    @abstractmethod
    def process_conversation(
        self,
        turns: List[Dict[str, str]],
        compression_turn: int,
    ) -> Dict[str, Any]:
        """
        Process a conversation up to the compression point.

        This is where each method differs:
        - Native: Just stores turns, will truncate later
        - Summarization: Generates summary
        - RAG: Indexes turns for retrieval
        - CogCanvas: Extracts canvas objects

        Args:
            turns: List of {"user": str, "assistant": str} dicts
            compression_turn: Turn number where compression happens

        Returns:
            State dict to be used in answer_question()
        """
        pass

    @abstractmethod
    def get_context_for_question(
        self,
        state: Dict[str, Any],
        question: str,
        recent_turns: List[Dict[str, str]],
    ) -> str:
        """
        Get the context to use when answering a question.

        This is called AFTER compression. Each method provides different context:
        - Native: Only recent_turns (truncated)
        - Summarization: Summary + recent_turns
        - RAG: Retrieved turns + recent_turns
        - CogCanvas: Injected canvas + recent_turns

        Args:
            state: State from process_conversation()
            question: The test question
            recent_turns: Turns retained after compression

        Returns:
            Context string to include in the prompt
        """
        pass

    def answer_question(
        self,
        state: Dict[str, Any],
        question: str,
        recent_turns: List[Dict[str, str]],
    ) -> str:
        """
        Answer a test question using the method's context.

        Args:
            state: State from process_conversation()
            question: The test question
            recent_turns: Turns retained after compression

        Returns:
            The model's answer
        """
        context = self.get_context_for_question(state, question, recent_turns)

        prompt = self._build_answer_prompt(context, question)

        if self.llm_client is None:
            # Mock response for testing
            return "[MOCK] I don't have enough context to answer this question."

        return self._call_llm(prompt)

    def _build_answer_prompt(self, context: str, question: str) -> str:
        """Build the prompt for answering a question."""
        return f"""Based on the following conversation context, answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer this."

## Conversation Context
{context}

## Question
{question}

## Answer
Provide a concise, direct answer based only on the information in the context above."""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM to generate a response."""
        if hasattr(self.llm_client, 'chat'):
            # OpenAI-style client
            response = self.llm_client.chat.completions.create(
                model=self.config.get("model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            return response.choices[0].message.content
        elif hasattr(self.llm_client, 'messages'):
            # Anthropic-style client
            response = self.llm_client.messages.create(
                model=self.config.get("model", "claude-3-5-haiku-latest"),
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        else:
            raise ValueError("Unknown LLM client type")

    def _format_turns(self, turns: List[Dict[str, str]]) -> str:
        """Format turns into a readable string."""
        lines = []
        for i, turn in enumerate(turns, 1):
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
            lines.append("")
        return "\n".join(lines)

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
