"""
Summarization Baseline Agent: Compresses history via LLM summarization.

When context grows too large, this agent calls the LLM to generate a summary
of the conversation so far, then uses Summary + Recent turns as context.

Expected performance: Better than Native, but likely loses specific details
(e.g., exact numbers like "API rate limit is 100 requests per minute").
"""

from typing import List, Optional
import time
import os

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from experiments.llm_utils import call_llm_with_retry


class SummarizationAgent(Agent):
    """
    Summarization baseline agent - compresses via LLM summarization.

    On compression:
    - Generate summary of all conversation history
    - Keep summary + recent turns as context

    On answer:
    - Uses summary + retained history to answer
    - May lose specific details that weren't captured in summary
    """

    def __init__(
        self,
        model: str = None,
        retain_recent: int = 5,
    ):
        """
        Initialize SummarizationAgent.

        Args:
            model: Model name for summarization and answering (None = load from env)
            retain_recent: Number of recent turns to keep alongside summary
        """
        from dotenv import load_dotenv

        load_dotenv()

        self.retain_recent = retain_recent

        # Use MODEL_DEFAULT by default (same as other agents for fair comparison)
        self.model = model or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")

        # Initialize LLM client
        self._client = None
        self._init_client()

        # State
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []
        self._summary: str = ""

    def _init_client(self):
        """Initialize LLM client using API_KEY and API_BASE from .env."""
        try:
            from openai import OpenAI

            # Use unified API_KEY and API_BASE (supports one-api proxy)
            api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if not api_key:
                print("Warning: API_KEY not set, using mock responses")
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
        return f"Summarization(model={self.model}, retain={self.retain_recent})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self._history = []
        self._retained_history = []
        self._summary = ""

    def process_turn(self, turn: ConversationTurn) -> None:
        """
        Process a conversation turn.

        Simply stores the turn in history. Summarization happens at compression.
        """
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle compression event.

        Generate a summary of all history BEFORE compression, then keep
        summary + retained turns.
        """
        # Generate summary of history that will be "lost"
        # (everything before the retained turns)
        turns_to_summarize = [t for t in self._history if t not in retained_turns]

        if turns_to_summarize:
            self._summary = self._generate_summary(turns_to_summarize)
        else:
            self._summary = ""

        # Update retained history
        self._retained_history = retained_turns

        # Clear full history (simulating actual context truncation)
        self._history = list(retained_turns)

    def _generate_summary(self, turns: List[ConversationTurn]) -> str:
        """Generate a summary of conversation turns."""
        if not turns:
            return ""

        # Format turns for summarization
        conversation_text = self._format_history(turns)

        prompt = f"""Summarize the following conversation, capturing all important information including:
- Key decisions made
- Important facts, numbers, and constraints mentioned
- Action items and todos
- User preferences and requirements

Be thorough but concise. Preserve specific details like exact numbers, names, and technical choices.

## Conversation
{conversation_text}

## Summary
Provide a comprehensive summary:"""

        if self._client is None:
            return "[Mock summary: Important decisions and facts were discussed.]"

        try:
            return call_llm_with_retry(
                client=self._client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
        except Exception as e:
            return f"[Summary generation error: {e}]"

    def answer_question(self, question: str) -> AgentResponse:
        """
        Answer a recall question using summary + retained history.

        The summary should contain important facts from early turns,
        but may lose specific details.
        """
        start_time = time.time()

        # Build context: summary + retained history
        context_parts = []

        if self._summary:
            context_parts.append("## Summary of Earlier Conversation")
            context_parts.append(self._summary)
            context_parts.append("")

        # Use _history (includes post-compression turns) instead of _retained_history
        if self._history:
            context_parts.append("## Recent Conversation")
            context_parts.append(self._format_history(self._history))

        context = (
            "\n".join(context_parts) if context_parts else "[No context available]"
        )

        # Generate answer
        answer = self._generate_answer(context, question)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "has_summary": bool(self._summary),
                "summary_length": len(self._summary),
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
