"""
Full-Context Baseline Agent: entire conversation in the answerer prompt.

Upper-bound reference baseline: no retrieval, no compression, no memory
mechanism. The agent keeps every turn it has seen (compression events are
ignored) and puts the full transcript -- with session timestamps and turn
markers -- directly into the answer prompt.

LoCoMo conversations are ~26K tokens, well within gpt-4o's 128K window.
A protective truncation (drop oldest turns) only kicks in if the estimated
prompt would exceed MAX_CONTEXT_TOKENS; every truncation is recorded in the
response metadata.
"""

from typing import List, Optional
import time
import os

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from experiments.llm_utils import call_llm_with_retry


class FullContextAgent(Agent):
    """
    Full-context baseline agent - no retrieval, no compression.

    On compression:
    - No-op: the full history is kept (this baseline assumes an unlimited
      context window, unlike Native which truncates).

    On answer:
    - Entire conversation (session timestamps + turn markers) is placed in
      the prompt and the answer model reads it directly.
    """

    # Protective cap on estimated prompt tokens (gpt-4o context is 128K;
    # leave headroom for prompt template + completion).
    MAX_CONTEXT_TOKENS = 110_000

    def __init__(
        self,
        model: str = None,
    ):
        """
        Initialize FullContextAgent.

        Args:
            model: Model name for answer generation (None = load from env ANSWER_MODEL)
        """
        from dotenv import load_dotenv

        load_dotenv()

        # Use ANSWER_MODEL by default (same as other agents for fair comparison)
        self.model = model or os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")

        # Initialize LLM client
        self._client = None
        self._init_client()

        # State: full conversation, never truncated by compression
        self._all_turns: List[ConversationTurn] = []
        self._truncation_events: int = 0

    def _init_client(self):
        """Initialize LLM client using ANSWER_API_* (fallback API_*) from .env."""
        try:
            from openai import OpenAI

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
        return f"FullContext(model={self.model})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self._all_turns = []
        self._truncation_events = 0

    def process_turn(self, turn: ConversationTurn) -> None:
        """Store the turn. No extraction or augmentation."""
        self._all_turns.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Compression is a no-op for the full-context baseline.

        This agent models an unlimited context window: nothing is lost.
        """
        pass

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars/token for English text)."""
        return len(text) // 4

    def _format_full_history(self, turns: List[ConversationTurn]) -> str:
        """
        Format the full conversation with session timestamps and turn markers.

        Format aligned with how other agents present turns (User/Assistant),
        plus session datetime headers (printed whenever the session changes)
        and per-turn markers.
        """
        if not turns:
            return "[No conversation history available]"

        lines = []
        current_session = None
        for turn in turns:
            session_dt = getattr(turn, "session_datetime", None)
            if session_dt and session_dt != current_session:
                current_session = session_dt
                lines.append(f"=== Session: {session_dt} ===")
            lines.append(f"[Turn {turn.turn_id}]")
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")
            lines.append("")

        return "\n".join(lines)

    def answer_question(self, question: str) -> AgentResponse:
        """
        Answer a recall question with the FULL conversation in the prompt.

        Applies a protective truncation (drop oldest turns) only if the
        estimated prompt would exceed MAX_CONTEXT_TOKENS.
        """
        start_time = time.time()

        turns = self._all_turns
        truncated = False
        context = self._format_full_history(turns)

        # Protective truncation: drop oldest turns until under the cap
        while turns and self._estimate_tokens(context) > self.MAX_CONTEXT_TOKENS:
            truncated = True
            turns = turns[10:]  # Drop oldest 10 turns per step
            context = self._format_full_history(turns)

        if truncated:
            self._truncation_events += 1
            print(
                f"[FullContext] WARNING: protective truncation applied "
                f"({len(self._all_turns)} -> {len(turns)} turns, "
                f"~{self._estimate_tokens(context)} tokens)"
            )

        answer = self._generate_answer(context, question)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "num_turns": len(self._all_turns),
                "num_turns_in_context": len(turns),
                "truncated": truncated,
                "context_length": len(context),
                "estimated_context_tokens": self._estimate_tokens(context),
            },
        )

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer using LLM (same prompt frame as other baselines)."""
        prompt = f"""You are an expert reasoning agent. Your goal is to answer the user's question by connecting discrete facts from the conversation below.

## Full Conversation History
{context}

## Instructions
1. Read the conversation carefully (session timestamps mark when each part happened)
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
                call_type="gen",
            )
        except Exception as e:
            return f"Error generating answer: {e}"
