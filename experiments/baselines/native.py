"""
Native Baseline: No intervention, just truncation.

This is the worst-case baseline that simulates what happens when
context is simply truncated without any memory augmentation.

Expected performance: <20% retention on early facts
"""

from typing import List, Dict, Any
from .base import BaseMethod


class NativeBaseline(BaseMethod):
    """
    Native context baseline - no intervention.

    After compression, only the most recent N turns are available.
    Early facts are completely lost.
    """

    name = "native"

    def __init__(self, llm_client=None, config: Dict = None):
        super().__init__(llm_client, config)
        self.retain_recent = config.get("retain_recent", 5) if config else 5

    def process_conversation(
        self,
        turns: List[Dict[str, str]],
        compression_turn: int,
    ) -> Dict[str, Any]:
        """
        Native baseline: Just store turns, no processing.

        The "compression" happens in get_context_for_question() where
        we only return recent turns.
        """
        return {
            "all_turns": turns,
            "compression_turn": compression_turn,
        }

    def get_context_for_question(
        self,
        state: Dict[str, Any],
        question: str,
        recent_turns: List[Dict[str, str]],
    ) -> str:
        """
        Return only recent turns - simulating complete loss of early context.

        This is the key limitation we're trying to solve:
        Early facts are simply gone after truncation.
        """
        # Only use recent turns (post-compression context)
        context_turns = recent_turns[-self.retain_recent:]

        return self._format_turns(context_turns)
