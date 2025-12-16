"""Base LLM interface for CogCanvas."""

from abc import ABC, abstractmethod
from typing import List, Optional
from cogcanvas.models import CanvasObject, ObjectType


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def extract_objects(
        self,
        user_message: str,
        assistant_message: str,
        existing_objects: Optional[List[CanvasObject]] = None,
    ) -> List[CanvasObject]:
        """
        Extract canvas objects from a dialogue turn.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            existing_objects: Existing objects for context (optional)

        Returns:
            List of extracted CanvasObjects
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass


class MockLLMBackend(LLMBackend):
    """Mock backend for testing without API calls."""

    def __init__(self):
        self._turn_counter = 0

    def extract_objects(
        self,
        user_message: str,
        assistant_message: str,
        existing_objects: Optional[List[CanvasObject]] = None,
    ) -> List[CanvasObject]:
        """
        Rule-based mock extraction for testing.

        Enhanced to match data_gen.py templates:
        - DECISION: "recommend", "suggest", "go with", "use"
        - KEY_FACT: "rate limit", "deadline", "budget", numbers
        - REMINDER: "prefer", "should", "style", "strategy"
        """
        self._turn_counter += 1
        objects = []
        combined = f"{user_message} {assistant_message}".lower()
        assistant_lower = assistant_message.lower()

        # DECISION detection (matches data_gen DECISION_TEMPLATES)
        decision_triggers = ["recommend", "suggest", "go with", "let's use", "i recommend", "we should use"]
        if any(trigger in assistant_lower for trigger in decision_triggers):
            objects.append(
                CanvasObject(
                    type=ObjectType.DECISION,
                    content=assistant_message[:200],  # Store actual content
                    context=f"Decision from turn {self._turn_counter}",
                    quote=assistant_message,  # Full quote for grounding
                    turn_id=self._turn_counter,
                    source="assistant",
                )
            )

        # KEY_FACT detection (matches data_gen KEY_FACT_TEMPLATES)
        fact_triggers = ["rate limit", "deadline", "budget", "team of", "approximately", "have a"]
        if any(trigger in assistant_lower for trigger in fact_triggers):
            objects.append(
                CanvasObject(
                    type=ObjectType.KEY_FACT,
                    content=assistant_message[:200],
                    context=f"Fact from turn {self._turn_counter}",
                    quote=assistant_message,
                    turn_id=self._turn_counter,
                    source="assistant",
                )
            )

        # REMINDER detection (matches data_gen REMINDER_TEMPLATES)
        reminder_triggers = ["please", "keep", "maintain", "follow", "ensure", "preference"]
        if any(trigger in assistant_lower for trigger in reminder_triggers):
            objects.append(
                CanvasObject(
                    type=ObjectType.REMINDER,
                    content=assistant_message[:200],
                    context=f"Reminder from turn {self._turn_counter}",
                    quote=assistant_message,
                    turn_id=self._turn_counter,
                    source="assistant",
                )
            )

        return objects

    def embed(self, text: str) -> List[float]:
        """Generate mock embedding using hash."""
        import hashlib

        hash_bytes = hashlib.md5(text.encode()).digest()
        # Convert to 384-dim vector (like all-MiniLM-L6-v2)
        embedding = []
        for i in range(384):
            byte_idx = i % 16
            embedding.append(hash_bytes[byte_idx] / 255.0)
        return embedding
