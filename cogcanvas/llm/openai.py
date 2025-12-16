"""OpenAI LLM backend for CogCanvas extraction."""

import os
import json
from typing import List, Optional

from cogcanvas.llm.base import LLMBackend
from cogcanvas.models import CanvasObject, ObjectType

# Extraction prompt for typed cognitive objects with provenance
EXTRACTION_PROMPT = """You are an expert at extracting structured cognitive objects from dialogue.

Given a conversation turn (user message + assistant response), extract any of these object types:

1. **decision**: A choice or decision made (e.g., "Let's use PostgreSQL", "We'll go with approach B")
2. **todo**: Action items, tasks to do (e.g., "Need to implement auth", "Should add tests")
3. **key_fact**: Important facts, numbers, names, constraints (e.g., "Budget is $50k", "API rate limit is 100/min")
4. **reminder**: User preferences, rules, constraints to remember (e.g., "User prefers TypeScript", "No external dependencies")
5. **insight**: Conclusions, learnings, realizations (e.g., "The bottleneck is in the database", "This approach won't scale")

Rules:
- Only extract objects that are explicitly stated or clearly implied
- Each object should be self-contained and understandable without the original context
- **CRITICAL**: You MUST include a "citation" field with the EXACT quote from the original dialogue that supports this extraction
- The citation must be a verbatim substring from either the user or assistant message
- Be conservative: only extract genuinely important information
- Skip trivial or obvious statements

Output JSON array (can be empty if nothing worth extracting):
[
  {
    "type": "decision|todo|key_fact|reminder|insight",
    "content": "The extracted information in a clear, standalone statement",
    "citation": "EXACT verbatim quote from the dialogue that supports this extraction",
    "context": "Brief explanation of why this was extracted",
    "confidence": 0.0-1.0
  }
]

Example:
User: "Let's use PostgreSQL, it's better for our budget of $50k"
Assistant: "Good choice!"

Output:
[
  {"type": "decision", "content": "Use PostgreSQL as the database", "citation": "Let's use PostgreSQL", "context": "User made database decision", "confidence": 0.95},
  {"type": "key_fact", "content": "Budget is $50,000", "citation": "budget of $50k", "context": "Budget constraint mentioned", "confidence": 0.9}
]"""


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible LLM backend for extraction and embeddings."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize OpenAI backend.

        Args:
            model: Model for extraction (default: gpt-4o-mini for cost efficiency)
            embedding_model: Model for embeddings
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            api_base: Custom API base URL (for compatible APIs)
        """
        self.model = model
        self.embedding_model = embedding_model
        # Support both OPENAI_API_KEY and API_KEY (for compatibility with different .env setups)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("API_BASE", "https://api.openai.com/v1")

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

        # Initialize client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def extract_objects(
        self,
        user_message: str,
        assistant_message: str,
        existing_objects: Optional[List[CanvasObject]] = None,
    ) -> List[CanvasObject]:
        """
        Extract canvas objects using LLM.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            existing_objects: Existing objects for deduplication context

        Returns:
            List of extracted CanvasObjects
        """
        # Build the extraction prompt
        dialogue = f"User: {user_message}\n\nAssistant: {assistant_message}"

        # Add existing objects context for deduplication
        context_hint = ""
        if existing_objects:
            existing_summary = "\n".join(
                f"- [{obj.type.value}] {obj.content[:100]}"
                for obj in existing_objects[-10:]  # Last 10 objects
            )
            context_hint = f"\n\nAlready extracted (avoid duplicates):\n{existing_summary}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT + context_hint},
                    {"role": "user", "content": dialogue},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000,
            )

            raw_response = response.choices[0].message.content.strip()
            return self._parse_extraction_response(raw_response)

        except Exception as e:
            # Log error but don't crash - return empty list
            print(f"Extraction error: {e}")
            return []

    def _parse_extraction_response(self, raw_response: str) -> List[CanvasObject]:
        """Parse LLM response into CanvasObjects."""
        objects = []

        try:
            # Handle markdown code blocks
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0]
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0]

            data = json.loads(raw_response.strip())

            # Ensure it's a list
            if isinstance(data, dict):
                data = [data]

            for item in data:
                try:
                    obj_type = ObjectType(item.get("type", "key_fact"))
                    objects.append(
                        CanvasObject(
                            type=obj_type,
                            content=item.get("content", ""),
                            quote=item.get("citation", ""),  # Provenance for verification (maps to CanvasObject.quote)
                            context=item.get("context", ""),
                            confidence=float(item.get("confidence", 0.8)),
                        )
                    )
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid object: {e}")
                    continue

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Try to salvage partial response
            pass

        return objects

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI embeddings API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Default OpenAI embedding dimension
