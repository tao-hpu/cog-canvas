"""Anthropic LLM backend for CogCanvas extraction."""

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

Use the extract_canvas_objects tool to return the extracted objects."""


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend for extraction and embeddings."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Anthropic backend.

        Args:
            model: Model for extraction (default: claude-3-5-haiku-latest for cost efficiency)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )

        # Initialize client
        try:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

    def extract_objects(
        self,
        user_message: str,
        assistant_message: str,
        existing_objects: Optional[List[CanvasObject]] = None,
    ) -> List[CanvasObject]:
        """
        Extract canvas objects using Claude with tool use.

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
            context_hint = (
                f"\n\nAlready extracted (avoid duplicates):\n{existing_summary}"
            )

        # Define tool for structured extraction
        tools = [
            {
                "name": "extract_canvas_objects",
                "description": "Extract structured cognitive objects from the dialogue",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "decision",
                                            "todo",
                                            "key_fact",
                                            "reminder",
                                            "insight",
                                        ],
                                        "description": "The type of cognitive object",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The extracted information in a clear, standalone statement",
                                    },
                                    "citation": {
                                        "type": "string",
                                        "description": "EXACT verbatim quote from the dialogue that supports this extraction",
                                    },
                                    "context": {
                                        "type": "string",
                                        "description": "Brief explanation of source/reason for extraction",
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "description": "Confidence score between 0.0 and 1.0",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                },
                                "required": ["type", "content", "citation", "context", "confidence"],
                            },
                        }
                    },
                    "required": ["objects"],
                },
            }
        ]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent extraction
                tools=tools,
                messages=[
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT + context_hint + "\n\n" + dialogue,
                    }
                ],
            )

            # Parse tool use response
            return self._parse_tool_response(response)

        except Exception as e:
            # Log error but don't crash - return empty list
            print(f"Extraction error: {e}")
            return []

    def _parse_tool_response(self, response) -> List[CanvasObject]:
        """Parse Claude tool use response into CanvasObjects."""
        objects = []

        try:
            # Find tool use in response
            for content in response.content:
                if content.type == "tool_use" and content.name == "extract_canvas_objects":
                    tool_input = content.input
                    extracted_objects = tool_input.get("objects", [])

                    for item in extracted_objects:
                        try:
                            obj_type = ObjectType(item.get("type", "key_fact"))
                            objects.append(
                                CanvasObject(
                                    type=obj_type,
                                    content=item.get("content", ""),
                                    citation=item.get("citation", ""),  # Provenance for verification
                                    context=item.get("context", ""),
                                    confidence=float(item.get("confidence", 0.8)),
                                )
                            )
                        except (ValueError, KeyError) as e:
                            print(f"Skipping invalid object: {e}")
                            continue

        except Exception as e:
            print(f"Tool response parse error: {e}")

        return objects

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Note: Anthropic doesn't provide an embeddings API, so this falls back
        to a simple hash-based mock embedding. For production use, consider
        using OpenAI embeddings or a local model like sentence-transformers.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (mock implementation)
        """
        import hashlib

        # Use deterministic hash-based embedding as fallback
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(384):  # Match sentence-transformers dimension
            byte_idx = i % 32
            embedding.append(hash_bytes[byte_idx] / 255.0)

        print(
            "Warning: Anthropic backend using mock embeddings. "
            "Consider using OpenAI or sentence-transformers for embeddings."
        )
        return embedding
