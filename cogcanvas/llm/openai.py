"""OpenAI LLM backend for CogCanvas extraction."""

import os
import json
from typing import List, Optional

from cogcanvas.llm.base import LLMBackend
from cogcanvas.models import CanvasObject, ObjectType

# Extraction prompt for typed cognitive objects with provenance
EXTRACTION_PROMPT = """You are an expert at extracting structured cognitive objects from dialogue.

Given a conversation turn (user message + assistant response), extract any of these object types:

**Task-oriented types:**
1. **decision**: A choice or decision made (e.g., "Let's use PostgreSQL", "I decided to pursue counseling")
2. **todo**: Action items, tasks to do (e.g., "Need to implement auth", "Planning to go camping next month")
3. **key_fact**: Important facts, numbers, constraints (e.g., "Budget is $50k", "The event is in June 2023")
4. **reminder**: Preferences, rules to remember (e.g., "User prefers TypeScript", "Always meditate in morning")
5. **insight**: Conclusions, learnings (e.g., "The bottleneck is in the database", "Exercise helps my anxiety")

**Personal/Social types (IMPORTANT for conversations about people):**
6. **person_attribute**: Personal traits, identity, status (e.g., "Caroline is a transgender woman", "Melanie is married with two kids", "John is single", "She moved from Sweden 4 years ago")
7. **event**: Activities or occurrences WITH time (e.g., "Attended LGBTQ support group on 7 May 2023", "Ran a charity race last Sunday", "Painting a sunrise in 2022")
8. **relationship**: Interpersonal connections (e.g., "Caroline and Melanie are close friends", "Known each other for 4 years")

**EXTRACTION RULES:**
- Extract ALL factual information about people (identity, status, activities, preferences)
- Extract ALL events with their time expressions
- Extract ALL relationships mentioned
- Each object should be self-contained and understandable without context
- **CRITICAL**: Include "citation" field with EXACT quote from dialogue
- Do NOT skip personal information - it is important!

**CRITICAL TEMPORAL RULES:**
- Preserve dates EXACTLY as written: "7 May 2023", "June 2023", "the sunday before 25 May 2023"
- DO NOT convert specific dates to relative expressions
- Include dates in both "content" and "time_expression" fields

Output JSON array:
[
  {
    "type": "decision|todo|key_fact|reminder|insight|person_attribute|event|relationship",
    "content": "The extracted information INCLUDING exact dates if mentioned",
    "citation": "EXACT verbatim quote from dialogue",
    "context": "Brief explanation of why extracted",
    "confidence": 0.0-1.0,
    "time_expression": "VERBATIM time expression or empty string"
  }
]

**Example 1 (Social conversation):**
User: "Caroline is a transgender woman who moved from Sweden 4 years ago. She went to the LGBTQ support group on 7 May 2023."
Assistant: "That's wonderful that she found a supportive community!"

Output:
[
  {"type": "person_attribute", "content": "Caroline is a transgender woman", "citation": "Caroline is a transgender woman", "context": "Identity information", "confidence": 0.95, "time_expression": ""},
  {"type": "person_attribute", "content": "Caroline moved from Sweden 4 years ago", "citation": "moved from Sweden 4 years ago", "context": "Background information", "confidence": 0.9, "time_expression": "4 years ago"},
  {"type": "event", "content": "Caroline attended LGBTQ support group on 7 May 2023", "citation": "went to the LGBTQ support group on 7 May 2023", "context": "Activity with specific date", "confidence": 0.95, "time_expression": "7 May 2023"}
]

**Example 2 (Mixed conversation):**
User: "Melanie is married with two kids. She's planning to go camping in June 2023. We've been friends for 4 years."
Assistant: "Sounds like a fun family trip!"

Output:
[
  {"type": "person_attribute", "content": "Melanie is married with two kids", "citation": "Melanie is married with two kids", "context": "Family status", "confidence": 0.95, "time_expression": ""},
  {"type": "event", "content": "Melanie planning camping trip in June 2023", "citation": "planning to go camping in June 2023", "context": "Future activity with date", "confidence": 0.9, "time_expression": "June 2023"},
  {"type": "relationship", "content": "User and Melanie have been friends for 4 years", "citation": "We've been friends for 4 years", "context": "Friendship duration", "confidence": 0.9, "time_expression": "4 years"}
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
        turn_id: int = 0,
        enable_temporal_fallback: bool = True,
        session_datetime: Optional[str] = None,
    ) -> List[CanvasObject]:
        """
        Extract canvas objects using LLM with optional temporal enhancement.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            existing_objects: Existing objects for deduplication context
            turn_id: Current conversation turn (for temporal artifacts)
            enable_temporal_fallback: Use regex to catch dates LLM might miss
            session_datetime: Session timestamp for relative time resolution
                             (e.g., "1:56 pm on 8 May, 2023")

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

        llm_objects = []
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
            llm_objects = self._parse_extraction_response(raw_response)

            # Set turn_id and session_datetime for all objects
            for obj in llm_objects:
                obj.turn_id = turn_id
                obj.session_datetime = session_datetime

        except Exception as e:
            # Log error but don't crash - continue with temporal fallback
            print(f"Extraction error: {e}")

        # Temporal fallback: use regex to catch dates LLM might have missed
        # Also resolves relative times to absolute when session_datetime is provided
        if enable_temporal_fallback:
            try:
                from cogcanvas.temporal import extract_and_merge_temporal
                full_text = f"{user_message}\n{assistant_message}"
                llm_objects = extract_and_merge_temporal(
                    full_text, llm_objects, turn_id, source="user",
                    session_datetime=session_datetime
                )
            except ImportError:
                pass  # Temporal module not available, skip fallback

        return llm_objects

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
                    # Extract time expression if present
                    time_expr = item.get("time_expression", "")
                    event_time_raw = time_expr if time_expr else None

                    objects.append(
                        CanvasObject(
                            type=obj_type,
                            content=item.get("content", ""),
                            quote=item.get("citation", ""),  # Provenance for verification (maps to CanvasObject.quote)
                            context=item.get("context", ""),
                            confidence=float(item.get("confidence", 0.8)),
                            event_time_raw=event_time_raw,  # Store raw time expression
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
