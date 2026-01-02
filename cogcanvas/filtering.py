"""LLM-based filtering for canvas retrieval.

This module provides intelligent filtering of retrieved canvas objects using
an LLM to assess relevance, improving precision by filtering out "related
but not relevant" content before injection.
"""

import json
import time
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cogcanvas.models import CanvasObject, RetrievalResult

logger = logging.getLogger(__name__)


FILTER_PROMPT_TEMPLATE = """You are a relevance assessment expert. Given a user's question and a list of memory objects from a conversation, rate each object's relevance to DIRECTLY answering the question.

## User Question
{query}

## Candidate Memory Objects
{candidates_formatted}

## Instructions
For each object, assess:
1. **Direct Relevance**: Does this object directly answer or provide key information for the question?
2. **Temporal Correctness**: If the question asks about "current" or "final" state, is this the most recent/valid information?
   - Pay special attention to objects with time information (marked with [Event: ...] or [Time: ...])
   - For temporal queries (when, yesterday, last week, etc.), prioritize objects with matching time frames
3. **Specificity**: Does this object specifically address the topic, or is it merely tangentially related?

Rate each object from 0-10:
- 10: Directly answers the question (bonus for matching temporal context)
- 7-9: Highly relevant, provides important context
- 4-6: Somewhat related but may not directly help
- 1-3: Tangentially related, could be misleading
- 0: Not relevant or potentially contradictory

## Output Format (JSON)
{{
  "assessments": [
    {{"id": "obj_id_1", "score": 9, "reason": "Directly states the constraint"}},
    {{"id": "obj_id_2", "score": 3, "reason": "Old decision, superseded later"}}
  ]
}}

Output ONLY valid JSON, no additional text."""


@dataclass
class FilteredRetrievalResult:
    """Result of retrieval + LLM filtering."""

    objects: List["CanvasObject"] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    query: str = ""
    retrieval_time: float = 0.0
    filter_time: float = 0.0
    original_count: int = 0
    filtered_count: int = 0

    # Debug info
    original_objects: List["CanvasObject"] = field(default_factory=list)
    filter_reasoning: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.objects)

    def to_retrieval_result(self) -> "RetrievalResult":
        """Convert to standard RetrievalResult for inject()."""
        from cogcanvas.models import RetrievalResult

        return RetrievalResult(
            objects=self.objects,
            scores=self.scores,
            query=self.query,
            retrieval_time=self.retrieval_time + self.filter_time,
        )


class RetrievalFilter:
    """
    LLM-based filtering for retrieved canvas objects.

    Uses a single LLM call to assess all candidates, selecting
    the most relevant ones for final injection.

    Example:
        >>> filter = RetrievalFilter()
        >>> candidates = canvas.retrieve(query, top_k=20)
        >>> filtered = filter.filter(query, candidates, top_k=5)
        >>> context = canvas.inject(filtered.to_retrieval_result())
    """

    def __init__(
        self,
        model: str = None,
        prompt_template: str = None,
        min_relevance_score: float = 0.3,
    ):
        """
        Initialize the retrieval filter.

        Args:
            model: Model to use for filtering (default: MODEL_DEFAULT or gpt-4o-mini)
            prompt_template: Custom filter prompt template
            min_relevance_score: Minimum score (0-1) to include an object
        """
        self._client = None
        self._model = model or os.getenv("EXTRACTOR_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
        self.prompt_template = prompt_template or FILTER_PROMPT_TEMPLATE
        self.min_relevance_score = min_relevance_score

        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = os.getenv("EXTRACTOR_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("EXTRACTOR_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if api_key:
                self._client = OpenAI(api_key=api_key, base_url=api_base)
                logger.debug(f"RetrievalFilter initialized with model: {self._model}")
            else:
                logger.warning("No API key found, LLM filtering disabled")
        except ImportError:
            logger.warning("OpenAI package not available, LLM filtering disabled")

    def filter(
        self,
        query: str,
        candidates: "RetrievalResult",
        top_k: int = 5,
    ) -> FilteredRetrievalResult:
        """
        Filter retrieval candidates using LLM.

        Args:
            query: The user's question
            candidates: RetrievalResult from canvas.retrieve()
            top_k: Number of objects to keep after filtering

        Returns:
            FilteredRetrievalResult with top_k most relevant objects
        """
        start_time = time.time()

        # If no candidates or no client, return as-is
        if not candidates.objects:
            return FilteredRetrievalResult(
                objects=[],
                scores=[],
                query=query,
                retrieval_time=candidates.retrieval_time,
                filter_time=0.0,
                original_count=0,
                filtered_count=0,
            )

        if not self._client:
            # Fallback: return top_k without filtering
            logger.warning("LLM client not available, returning unfiltered results")
            return FilteredRetrievalResult(
                objects=candidates.objects[:top_k],
                scores=(
                    candidates.scores[:top_k]
                    if candidates.scores
                    else [1.0] * min(top_k, len(candidates.objects))
                ),
                query=query,
                retrieval_time=candidates.retrieval_time,
                filter_time=0.0,
                original_count=len(candidates.objects),
                filtered_count=min(top_k, len(candidates.objects)),
            )

        # Format candidates for prompt
        candidates_formatted = self._format_candidates(candidates.objects)

        # Build prompt
        prompt = self.prompt_template.format(
            query=query,
            candidates_formatted=candidates_formatted,
        )

        # Call LLM for filtering
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1500,
            )
            raw_response = response.choices[0].message.content.strip()
            assessments = self._parse_filter_response(raw_response, candidates.objects)

        except Exception as e:
            logger.warning(f"LLM filtering failed: {e}, returning unfiltered")
            return FilteredRetrievalResult(
                objects=candidates.objects[:top_k],
                scores=(
                    candidates.scores[:top_k]
                    if candidates.scores
                    else [1.0] * min(top_k, len(candidates.objects))
                ),
                query=query,
                retrieval_time=candidates.retrieval_time,
                filter_time=time.time() - start_time,
                original_count=len(candidates.objects),
                filtered_count=min(top_k, len(candidates.objects)),
            )

        # Build id -> object map
        obj_map = {obj.id: obj for obj in candidates.objects}

        # Sort by LLM score and select top_k
        sorted_assessments = sorted(
            assessments, key=lambda x: x.get("score", 0), reverse=True
        )

        selected_objects = []
        selected_scores = []
        filter_reasoning = []

        for assessment in sorted_assessments:
            if len(selected_objects) >= top_k:
                break

            obj_id = assessment.get("id")
            score = assessment.get("score", 0) / 10.0  # Normalize to 0-1
            reason = assessment.get("reason", "")

            if obj_id in obj_map and score >= self.min_relevance_score:
                selected_objects.append(obj_map[obj_id])
                selected_scores.append(score)
                filter_reasoning.append(reason)

        filter_time = time.time() - start_time

        logger.info(
            f"LLM filter: {len(candidates.objects)} -> {len(selected_objects)} "
            f"({filter_time:.2f}s)"
        )

        return FilteredRetrievalResult(
            objects=selected_objects,
            scores=selected_scores,
            query=query,
            retrieval_time=candidates.retrieval_time,
            filter_time=filter_time,
            original_count=len(candidates.objects),
            filtered_count=len(selected_objects),
            original_objects=candidates.objects,
            filter_reasoning=filter_reasoning,
        )

    def _format_candidates(self, objects: List["CanvasObject"]) -> str:
        """Format candidate objects for the filter prompt."""
        lines = []
        for obj in objects:
            type_label = obj.type.value.upper()
            content = obj.content[:200]  # Truncate for token efficiency
            turn_info = (
                f"(Turn {obj.turn_id})"
                if hasattr(obj, "turn_id") and obj.turn_id
                else ""
            )

            # Include quote if available for better context
            quote_info = ""
            if hasattr(obj, "quote") and obj.quote:
                quote_info = (
                    f' | Quote: "{obj.quote[:100]}..."'
                    if len(obj.quote) > 100
                    else f' | Quote: "{obj.quote}"'
                )

            lines.append(f"[{obj.id}] [{type_label}] {turn_info} {content}{quote_info}")
        return "\n".join(lines)

    def _parse_filter_response(
        self, raw_response: str, original_objects: List["CanvasObject"]
    ) -> List[Dict[str, Any]]:
        """Parse LLM filter response."""
        try:
            # Handle markdown code blocks
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0]
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0]

            data = json.loads(raw_response.strip())

            if isinstance(data, dict) and "assessments" in data:
                return data["assessments"]
            elif isinstance(data, list):
                return data
            else:
                logger.warning(
                    "Unexpected response format, falling back to original order"
                )
                return [{"id": obj.id, "score": 5} for obj in original_objects]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse filter response: {e}")
            # Fallback: return original objects with neutral scores
            return [{"id": obj.id, "score": 5} for obj in original_objects]
