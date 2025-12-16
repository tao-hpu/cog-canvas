"""Fuzzy Linking Resolver - Maps natural language references to Canvas object IDs.

This is CRITICAL for graph construction. The extractor model doesn't know object IDs,
so it outputs natural language references like "that postgres decision". This resolver
maps those references back to actual CanvasObject IDs using semantic similarity.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import re

if TYPE_CHECKING:
    from .models import CanvasObject, ObjectType


@dataclass
class ResolveResult:
    """Result of resolving a natural language reference."""
    reference_text: str          # Original reference, e.g. "that postgres decision"
    resolved_id: Optional[str]   # Matched object ID, or None if no match
    confidence: float            # Match confidence [0, 1]
    matched_content: str = ""    # Content of the matched object (for debugging)


class ReferenceResolver(ABC):
    """Abstract base class for resolving natural language references to object IDs."""

    @abstractmethod
    def resolve(
        self,
        reference: str,
        candidates: List["CanvasObject"],
        type_hint: Optional["ObjectType"] = None,
    ) -> ResolveResult:
        """
        Resolve a natural language reference to a CanvasObject ID.

        Args:
            reference: Natural language reference, e.g. "that postgres decision"
            candidates: List of existing CanvasObjects to match against
            type_hint: Optional type hint extracted from reference (e.g. "decision")

        Returns:
            ResolveResult with the best matching object ID and confidence
        """
        pass

    def resolve_batch(
        self,
        references: List[str],
        candidates: List["CanvasObject"],
    ) -> List[ResolveResult]:
        """Resolve multiple references."""
        return [self.resolve(ref, candidates) for ref in references]


class KeywordResolver(ReferenceResolver):
    """
    Simple keyword-based resolver. Fast but limited recall.
    Use as fallback when embedding-based resolver is unavailable.
    """

    # Type indicators in natural language
    TYPE_INDICATORS = {
        "decision": ["decision", "decided", "chose", "choice", "picked"],
        "todo": ["todo", "task", "action", "need to", "should"],
        "key_fact": ["fact", "info", "information", "detail", "number"],
        "reminder": ["reminder", "remember", "constraint", "preference"],
        "insight": ["insight", "realization", "found", "noticed"],
    }

    def resolve(
        self,
        reference: str,
        candidates: List["CanvasObject"],
        type_hint: Optional["ObjectType"] = None,
    ) -> ResolveResult:
        if not candidates:
            return ResolveResult(reference, None, 0.0)

        reference_lower = reference.lower()

        # Extract type hint from reference if not provided
        inferred_type = type_hint
        if not inferred_type:
            inferred_type = self._infer_type_from_reference(reference_lower)

        # Score each candidate
        best_match: Optional["CanvasObject"] = None
        best_score = 0.0

        for candidate in candidates:
            score = self._score_match(reference_lower, candidate, inferred_type)
            if score > best_score:
                best_score = score
                best_match = candidate

        # Require minimum confidence threshold
        if best_score < 0.3:
            return ResolveResult(reference, None, best_score)

        return ResolveResult(
            reference_text=reference,
            resolved_id=best_match.id if best_match else None,
            confidence=best_score,
            matched_content=best_match.content if best_match else "",
        )

    def _infer_type_from_reference(self, reference: str) -> Optional["ObjectType"]:
        """Infer object type from natural language reference."""
        from .models import ObjectType

        for type_name, indicators in self.TYPE_INDICATORS.items():
            if any(ind in reference for ind in indicators):
                return ObjectType(type_name)
        return None

    def _score_match(
        self,
        reference: str,
        candidate: "CanvasObject",
        type_hint: Optional["ObjectType"],
    ) -> float:
        """Score how well a candidate matches the reference."""
        score = 0.0

        # Type match bonus
        if type_hint and candidate.type == type_hint:
            score += 0.3

        # Keyword overlap in content
        ref_words = set(self._extract_keywords(reference))
        content_words = set(self._extract_keywords(candidate.content.lower()))
        quote_words = set(self._extract_keywords(candidate.quote.lower())) if candidate.quote else set()

        # Check overlap
        all_candidate_words = content_words | quote_words
        if ref_words and all_candidate_words:
            overlap = len(ref_words & all_candidate_words) / len(ref_words)
            score += 0.5 * overlap

        # Recency bonus (more recent = more likely to be referenced)
        # This is a heuristic: references often point to recent objects
        # Score is already 0-0.8, add small recency factor
        score += 0.1 * (1.0 / (1.0 + candidate.turn_id * 0.1))

        return min(score, 1.0)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common words
        stopwords = {
            "the", "a", "an", "that", "this", "those", "these",
            "about", "with", "from", "for", "to", "in", "on",
            "we", "i", "you", "it", "is", "are", "was", "were",
            "do", "does", "did", "have", "has", "had",
        }
        words = re.findall(r'\b[a-z]+\b', text)
        return [w for w in words if w not in stopwords and len(w) > 2]


class EmbeddingResolver(ReferenceResolver):
    """
    Embedding-based resolver using semantic similarity.
    Higher recall than keyword matching, handles synonyms (DB vs Database).
    """

    def __init__(self, embedding_backend=None, similarity_threshold: float = 0.5):
        """
        Args:
            embedding_backend: EmbeddingBackend instance for computing embeddings
            similarity_threshold: Minimum cosine similarity to consider a match
        """
        self.embedding_backend = embedding_backend
        self.similarity_threshold = similarity_threshold
        self._keyword_fallback = KeywordResolver()

    def resolve(
        self,
        reference: str,
        candidates: List["CanvasObject"],
        type_hint: Optional["ObjectType"] = None,
    ) -> ResolveResult:
        if not candidates:
            return ResolveResult(reference, None, 0.0)

        # Fallback to keyword if no embedding backend
        if self.embedding_backend is None:
            return self._keyword_fallback.resolve(reference, candidates, type_hint)

        # Filter by type hint first (reduces search space)
        if type_hint:
            typed_candidates = [c for c in candidates if c.type == type_hint]
            if typed_candidates:
                candidates = typed_candidates

        # Compute reference embedding
        ref_embedding = self.embedding_backend.embed(reference)

        # Score each candidate by cosine similarity
        best_match: Optional["CanvasObject"] = None
        best_score = 0.0

        for candidate in candidates:
            # Use pre-computed embedding if available
            if candidate.embedding:
                cand_embedding = candidate.embedding
            else:
                # Embed content + quote for better matching
                cand_text = f"{candidate.content} {candidate.quote}".strip()
                cand_embedding = self.embedding_backend.embed(cand_text)

            similarity = self._cosine_similarity(ref_embedding, cand_embedding)
            if similarity > best_score:
                best_score = similarity
                best_match = candidate

        # Check threshold
        if best_score < self.similarity_threshold:
            return ResolveResult(reference, None, best_score)

        return ResolveResult(
            reference_text=reference,
            resolved_id=best_match.id if best_match else None,
            confidence=best_score,
            matched_content=best_match.content if best_match else "",
        )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class HybridResolver(ReferenceResolver):
    """
    Combines keyword and embedding-based resolution.
    Uses keywords for type inference, embeddings for semantic matching.
    """

    def __init__(self, embedding_backend=None, weights: Tuple[float, float] = (0.3, 0.7)):
        """
        Args:
            embedding_backend: EmbeddingBackend for semantic matching
            weights: (keyword_weight, embedding_weight) for combining scores
        """
        self.keyword_resolver = KeywordResolver()
        self.embedding_resolver = EmbeddingResolver(embedding_backend)
        self.keyword_weight, self.embedding_weight = weights

    def resolve(
        self,
        reference: str,
        candidates: List["CanvasObject"],
        type_hint: Optional["ObjectType"] = None,
    ) -> ResolveResult:
        if not candidates:
            return ResolveResult(reference, None, 0.0)

        # Get both results
        kw_result = self.keyword_resolver.resolve(reference, candidates, type_hint)
        emb_result = self.embedding_resolver.resolve(reference, candidates, type_hint)

        # If both resolved to the same object, boost confidence
        if kw_result.resolved_id == emb_result.resolved_id and kw_result.resolved_id:
            combined_confidence = min(1.0, kw_result.confidence + emb_result.confidence)
            return ResolveResult(
                reference_text=reference,
                resolved_id=kw_result.resolved_id,
                confidence=combined_confidence,
                matched_content=kw_result.matched_content,
            )

        # Otherwise, use weighted combination - prefer embedding result
        if emb_result.confidence >= kw_result.confidence:
            return emb_result
        return kw_result


def get_resolver(embedding_backend=None, strategy: str = "hybrid") -> ReferenceResolver:
    """
    Factory function to get a reference resolver.

    Args:
        embedding_backend: Optional embedding backend for semantic matching
        strategy: "keyword", "embedding", or "hybrid"

    Returns:
        ReferenceResolver instance
    """
    if strategy == "keyword":
        return KeywordResolver()
    elif strategy == "embedding":
        return EmbeddingResolver(embedding_backend)
    else:
        return HybridResolver(embedding_backend)
