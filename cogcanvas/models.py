"""Core data models for CogCanvas."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import time
import uuid


class ObjectType(Enum):
    """Types of canvas objects that can be extracted from dialogue."""

    # Original types (task-oriented)
    DECISION = "decision"      # Choices/decisions made
    TODO = "todo"              # Action items, tasks to do
    KEY_FACT = "key_fact"      # Important facts, numbers, names
    REMINDER = "reminder"      # Constraints, preferences, rules
    INSIGHT = "insight"        # Conclusions, learnings, realizations

    # Extended types (social/personal conversations - LoCoMo)
    PERSON_ATTRIBUTE = "person_attribute"  # Personal traits, status, identity
    EVENT = "event"                        # Activities with time context
    RELATIONSHIP = "relationship"          # Interpersonal connections


@dataclass
class CanvasObject:
    """
    A cognitive object extracted from dialogue and stored on the canvas.

    These objects persist independently of the main conversation context,
    surviving context compression and enabling retrieval across the entire
    conversation history.
    """

    # Core identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ObjectType = ObjectType.KEY_FACT

    # Content
    content: str = ""           # Structured summary of the extracted info
    context: str = ""           # Why/how this was created

    # Grounding (CRITICAL for evaluation - prevents hallucination)
    quote: str = ""             # Exact quote from original text (verbatim)
    span: Optional[Tuple[int, int]] = None  # Character position (start, end) in source
    source: str = ""            # "user" or "assistant" - which message the quote is from

    # Fuzzy linking (for graph construction without ID dependency)
    references_text: List[str] = field(default_factory=list)  # Natural language refs, e.g. ["that postgres decision"]

    # Metadata
    turn_id: int = 0            # Which conversation turn
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0     # Extraction confidence [0, 1]

    # Time context (Phase 1: Temporal Grounding)
    event_time: Optional[str] = None       # Normalized time (ISO format or description)
    event_time_raw: Optional[str] = None   # Raw time expression (e.g., "yesterday")
    session_datetime: Optional[str] = None # Session occurrence time (e.g., "1:56 pm on 8 May, 2023")

    # Vector representation (for retrieval)
    embedding: Optional[List[float]] = None

    # Graph relationships
    references: List[str] = field(default_factory=list)      # Objects this refers to
    referenced_by: List[str] = field(default_factory=list)   # Objects that refer to this
    leads_to: List[str] = field(default_factory=list)        # Causal: this led to...
    caused_by: List[str] = field(default_factory=list)       # Causal: caused by...

    # VAGE: Learned schema fields (Phase 2)
    cluster_id: Optional[int] = None          # Learned cluster assignment
    type_confidence: float = 1.0              # LLM's confidence in type assignment
    vulnerability_score: Optional[float] = None  # Predicted vulnerability to compression

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "context": self.context,
            # Grounding fields
            "quote": self.quote,
            "span": list(self.span) if self.span else None,
            "source": self.source,
            "references_text": self.references_text,
            # Metadata
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            # Time context
            "event_time": self.event_time,
            "event_time_raw": self.event_time_raw,
            "session_datetime": self.session_datetime,
            "embedding": self.embedding,
            # Graph relationships (resolved IDs)
            "references": self.references,
            "referenced_by": self.referenced_by,
            "leads_to": self.leads_to,
            "caused_by": self.caused_by,
            # VAGE: Learned schema fields
            "cluster_id": self.cluster_id,
            "type_confidence": self.type_confidence,
            "vulnerability_score": self.vulnerability_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanvasObject":
        """Create from dictionary."""
        data = data.copy()
        data["type"] = ObjectType(data["type"])
        # Convert span back to tuple
        if data.get("span"):
            data["span"] = tuple(data["span"])
        # Handle legacy data without new fields
        data.setdefault("quote", "")
        data.setdefault("source", "")
        data.setdefault("references_text", [])
        # Handle legacy data without time context fields
        data.setdefault("event_time", None)
        data.setdefault("event_time_raw", None)
        data.setdefault("session_datetime", None)
        # Handle legacy data without VAGE fields
        data.setdefault("cluster_id", None)
        data.setdefault("type_confidence", 1.0)
        data.setdefault("vulnerability_score", None)
        # Remove legacy citation field if present
        data.pop("citation", None)
        return cls(**data)

    def __repr__(self) -> str:
        return f"CanvasObject({self.type.value}: {self.content[:50]}...)"


@dataclass
class ExtractionResult:
    """Result of extracting objects from a dialogue turn."""

    objects: List[CanvasObject] = field(default_factory=list)
    raw_response: str = ""
    extraction_time: float = 0.0
    model_used: str = ""
    filtered_objects: List[CanvasObject] = field(default_factory=list)  # Objects filtered by confidence
    filtered_count: int = 0  # Number of objects filtered out

    @property
    def count(self) -> int:
        return len(self.objects)

    @property
    def total_extracted(self) -> int:
        """Total objects extracted before filtering."""
        return len(self.objects) + self.filtered_count

    def by_type(self, obj_type: ObjectType) -> List[CanvasObject]:
        """Filter objects by type."""
        return [obj for obj in self.objects if obj.type == obj_type]


@dataclass
class RetrievalResult:
    """Result of retrieving objects from the canvas."""

    objects: List[CanvasObject] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    query: str = ""
    retrieval_time: float = 0.0

    @property
    def count(self) -> int:
        return len(self.objects)

    def top_k(self, k: int) -> List[CanvasObject]:
        """Get top k objects by score."""
        return self.objects[:k]
