"""Pydantic models for API request/response schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(default="default", description="Session identifier")


class CanvasObjectResponse(BaseModel):
    """Response model for a canvas object."""
    id: str
    type: str
    content: str
    context: str
    confidence: float
    turn_id: int
    quote: str = ""
    source: str = ""
    references_text: List[str] = []
    references: List[str] = []
    referenced_by: List[str] = []
    leads_to: List[str] = []
    caused_by: List[str] = []
    timestamp: float
    span: Optional[List[int]] = None

    @classmethod
    def from_canvas_object(cls, obj: Any) -> "CanvasObjectResponse":
        """Convert CanvasObject to response model."""
        return cls(
            id=obj.id,
            type=obj.type.value,
            content=obj.content,
            context=obj.context,
            confidence=obj.confidence,
            turn_id=obj.turn_id,
            quote=obj.quote,
            source=obj.source,
            references_text=obj.references_text,
            references=obj.references,
            referenced_by=obj.referenced_by,
            leads_to=obj.leads_to,
            caused_by=obj.caused_by,
            timestamp=obj.timestamp,
            span=list(obj.span) if obj.span else None,
        )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    role: str
    content: str
    extracted_objects: Optional[List[CanvasObjectResponse]] = None


class GraphNode(BaseModel):
    """Graph node for react-force-graph."""
    id: str
    name: str
    type: str
    content: str
    context: str
    confidence: float
    turn_id: int
    quote: str = ""
    timestamp: float


class GraphLink(BaseModel):
    """Graph link for react-force-graph."""
    source: str
    target: str
    relation: str  # "references", "leads_to", "caused_by"


class GraphData(BaseModel):
    """Graph data containing nodes and links."""
    nodes: List[GraphNode]
    links: List[GraphLink]


class StatsResponse(BaseModel):
    """Statistics about the canvas."""
    total_objects: int
    turn_count: int
    by_type: Dict[str, int]
    avg_confidence: float = 0.0


class RetrieveRequest(BaseModel):
    """Request for retrieving canvas objects."""
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    obj_type: Optional[str] = None
    method: str = Field(default="semantic", pattern="^(semantic|keyword)$")
    include_related: bool = False
    session_id: Optional[str] = Field(default="default", description="Session identifier")


class RetrieveResponse(BaseModel):
    """Response for retrieve endpoint."""
    objects: List[CanvasObjectResponse]
    scores: List[float]
    query: str
    retrieval_time: float


class ClearRequest(BaseModel):
    """Request to clear canvas."""
    session_id: Optional[str] = Field(default="default", description="Session identifier")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
