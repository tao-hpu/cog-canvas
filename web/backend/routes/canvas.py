"""Canvas API endpoints."""

import os
from typing import Dict
from fastapi import APIRouter, HTTPException
from cogcanvas import Canvas, ObjectType

from models import (
    CanvasObjectResponse,
    StatsResponse,
    GraphData,
    GraphNode,
    GraphLink,
    RetrieveRequest,
    RetrieveResponse,
    ClearRequest,
    ErrorResponse,
)

router = APIRouter(prefix="/api/canvas", tags=["canvas"])

# Global canvas instances by session_id
_canvas_instances: Dict[str, Canvas] = {}


def get_canvas(session_id: str = "default") -> Canvas:
    """Get or create a canvas instance for the session."""
    if session_id not in _canvas_instances:
        # Read env vars at runtime (after load_dotenv in main.py)
        extractor_model = os.environ.get("MODEL_DEFAULT", "gpt-4o-mini")
        embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

        _canvas_instances[session_id] = Canvas(
            extractor_model=extractor_model, embedding_model=embedding_model
        )
        print(
            f"Created canvas with extractor={extractor_model}, embedding={embedding_model}"
        )
    return _canvas_instances[session_id]


@router.get("/objects", response_model=list[CanvasObjectResponse])
async def get_all_objects(session_id: str = "default"):
    """Get all canvas objects for the session."""
    canvas = get_canvas(session_id)
    objects = canvas.list_objects()
    return [CanvasObjectResponse.from_canvas_object(obj) for obj in objects]


@router.get("/stats", response_model=StatsResponse)
async def get_stats(session_id: str = "default"):
    """Get statistics about the canvas."""
    canvas = get_canvas(session_id)
    stats = canvas.stats()

    return StatsResponse(
        total_objects=stats["total_objects"],
        turn_count=stats["turn_count"],
        by_type=stats["by_type"],
        avg_confidence=stats.get("avg_confidence", 0.0),
    )


@router.get("/graph", response_model=GraphData)
async def get_graph(session_id: str = "default"):
    """Get graph structure for visualization with react-force-graph."""
    canvas = get_canvas(session_id)
    objects = canvas.list_objects()

    # Build nodes
    nodes = []
    for obj in objects:
        nodes.append(
            GraphNode(
                id=obj.id,
                name=obj.content[:50] + "..." if len(obj.content) > 50 else obj.content,
                type=obj.type.value,
                content=obj.content,
                context=obj.context,
                confidence=obj.confidence,
                turn_id=obj.turn_id,
                quote=obj.quote,
                timestamp=obj.timestamp,
            )
        )

    # Build links from relationships
    links = []
    for obj in objects:
        # References relationships
        for ref_id in obj.references:
            links.append(GraphLink(source=obj.id, target=ref_id, relation="references"))

        # Causal relationships (leads_to)
        for target_id in obj.leads_to:
            links.append(
                GraphLink(source=obj.id, target=target_id, relation="leads_to")
            )

        # Causal relationships (caused_by) - reverse direction
        for source_id in obj.caused_by:
            links.append(
                GraphLink(source=source_id, target=obj.id, relation="caused_by")
            )

    return GraphData(nodes=nodes, links=links)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_objects(request: RetrieveRequest):
    """Retrieve relevant canvas objects based on query."""
    canvas = get_canvas(request.session_id)

    # Parse obj_type if provided
    obj_type = None
    if request.obj_type:
        try:
            obj_type = ObjectType(request.obj_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid object type: {request.obj_type}. "
                f"Valid types: {[t.value for t in ObjectType]}",
            )

    # Retrieve objects
    result = canvas.retrieve(
        query=request.query,
        top_k=request.top_k,
        obj_type=obj_type,
        method=request.method,
        include_related=request.include_related,
    )

    return RetrieveResponse(
        objects=[
            CanvasObjectResponse.from_canvas_object(obj) for obj in result.objects
        ],
        scores=result.scores,
        query=result.query,
        retrieval_time=result.retrieval_time,
    )


@router.post("/clear")
async def clear_canvas(session_id: str = "default"):
    """Clear all objects from the canvas."""
    canvas = get_canvas(session_id)
    canvas.clear()
    return {"message": "Canvas cleared successfully", "session_id": session_id}
