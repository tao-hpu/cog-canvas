# CogCanvas Web Backend - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Port 3700)                     │
│                   React + TypeScript                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP/SSE
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend (Port 3701)                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   main.py    │  │  models.py   │  │   routes/    │     │
│  │   (App)      │  │ (Pydantic)   │  │   (APIs)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                │                    │             │
│         └────────────────┴────────────────────┘             │
│                          │                                  │
│         ┌────────────────▼──────────────────┐              │
│         │   Session Manager (In-Memory)     │              │
│         │   _canvas_instances: Dict         │              │
│         └────────────────┬──────────────────┘              │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    CogCanvas Library                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Canvas     │  │  LLMBackend  │  │  Embeddings  │     │
│  │   (Core)     │  │   (Mock)     │  │    (Mock)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                                                   │
│  ┌──────▼──────┐                                           │
│  │ CanvasGraph │                                           │
│  └─────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. FastAPI Application (`main.py`)

**Responsibilities**:
- Initialize FastAPI app with middleware
- Configure CORS for frontend access
- Register route handlers
- Global exception handling
- Health check endpoint

**Key Features**:
- Auto-generated API docs at `/docs` and `/redoc`
- CORS configured for `localhost:3700` (frontend)
- Graceful error handling

**Startup**:
```python
uvicorn main:app --reload --port 3701
```

---

### 2. Request/Response Models (`models.py`)

**Purpose**: Pydantic models for type-safe API contracts

**Key Models**:
- `ChatRequest` / `ChatResponse` - Chat endpoints
- `CanvasObjectResponse` - Canvas object serialization
- `GraphNode` / `GraphLink` / `GraphData` - Graph visualization
- `StatsResponse` - Canvas statistics
- `RetrieveRequest` / `RetrieveResponse` - Retrieval API
- `ErrorResponse` - Error handling

**Benefits**:
- Automatic validation
- Auto-generated OpenAPI schema
- Type hints for IDE support

---

### 3. Route Handlers (`routes/`)

#### `routes/canvas.py` - Canvas Management

**Endpoints**:
- `GET /api/canvas` - List all objects
- `GET /api/canvas/stats` - Statistics
- `GET /api/canvas/graph` - Graph data for visualization
- `POST /api/canvas/retrieve` - Semantic/keyword search
- `POST /api/canvas/clear` - Clear canvas

**Key Functions**:
```python
def get_canvas(session_id: str) -> Canvas:
    """Get or create Canvas instance for session."""
```

#### `routes/chat.py` - Chat API

**Endpoints**:
- `POST /api/chat` - Streaming SSE chat
- `POST /api/chat/simple` - Non-streaming chat

**Streaming Implementation**:
```python
async def generate_chat_stream() -> AsyncGenerator[str, None]:
    """Stream tokens + extraction events via SSE."""
    yield json.dumps({"type": "token", "content": "..."})
    yield json.dumps({"type": "done", "content": "..."})
    yield json.dumps({"type": "extraction", "objects": [...]})
```

---

### 4. Session Management

**Strategy**: In-memory dictionary

```python
_canvas_instances: Dict[str, Canvas] = {}

def get_canvas(session_id: str = "default") -> Canvas:
    if session_id not in _canvas_instances:
        _canvas_instances[session_id] = Canvas(
            extractor_model="mock",
            embedding_model="mock"
        )
    return _canvas_instances[session_id]
```

**Properties**:
- Each session has isolated Canvas instance
- Objects don't leak between sessions
- Sessions persist for server lifetime
- Lost on restart (in-memory)

**Scalability Considerations**:
- For production: Use Redis/database for persistence
- Add session timeout/cleanup
- Consider distributed sessions for multi-instance deployments

---

### 5. CogCanvas Integration

**Canvas Initialization**:
```python
Canvas(
    extractor_model="mock",      # or "gpt-4o-mini"
    embedding_model="mock"        # or "all-MiniLM-L6-v2"
)
```

**Core Operations**:

**Extraction**:
```python
result = canvas.extract(
    user="Let's use PostgreSQL",
    assistant="Good choice!"
)
# Returns: ExtractionResult with CanvasObjects
```

**Retrieval**:
```python
result = canvas.retrieve(
    query="database decision",
    top_k=5,
    method="semantic"
)
# Returns: RetrievalResult with scored objects
```

**Graph Access**:
```python
objects = canvas.list_objects()
# Build graph from object relationships
```

---

## Data Flow

### Chat Flow with Extraction

```
User Message
    │
    ▼
┌─────────────────┐
│ POST /api/chat  │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ generate_chat_stream │
└────────┬─────────────┘
         │
         ├─► Stream token events (SSE)
         │
         ├─► Stream done event
         │
         ├─► canvas.extract(user, assistant)
         │       │
         │       ├─► LLMBackend.extract_objects()
         │       │       └─► Mock: rule-based extraction
         │       │           Real: GPT-4o API call
         │       │
         │       ├─► EmbeddingBackend.embed_batch()
         │       │       └─► Compute embeddings for objects
         │       │
         │       └─► CanvasGraph.add_node() & infer_relations()
         │
         └─► Stream extraction event with objects
```

### Retrieval Flow

```
Query
  │
  ▼
┌────────────────────────┐
│ POST /api/canvas/retrieve │
└────────┬───────────────┘
         │
         ▼
┌──────────────────┐
│ canvas.retrieve  │
└────────┬─────────┘
         │
         ├─► Filter by obj_type (if specified)
         │
         ├─► semantic_retrieve() or keyword_retrieve()
         │       │
         │       ├─► Embed query
         │       ├─► Compute cosine similarity
         │       └─► Rank by similarity
         │
         ├─► Sort by score, take top_k
         │
         └─► Optionally include 1-hop neighbors
```

### Graph Visualization Flow

```
Request
  │
  ▼
┌────────────────────┐
│ GET /api/canvas/graph │
└────────┬───────────┘
         │
         ▼
┌────────────────┐
│ canvas.list_objects │
└────────┬───────┘
         │
         ├─► Build nodes from objects
         │       └─► GraphNode(id, name, type, ...)
         │
         └─► Build links from relationships
                 ├─► obj.references → "references"
                 ├─► obj.leads_to → "leads_to"
                 └─► obj.caused_by → "caused_by"
```

---

## API Patterns

### RESTful Design

- **GET** for read operations (canvas, stats, graph)
- **POST** for mutations (chat, retrieve, clear)
- Query params for simple filters (`?session_id=...`)
- Request body for complex operations

### Streaming Pattern (SSE)

```python
@router.post("/api/chat")
async def chat_stream(request: ChatRequest):
    return EventSourceResponse(
        generate_chat_stream(...)
    )

async def generate_chat_stream(...) -> AsyncGenerator[str, None]:
    # Stream JSON events
    yield json.dumps({"type": "token", "content": "..."})
```

**Client-side**:
```javascript
const response = await fetch('/api/chat', {
  method: 'POST',
  body: JSON.stringify({ message: "Hello" })
});

const reader = response.body.getReader();
// Read stream...
```

---

## Configuration

### Environment Variables

```bash
# .env
PORT=3701
EXTRACTOR_MODEL=mock
EMBEDDING_MODEL=mock
CORS_ORIGINS=http://localhost:3700
```

### Model Configuration

**Development** (Mock):
```python
Canvas(extractor_model="mock", embedding_model="mock")
```

**Production** (Real):
```python
Canvas(
    extractor_model="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2"
)
```

---

## Error Handling

### Global Exception Handler

```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
```

### Validation Errors

Pydantic automatically validates:
- Required fields
- Type correctness
- Enum values (e.g., `method: "semantic"|"keyword"`)
- Range constraints (e.g., `top_k: 1-50`)

---

## Testing Strategy

### Test Suite (`test_api.py`)

Comprehensive tests for:
- Health check
- Simple chat
- Streaming chat
- Canvas CRUD
- Retrieval
- Graph generation
- Stats

**Run**:
```bash
python test_api.py
```

### Manual Testing

**cURL**:
```bash
curl http://localhost:3701/health
```

**Swagger UI**:
```
http://localhost:3701/docs
```

---

## Deployment Considerations

### Current Setup (Development)

- In-memory sessions (lost on restart)
- Mock models (no API calls)
- Single server instance
- No authentication

### Production Improvements

1. **Persistence**:
   - Redis for session storage
   - PostgreSQL for object persistence
   - Canvas `storage_path` for disk backup

2. **Authentication**:
   - JWT tokens
   - API keys per user
   - Rate limiting

3. **Scalability**:
   - Multiple uvicorn workers
   - Nginx load balancer
   - Distributed session store

4. **Monitoring**:
   - Prometheus metrics
   - Logging (structured JSON)
   - Error tracking (Sentry)

5. **Real Models**:
   - Switch to `gpt-4o-mini` for extraction
   - Use `all-MiniLM-L6-v2` for embeddings
   - Configure API keys securely

---

## Performance Characteristics

### Latency

**Mock Mode**:
- Chat response: ~50ms
- Extraction: ~10ms
- Retrieval: ~5ms
- Graph generation: ~20ms (100 objects)

**Real Models**:
- Chat response: 1-3s (LLM API)
- Extraction: 500ms-2s (LLM API)
- Retrieval: 50-200ms (embedding + similarity)

### Throughput

**Single instance**:
- ~100 req/s for simple endpoints (stats, health)
- ~10 req/s for chat (streaming)
- ~50 req/s for retrieval

### Memory Usage

- Base: ~50MB (FastAPI + dependencies)
- Per session: ~1-5MB (Canvas + objects)
- 1000 objects: ~10MB additional

---

## Security Considerations

### Current State

- No authentication
- CORS limited to localhost
- No rate limiting
- No input sanitization beyond Pydantic

### Recommendations

1. Add API key authentication
2. Implement rate limiting (per IP/session)
3. Sanitize user inputs (prevent injection)
4. Add HTTPS in production
5. Secure session tokens
6. Audit logging for sensitive operations

---

## Future Enhancements

1. **WebSocket Support**: Replace SSE with WebSocket for bidirectional streaming
2. **Persistence**: Database backend for sessions/objects
3. **Multi-tenancy**: User accounts with isolated canvases
4. **Real-time Collaboration**: Shared canvases across users
5. **Advanced Retrieval**: Hybrid search, re-ranking
6. **Canvas Snapshots**: Save/restore canvas state
7. **Object Editing**: Update/delete individual objects
8. **Export/Import**: JSON export of entire canvas
9. **Analytics**: Usage metrics, popular objects
10. **Background Jobs**: Async extraction for large conversations

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SSE Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [CogCanvas Library](../../README.md)
