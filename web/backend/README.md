# CogCanvas Web Backend

FastAPI backend for the CogCanvas Web Interface, providing RESTful APIs for chat streaming, canvas object management, and graph visualization.

## Features

- **Streaming Chat API**: Server-Sent Events (SSE) for real-time LLM response streaming
- **Canvas Object Management**: Extract, retrieve, and manage cognitive objects
- **Graph Visualization**: Generate graph data for react-force-graph
- **Session Management**: Multi-session support with in-memory storage
- **Mock Mode**: Built-in mock LLM and embedding models for development

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Server

### Method 1: Using uvicorn directly
```bash
uvicorn main:app --reload --port 3701
```

### Method 2: Using Python
```bash
python main.py
```

### Method 3: Using the start script
```bash
chmod +x start.sh
./start.sh
```

The server will start at `http://localhost:3701`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:3701/docs
- **ReDoc**: http://localhost:3701/redoc

## API Endpoints

### Chat

#### `POST /api/chat`
Stream chat response with Server-Sent Events.

**Request:**
```json
{
  "message": "Let's use PostgreSQL",
  "session_id": "default"
}
```

**Response:** SSE stream with events:
- `type: "token"` - LLM generated token
- `type: "done"` - Response complete
- `type: "extraction"` - Extracted canvas objects

#### `POST /api/chat/simple`
Non-streaming chat endpoint (for testing).

### Canvas

#### `GET /api/canvas?session_id=default`
Get all canvas objects.

**Response:**
```json
[
  {
    "id": "abc123",
    "type": "decision",
    "content": "Use PostgreSQL for database",
    "context": "Database selection discussion",
    "confidence": 0.95,
    "turn_id": 1,
    "quote": "Let's use PostgreSQL",
    "source": "user",
    "timestamp": 1702742400.0
  }
]
```

#### `GET /api/canvas/stats?session_id=default`
Get canvas statistics.

**Response:**
```json
{
  "object_count": 5,
  "turn_count": 3,
  "type_counts": {
    "decision": 2,
    "todo": 1,
    "key_fact": 2
  }
}
```

#### `GET /api/canvas/graph?session_id=default`
Get graph structure for visualization.

**Response:**
```json
{
  "nodes": [
    {
      "id": "abc123",
      "name": "Use PostgreSQL for database",
      "type": "decision",
      "content": "Use PostgreSQL for database",
      "confidence": 0.95,
      "turn_id": 1
    }
  ],
  "links": [
    {
      "source": "abc123",
      "target": "def456",
      "relation": "references"
    }
  ]
}
```

#### `POST /api/canvas/retrieve`
Retrieve relevant canvas objects.

**Request:**
```json
{
  "query": "database",
  "top_k": 5,
  "obj_type": "decision",
  "method": "semantic",
  "include_related": false,
  "session_id": "default"
}
```

**Response:**
```json
{
  "objects": [...],
  "scores": [0.95, 0.87, 0.75],
  "query": "database",
  "retrieval_time": 0.023
}
```

#### `POST /api/canvas/clear`
Clear all objects from canvas.

**Request:**
```json
{
  "session_id": "default"
}
```

## Configuration

### Port
The backend runs on port **3701** (configured in `main.py`).

### CORS
CORS is configured to allow:
- `http://localhost:3700` (frontend)
- `http://localhost:3000` (alternative frontend port)

### CogCanvas Models
By default, the backend uses mock models for development:
- `extractor_model="mock"` - Mock LLM for object extraction
- `embedding_model="mock"` - Mock embedding model for retrieval

To use real models, update `routes/canvas.py`:
```python
Canvas(
    extractor_model="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2"
)
```

## Session Management

Sessions are managed in-memory using a dictionary:
```python
_canvas_instances: Dict[str, Canvas] = {}
```

Each session has its own Canvas instance with isolated objects and state.

**Default session**: `"default"`

Pass `session_id` in requests to use different sessions:
```json
{
  "message": "Hello",
  "session_id": "user-123"
}
```

## Development

### Project Structure
```
backend/
├── main.py              # FastAPI app entry point
├── models.py            # Pydantic request/response models
├── requirements.txt     # Python dependencies
├── routes/
│   ├── __init__.py
│   ├── canvas.py        # Canvas API endpoints
│   └── chat.py          # Chat streaming API
└── README.md
```

### Adding New Endpoints

1. Add route in `routes/canvas.py` or `routes/chat.py`
2. Define request/response models in `models.py`
3. Import and use `get_canvas(session_id)` to access Canvas instance

### Testing

```bash
# Test health endpoint
curl http://localhost:3701/health

# Test simple chat
curl -X POST http://localhost:3701/api/chat/simple \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "default"}'

# Test canvas stats
curl http://localhost:3701/api/canvas/stats?session_id=default

# Test SSE streaming (use a SSE client or browser)
curl -N http://localhost:3701/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Let'\''s use PostgreSQL", "session_id": "default"}'
```

## Troubleshooting

### Import Errors
Make sure the cogcanvas package is accessible. The backend automatically adds the parent directory to Python path:
```python
cogcanvas_path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, cogcanvas_path)
```

### CORS Issues
If frontend can't connect, verify CORS origins in `main.py` match your frontend URL.

### Port Already in Use
Change the port in `main.py`:
```python
uvicorn.run("main:app", port=3701)  # Change to another port
```

## License

Same as CogCanvas project.
