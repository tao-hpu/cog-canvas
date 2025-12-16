# CogCanvas Web Backend - API Specification

Version: 0.1.0
Base URL: `http://localhost:3701`

## Table of Contents

- [Authentication](#authentication)
- [Sessions](#sessions)
- [Endpoints](#endpoints)
  - [Chat Endpoints](#chat-endpoints)
  - [Canvas Endpoints](#canvas-endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)

---

## Authentication

Currently, the API does not require authentication. Session isolation is handled via `session_id` parameter.

---

## Sessions

Each session maintains an isolated Canvas instance with its own objects and state.

**Default session**: `"default"`

**Session parameters**:
- Passed in request body as `session_id` (chat endpoints)
- Passed as query parameter `?session_id=xxx` (canvas endpoints)

**Session lifecycle**:
- Sessions are created automatically on first use
- Sessions persist in memory while server is running
- Sessions are lost on server restart (in-memory storage)

---

## Endpoints

### Chat Endpoints

#### POST /api/chat

Stream chat response with Server-Sent Events (SSE).

**Request Body**:
```json
{
  "message": "Let's use PostgreSQL for our database",
  "session_id": "default"  // optional
}
```

**Response**: SSE stream with JSON events

**Event Types**:

1. **Token Event** (streaming response):
```json
{
  "type": "token",
  "content": "I "
}
```

2. **Done Event** (response complete):
```json
{
  "type": "done",
  "content": "I understand you want to discuss this..."
}
```

3. **Extraction Event** (extracted canvas objects):
```json
{
  "type": "extraction",
  "objects": [
    {
      "id": "abc123",
      "type": "decision",
      "content": "Use PostgreSQL for database",
      "confidence": 0.95,
      ...
    }
  ],
  "count": 2
}
```

4. **Error Event**:
```json
{
  "type": "error",
  "error": "Error message"
}
```

**Example Usage** (JavaScript):
```javascript
const eventSource = new EventSource('/api/chat', {
  method: 'POST',
  body: JSON.stringify({ message: "Hello", session_id: "user-123" })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'token':
      console.log('Token:', data.content);
      break;
    case 'done':
      console.log('Response complete:', data.content);
      break;
    case 'extraction':
      console.log('Extracted objects:', data.objects);
      break;
  }
};
```

---

#### POST /api/chat/simple

Non-streaming chat endpoint (for testing/debugging).

**Request Body**:
```json
{
  "message": "Let's use PostgreSQL",
  "session_id": "default"
}
```

**Response**:
```json
{
  "role": "assistant",
  "content": "I understand you want to discuss this...",
  "extracted_objects": [
    {
      "id": "abc123",
      "type": "decision",
      "content": "Use PostgreSQL for database",
      "context": "Database selection discussion",
      "confidence": 0.95,
      "turn_id": 1,
      "quote": "Let's use PostgreSQL",
      "source": "user",
      "references": [],
      "referenced_by": [],
      "leads_to": [],
      "caused_by": [],
      "timestamp": 1702742400.0,
      "span": null
    }
  ]
}
```

---

### Canvas Endpoints

#### GET /api/canvas

Get all canvas objects for the session.

**Query Parameters**:
- `session_id` (optional, default: "default")

**Response**:
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
    "references_text": [],
    "references": [],
    "referenced_by": [],
    "leads_to": ["def456"],
    "caused_by": [],
    "timestamp": 1702742400.0,
    "span": null
  }
]
```

---

#### GET /api/canvas/stats

Get statistics about the canvas.

**Query Parameters**:
- `session_id` (optional, default: "default")

**Response**:
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

---

#### GET /api/canvas/graph

Get graph structure for visualization with react-force-graph.

**Query Parameters**:
- `session_id` (optional, default: "default")

**Response**:
```json
{
  "nodes": [
    {
      "id": "abc123",
      "name": "Use PostgreSQL for database",
      "type": "decision",
      "content": "Use PostgreSQL for database",
      "context": "Database selection discussion",
      "confidence": 0.95,
      "turn_id": 1,
      "quote": "Let's use PostgreSQL",
      "timestamp": 1702742400.0
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

**Graph Structure**:
- **nodes**: Array of graph nodes
- **links**: Array of edges between nodes

**Relation Types**:
- `"references"`: Object A references object B
- `"leads_to"`: Object A leads to object B (causal)
- `"caused_by"`: Object A is caused by object B (causal reverse)

---

#### POST /api/canvas/retrieve

Retrieve relevant canvas objects based on semantic or keyword search.

**Request Body**:
```json
{
  "query": "database decision",
  "top_k": 5,
  "obj_type": "decision",        // optional: decision|todo|key_fact|reminder|insight
  "method": "semantic",           // semantic|keyword
  "include_related": false,       // include 1-hop neighbors
  "session_id": "default"
}
```

**Response**:
```json
{
  "objects": [
    {
      "id": "abc123",
      "type": "decision",
      "content": "Use PostgreSQL for database",
      ...
    }
  ],
  "scores": [0.95, 0.87, 0.75],
  "query": "database decision",
  "retrieval_time": 0.023
}
```

**Retrieval Methods**:
- `"semantic"`: Use embedding similarity (default, more intelligent)
- `"keyword"`: Simple keyword matching

**Object Types**:
- `"decision"`: Decisions made
- `"todo"`: Action items, tasks
- `"key_fact"`: Important facts, numbers, names
- `"reminder"`: Constraints, preferences, rules
- `"insight"`: Conclusions, learnings, realizations

---

#### POST /api/canvas/clear

Clear all objects from the canvas.

**Request Body**:
```json
{
  "session_id": "default"
}
```

**Response**:
```json
{
  "message": "Canvas cleared successfully",
  "session_id": "default"
}
```

---

## Data Models

### CanvasObjectResponse

```typescript
interface CanvasObjectResponse {
  id: string;                    // Unique identifier
  type: string;                  // Object type
  content: string;               // Structured content
  context: string;               // Why/how created
  confidence: number;            // Extraction confidence [0, 1]
  turn_id: number;               // Conversation turn number
  quote: string;                 // Exact quote from source
  source: string;                // "user" or "assistant"
  references_text: string[];     // Natural language references
  references: string[];          // Referenced object IDs
  referenced_by: string[];       // Objects that reference this
  leads_to: string[];            // Causal: this led to...
  caused_by: string[];           // Causal: caused by...
  timestamp: number;             // Unix timestamp
  span: [number, number] | null; // Character span in source
}
```

### GraphNode

```typescript
interface GraphNode {
  id: string;           // Node ID (same as object ID)
  name: string;         // Display name (truncated content)
  type: string;         // Object type
  content: string;      // Full content
  context: string;      // Context
  confidence: number;   // Confidence score
  turn_id: number;      // Turn ID
  quote: string;        // Quote
  timestamp: number;    // Timestamp
}
```

### GraphLink

```typescript
interface GraphLink {
  source: string;    // Source node ID
  target: string;    // Target node ID
  relation: string;  // "references" | "leads_to" | "caused_by"
}
```

### StatsResponse

```typescript
interface StatsResponse {
  object_count: number;           // Total objects
  turn_count: number;             // Total turns processed
  type_counts: Record<string, number>;  // Objects per type
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error message",
  "detail": "Detailed error information"
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

### Common Errors

**Invalid Object Type**:
```json
{
  "error": "Invalid object type: wrong_type. Valid types: ['decision', 'todo', 'key_fact', 'reminder', 'insight']"
}
```

**Connection Error**:
```json
{
  "error": "Internal server error",
  "detail": "Connection refused"
}
```

---

## Rate Limiting

Currently not implemented. May be added in future versions.

---

## Versioning

API version is included in the root endpoint response:

```bash
curl http://localhost:3701/
```

```json
{
  "name": "CogCanvas API",
  "version": "0.1.0",
  ...
}
```

---

## Testing

### Using cURL

**Test health**:
```bash
curl http://localhost:3701/health
```

**Test simple chat**:
```bash
curl -X POST http://localhost:3701/api/chat/simple \
  -H "Content-Type: application/json" \
  -d '{"message": "Let'\''s use PostgreSQL", "session_id": "test"}'
```

**Test canvas stats**:
```bash
curl http://localhost:3701/api/canvas/stats?session_id=test
```

**Test retrieval**:
```bash
curl -X POST http://localhost:3701/api/canvas/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "database", "top_k": 3, "session_id": "test"}'
```

### Using Python Test Suite

```bash
python test_api.py
```

This runs comprehensive tests for all endpoints.

---

## Additional Resources

- **Swagger UI**: http://localhost:3701/docs
- **ReDoc**: http://localhost:3701/redoc
- **GitHub**: [CogCanvas Repository]

---

## Support

For issues or questions:
1. Check the [README.md](README.md) file
2. Review API documentation at `/docs`
3. Run test suite with `python test_api.py`
