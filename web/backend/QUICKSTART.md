# CogCanvas Web Backend - Quick Start Guide

Get the backend running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- CogCanvas library installed (parent directory)

## Installation

```bash
# Navigate to backend directory
cd /Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/web/backend

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

### Option 1: Quick Start (Recommended)

```bash
./start.sh
```

### Option 2: Manual Start

```bash
uvicorn main:app --reload --port 3701
```

### Option 3: Python Script

```bash
python main.py
```

## Verify Installation

Once the server is running, open your browser to:

**Swagger UI**: http://localhost:3701/docs

You should see the interactive API documentation.

## Quick Test

### Using cURL

```bash
# Health check
curl http://localhost:3701/health

# Simple chat
curl -X POST http://localhost:3701/api/chat/simple \
  -H "Content-Type: application/json" \
  -d '{"message": "Let'\''s use PostgreSQL for our database"}'

# Get canvas stats
curl http://localhost:3701/api/canvas/stats
```

### Using Python Test Suite

```bash
python test_api.py
```

Expected output:
```
============================================================
CogCanvas Web Backend API Test Suite
============================================================
Testing /health endpoint...
✓ Health check passed

Testing / endpoint...
✓ Root endpoint passed

Testing /api/chat/simple endpoint...
✓ Chat response: I understand you want to discuss this...
✓ Extracted 1 objects
  - decision: Decision from turn 1: Let's use PostgreSQL...

...

============================================================
✓ All tests passed!
============================================================
```

## Configuration

### Port Configuration

Default port: **3701**

To change the port, edit `main.py`:
```python
uvicorn.run("main:app", port=3701)  # Change to your desired port
```

### CORS Configuration

Default allowed origins:
- `http://localhost:3700` (frontend)
- `http://localhost:3000`

To add more origins, edit `main.py`:
```python
allow_origins=[
    "http://localhost:3700",
    "http://your-frontend-url.com",
]
```

### Model Configuration

Default: Mock models (for development)

To use real models, edit `routes/canvas.py`:
```python
Canvas(
    extractor_model="gpt-4o-mini",      # Real OpenAI model
    embedding_model="all-MiniLM-L6-v2"  # Real embedding model
)
```

And set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Common Issues

### Port Already in Use

```
Error: [Errno 48] Address already in use
```

**Solution**: Kill the process using port 3701:
```bash
lsof -ti:3701 | xargs kill -9
```

Or change the port in `main.py`.

### Import Error: cogcanvas not found

```
ModuleNotFoundError: No module named 'cogcanvas'
```

**Solution**: Make sure you're in the correct directory and the cogcanvas library is accessible. The backend automatically adds the parent path to Python's sys.path.

If still failing, install cogcanvas:
```bash
cd /Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas
pip install -e .
```

### Missing Dependencies

```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

## API Usage Examples

### Chat Streaming (JavaScript)

```javascript
async function streamChat(message) {
  const response = await fetch('http://localhost:3701/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: 'user-123' })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));

        if (data.type === 'token') {
          console.log('Token:', data.content);
        } else if (data.type === 'done') {
          console.log('Response complete');
        } else if (data.type === 'extraction') {
          console.log('Extracted objects:', data.objects);
        }
      }
    }
  }
}

streamChat('Let\'s use PostgreSQL');
```

### Retrieve Objects (Python)

```python
import requests

response = requests.post('http://localhost:3701/api/canvas/retrieve', json={
    'query': 'database decision',
    'top_k': 5,
    'method': 'semantic',
    'session_id': 'user-123'
})

result = response.json()
for obj, score in zip(result['objects'], result['scores']):
    print(f"[{score:.2f}] {obj['content']}")
```

### Get Graph Data (Python)

```python
import requests

response = requests.get('http://localhost:3701/api/canvas/graph',
                       params={'session_id': 'user-123'})

graph = response.json()
print(f"Nodes: {len(graph['nodes'])}")
print(f"Links: {len(graph['links'])}")

# Use with react-force-graph
for node in graph['nodes']:
    print(f"- {node['name']} ({node['type']})")
```

## Next Steps

1. **Read the API Spec**: [API_SPEC.md](API_SPEC.md)
2. **Understand Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Explore API Docs**: http://localhost:3701/docs
4. **Run Tests**: `python test_api.py`
5. **Connect Frontend**: Configure frontend to use `http://localhost:3701`

## Development Tips

### Live Reload

The server auto-reloads on code changes (thanks to `--reload` flag):
```bash
uvicorn main:app --reload --port 3701
```

Just edit files and the server will restart automatically.

### Debugging

Add print statements or use a debugger:
```python
import pdb; pdb.set_trace()  # Breakpoint
```

Or use logging:
```python
import logging
logging.info("Debug message")
```

### Testing New Endpoints

1. Add route in `routes/canvas.py` or `routes/chat.py`
2. Test via Swagger UI: http://localhost:3701/docs
3. Add test case to `test_api.py`

### Session Management

Each session_id gets its own Canvas instance:
```python
# Session 1
{"message": "Hello", "session_id": "alice"}

# Session 2
{"message": "Hello", "session_id": "bob"}
```

They don't share objects!

## Production Deployment

For production, consider:

1. **Use Gunicorn**:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:3701
```

2. **Enable HTTPS**:
```bash
uvicorn main:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

3. **Use Real Models**:
```python
Canvas(extractor_model="gpt-4o-mini", embedding_model="all-MiniLM-L6-v2")
```

4. **Add Authentication**:
Implement JWT or API key authentication.

5. **Setup Database**:
Replace in-memory sessions with Redis/PostgreSQL.

## Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **CogCanvas Docs**: ../../README.md
- **API Spec**: API_SPEC.md
- **Architecture**: ARCHITECTURE.md

## Support

Issues? Check:
1. Server logs for errors
2. http://localhost:3701/docs for API reference
3. Run `python test_api.py` to identify problems

Happy coding! 🚀
