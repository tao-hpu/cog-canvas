# Quick Start: LLM Backends

A quick reference for using CogCanvas with different LLM backends.

## Installation

```bash
# Install with OpenAI support
pip install cogcanvas[openai]

# Install with Anthropic support
pip install cogcanvas[anthropic]

# Install with all backends
pip install cogcanvas[all]
```

## Setup API Keys

```bash
# OpenAI
export OPENAI_API_KEY=sk-your-openai-key

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

## Quick Examples

### Mock Backend (No API Key)

```python
from cogcanvas import Canvas
from cogcanvas.llm import get_backend

backend = get_backend("mock")
canvas = Canvas(llm_backend=backend)

result = canvas.extract(
    user="Let's use PostgreSQL for the database",
    assistant="Good choice!"
)

print(f"Extracted {len(result.objects)} objects")
```

### OpenAI Backend

```python
from cogcanvas import Canvas
from cogcanvas.llm import get_backend

backend = get_backend("openai", model="gpt-4o-mini")
canvas = Canvas(llm_backend=backend)

result = canvas.extract(
    user="The API rate limit is 1000/hour. We need to add caching.",
    assistant="I'll implement Redis caching to stay within limits."
)

for obj in result.objects:
    print(f"[{obj.type.value}] {obj.content}")
```

### Anthropic Backend

```python
from cogcanvas import Canvas
from cogcanvas.llm import get_backend

backend = get_backend("anthropic", model="claude-3-5-haiku-latest")
canvas = Canvas(llm_backend=backend)

result = canvas.extract(
    user="Should we use microservices or monolith?",
    assistant="For a small team, start with a monolith."
)

for obj in result.objects:
    print(f"[{obj.type.value}] {obj.content} (confidence: {obj.confidence})")
```

## Backend Comparison

| Feature | Mock | OpenAI | Anthropic |
|---------|------|--------|-----------|
| Speed | Instant | ~1s | ~0.5s |
| Cost | Free | $0.15/1M tokens | $0.80/1M tokens |
| Quality | Basic | High | Very High |
| API Key | ❌ No | ✅ Yes | ✅ Yes |
| Embeddings | Mock | Real | Mock |

## Object Types

All backends extract these five types:

- **decision**: Choices made
- **todo**: Action items
- **key_fact**: Important facts
- **reminder**: User preferences
- **insight**: Realizations

## Common Patterns

### Switch Backends

```python
# Development
dev_backend = get_backend("mock")
dev_canvas = Canvas(llm_backend=dev_backend)

# Production
prod_backend = get_backend("openai", model="gpt-4o-mini")
prod_canvas = Canvas(llm_backend=prod_backend)
```

### Automatic Selection

```python
# Uses mock by default
canvas = Canvas()

# Auto-uses OpenAI if API key is set
canvas = Canvas(extractor_model="gpt-4o-mini")
```

### Error Handling

```python
from cogcanvas.llm import get_backend

try:
    backend = get_backend("openai")
except ValueError as e:
    print(f"API key not set: {e}")
    backend = get_backend("mock")  # Fallback
```

## Testing

```bash
# Run tests
pytest tests/test_llm_backends.py -v

# Run with specific backend
export OPENAI_API_KEY=sk-...
pytest tests/test_llm_backends.py::TestOpenAIBackend -v
```

## Troubleshooting

### "API key required" Error

```bash
# Set the appropriate environment variable
export OPENAI_API_KEY=your-key
# or
export ANTHROPIC_API_KEY=your-key
```

### "No module named 'openai'" Error

```bash
# Install the optional dependency
pip install cogcanvas[openai]
```

### "Empty extraction" Issue

Check that:
- Conversation has extractable content
- API key is valid
- Model name is correct

## Next Steps

- Read full documentation: `LLM_BACKENDS.md`
- See more examples: `examples/llm_backends_demo.py`
- Run simple demo: `examples/simple_backend_usage.py`

## Support

For issues or questions, see:
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Full API reference: `LLM_BACKENDS.md`
- Test suite: `tests/test_llm_backends.py`
