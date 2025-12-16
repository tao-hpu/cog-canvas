# CogCanvas LLM Backends

This document describes the multiple LLM backend implementation for CogCanvas (P2.4).

## Overview

CogCanvas now supports three LLM backends for extracting canvas objects from conversation turns:

1. **Mock Backend** - Rule-based extraction for testing (no API required)
2. **OpenAI Backend** - Uses GPT models with structured output
3. **Anthropic Backend** - Uses Claude models with tool calling

## Backend Architecture

### Base Interface

All backends implement the `LLMBackend` abstract class:

```python
class LLMBackend(ABC):
    @abstractmethod
    def extract_objects(
        self,
        user_message: str,
        assistant_message: str,
        existing_objects: Optional[List[CanvasObject]] = None,
    ) -> List[CanvasObject]:
        """Extract canvas objects from a dialogue turn."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
```

### Available Backends

#### 1. MockLLMBackend

Simple rule-based extraction for testing without API calls.

**Features:**
- Keyword-based pattern matching
- No external dependencies
- Deterministic behavior
- Mock embeddings using hash functions

**Usage:**
```python
from cogcanvas import Canvas
from cogcanvas.llm import MockLLMBackend

backend = MockLLMBackend()
canvas = Canvas(llm_backend=backend)
```

#### 2. OpenAIBackend

Uses OpenAI's GPT models for intelligent extraction.

**Features:**
- Structured JSON output parsing
- Configurable models (default: gpt-4o-mini)
- Real embeddings via text-embedding-3-small
- Automatic deduplication with existing objects

**Requirements:**
```bash
pip install cogcanvas[openai]
export OPENAI_API_KEY=your-key-here
```

**Usage:**
```python
from cogcanvas import Canvas
from cogcanvas.llm import OpenAIBackend

# Using default model (gpt-4o-mini)
backend = OpenAIBackend()

# Using specific model
backend = OpenAIBackend(
    model="gpt-4o",
    embedding_model="text-embedding-3-small"
)

canvas = Canvas(llm_backend=backend)
```

**Extraction Prompt:**
The OpenAI backend uses a carefully crafted prompt that:
- Defines all 5 object types (decision, todo, key_fact, reminder, insight)
- Provides clear examples
- Requests structured JSON output
- Includes existing objects for deduplication

#### 3. AnthropicBackend

Uses Anthropic's Claude models with tool calling for structured extraction.

**Features:**
- Tool-based structured output (more reliable than JSON parsing)
- Configurable models (default: claude-3-5-haiku-latest)
- Provenance tracking via citation field
- Mock embeddings (Anthropic doesn't provide embedding API)

**Requirements:**
```bash
pip install cogcanvas[anthropic]
export ANTHROPIC_API_KEY=your-key-here
```

**Usage:**
```python
from cogcanvas import Canvas
from cogcanvas.llm import AnthropicBackend

# Using default model (claude-3-5-haiku-latest)
backend = AnthropicBackend()

# Using specific model
backend = AnthropicBackend(model="claude-3-5-sonnet-latest")

canvas = Canvas(llm_backend=backend)
```

**Tool Schema:**
The Anthropic backend defines a structured tool `extract_canvas_objects` with:
- Strict type validation (enum for object types)
- Required citation field for provenance
- Confidence scoring
- Context explanation

## Factory Function

Use `get_backend()` for convenient backend creation:

```python
from cogcanvas.llm import get_backend

# Create backends by name
mock = get_backend("mock")
openai = get_backend("openai", model="gpt-4o-mini")
anthropic = get_backend("anthropic", model="claude-3-5-haiku-latest")
```

## Canvas Integration

### Direct Backend Usage

```python
from cogcanvas import Canvas
from cogcanvas.llm import OpenAIBackend

backend = OpenAIBackend(model="gpt-4o-mini")
canvas = Canvas(llm_backend=backend)

result = canvas.extract(
    user="Let's use PostgreSQL for the database",
    assistant="Great choice! PostgreSQL is reliable and feature-rich."
)

print(f"Extracted {len(result.objects)} objects in {result.extraction_time:.3f}s")
for obj in result.objects:
    print(f"  [{obj.type.value}] {obj.content} (confidence: {obj.confidence})")
```

### Automatic Backend Selection

The Canvas class can automatically initialize backends:

```python
# Using mock (default)
canvas = Canvas()

# Auto-initialize OpenAI if API key is available
canvas = Canvas(extractor_model="gpt-4o-mini")
```

## Extraction Quality

### Object Types Extracted

All backends extract five types of cognitive objects:

1. **decision** - Choices or decisions made
   - Example: "Let's use PostgreSQL for the database"

2. **todo** - Action items and tasks
   - Example: "Need to implement authentication by next week"

3. **key_fact** - Important facts, numbers, constraints
   - Example: "API rate limit is 1000 requests per hour"

4. **reminder** - User preferences and rules
   - Example: "User prefers TypeScript over JavaScript"

5. **insight** - Conclusions and learnings
   - Example: "The performance bottleneck is in the database queries"

### Extraction Guidelines

All backends follow these principles:
- Conservative extraction (only genuinely important information)
- Self-contained objects (understandable without original context)
- Confidence scoring (0.0-1.0)
- Context explanation (why this was extracted)
- Deduplication awareness (existing objects provided for context)

## Error Handling

All backends implement graceful error handling:

```python
try:
    objects = backend.extract_objects(user_msg, assistant_msg)
except Exception as e:
    print(f"Extraction error: {e}")
    # Returns empty list, doesn't crash
    return []
```

## Performance Considerations

### Mock Backend
- **Speed**: Instant (microseconds)
- **Cost**: Free
- **Quality**: Basic keyword matching

### OpenAI Backend
- **Speed**: ~500-2000ms per extraction
- **Cost**: $0.150 per 1M input tokens (gpt-4o-mini)
- **Quality**: High, good for production

### Anthropic Backend
- **Speed**: ~300-1500ms per extraction
- **Cost**: $0.80 per 1M input tokens (claude-3-5-haiku)
- **Quality**: Very high, excellent for complex extraction

## Testing

Run the test suite:

```bash
# Without API keys (mock only)
pytest tests/test_llm_backends.py -v

# With OpenAI API key
export OPENAI_API_KEY=your-key
pytest tests/test_llm_backends.py -v

# With Anthropic API key
export ANTHROPIC_API_KEY=your-key
pytest tests/test_llm_backends.py -v

# With both
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key
pytest tests/test_llm_backends.py -v
```

## Example Demo

Run the comprehensive demo:

```bash
# Install in development mode
pip install -e .

# Set API keys (optional)
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key

# Run demo
python examples/llm_backends_demo.py
```

The demo will:
1. Test each available backend
2. Compare extraction quality
3. Show timing and performance metrics
4. Demonstrate the factory function

## Implementation Files

### Core Files

- `cogcanvas/llm/base.py` - Abstract base class and mock backend
- `cogcanvas/llm/openai.py` - OpenAI backend implementation
- `cogcanvas/llm/anthropic_backend.py` - Anthropic backend implementation
- `cogcanvas/llm/__init__.py` - Exports and factory function

### Integration

- `cogcanvas/canvas.py` - Updated to use llm_backend parameter
- `cogcanvas/models.py` - CanvasObject and related models (unchanged)

### Tests and Examples

- `tests/test_llm_backends.py` - Comprehensive test suite
- `examples/llm_backends_demo.py` - Demo script showcasing all backends

## Future Enhancements

Potential improvements for future versions:

1. **Async Support** - Add async versions of extract_objects
2. **Streaming** - Support streaming responses for real-time extraction
3. **Custom Prompts** - Allow user-defined extraction prompts
4. **Hybrid Backends** - Combine multiple backends for better quality
5. **Local Models** - Support for local LLMs (Ollama, llama.cpp)
6. **Embedding Support** - Add sentence-transformers for local embeddings
7. **Batch Extraction** - Process multiple turns at once
8. **Fine-tuning** - Support for fine-tuned extraction models

## Design Decisions

### Why Three Backends?

1. **Mock** - Essential for testing and development without API costs
2. **OpenAI** - Industry standard, widely available, good balance of cost/quality
3. **Anthropic** - Excellent for complex extraction, tool calling is more reliable than JSON parsing

### Why Tool Calling for Anthropic?

Claude's tool calling provides:
- Structured schema validation
- More reliable than JSON parsing
- Better error messages
- Native support for complex types

### Why Mock Embeddings for Anthropic?

Anthropic doesn't provide an embeddings API. For production use:
- Use OpenAI embeddings
- Use sentence-transformers locally
- Use a dedicated embedding service

## API Reference

### get_backend(name: str, **kwargs) -> LLMBackend

Factory function to create backends.

**Parameters:**
- `name` (str): Backend name - "mock", "openai", or "anthropic"
- `**kwargs`: Additional arguments passed to backend constructor

**Returns:**
- `LLMBackend`: Backend instance

**Raises:**
- `ValueError`: If backend name is not recognized

**Example:**
```python
backend = get_backend("openai", model="gpt-4o-mini", api_key="sk-...")
```

### OpenAIBackend.__init__()

**Parameters:**
- `model` (str): Model name (default: "gpt-4o-mini")
- `embedding_model` (str): Embedding model (default: "text-embedding-3-small")
- `api_key` (str, optional): API key (defaults to OPENAI_API_KEY env var)
- `api_base` (str, optional): Custom API base URL

### AnthropicBackend.__init__()

**Parameters:**
- `model` (str): Model name (default: "claude-3-5-haiku-latest")
- `api_key` (str, optional): API key (defaults to ANTHROPIC_API_KEY env var)

## License

MIT License - See LICENSE file for details.
