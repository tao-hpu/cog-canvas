# Canvas.py Modifications Required for Embedding Support

## Summary
The `cogcanvas/embeddings.py` module has been successfully created with the following components:
- `EmbeddingBackend` abstract class
- `MockEmbeddingBackend` for testing (384-dim, deterministic, normalized)
- `SentenceTransformerBackend` with lazy loading (default: all-MiniLM-L6-v2)
- `cosine_similarity()` and `batch_cosine_similarity()` helper functions

## Required Changes to `cogcanvas/canvas.py`

### 1. Update Imports (Lines 3-20)

```python
from typing import List, Optional, Dict, Any, Union, Literal  # Add Literal
import json
import time
from pathlib import Path

from cogcanvas.models import (
    CanvasObject,
    ObjectType,
    ExtractionResult,
    RetrievalResult,
)
from cogcanvas.llm.base import LLMBackend, MockLLMBackend
from cogcanvas.embeddings import (  # ADD THESE LINES
    EmbeddingBackend,
    MockEmbeddingBackend,
    SentenceTransformerBackend,
    batch_cosine_similarity,
)
from cogcanvas.graph import CanvasGraph
```

### 2. Update __init__ Signature (Line 37-43)

```python
def __init__(
    self,
    extractor_model: str = "mock",
    embedding_model: str = "mock",
    storage_path: Optional[str] = None,
    llm_backend: Optional[LLMBackend] = None,
    embedding_backend: Optional[EmbeddingBackend] = None,  # ADD THIS LINE
):
```

### 3. Update __init__ Docstring (Line 44-52)

```python
"""
Initialize a new Canvas.

Args:
    extractor_model: Model to use for extraction (e.g., "gpt-4o-mini", "mock")
    embedding_model: Model for embeddings (e.g., "all-MiniLM-L6-v2", "mock")  # UPDATE
    storage_path: Path to persist canvas state (optional)
    llm_backend: Pre-configured LLM backend (overrides extractor_model)
    embedding_backend: Pre-configured embedding backend (overrides embedding_model)  # ADD
"""
```

### 4. Add _init_embedding_backend Method (After line 86)

```python
def _init_embedding_backend(
    self, embedding_backend: Optional[EmbeddingBackend]
) -> EmbeddingBackend:
    """Initialize the embedding backend based on configuration."""
    if embedding_backend is not None:
        return embedding_backend

    if self.embedding_model == "mock":
        return MockEmbeddingBackend()

    # Use SentenceTransformer for local models
    return SentenceTransformerBackend(model_name=self.embedding_model)
```

### 5. Initialize Embedding Backend in __init__ (After line 58)

```python
# Initialize LLM backend
self._backend = self._init_backend(llm_backend)

# Initialize embedding backend  # ADD THESE LINES
self._embedding_backend = self._init_embedding_backend(embedding_backend)
```

### 6. Add Embedding Computation in extract() (After line 117)

```python
# Use LLM backend for extraction (real or mock)
existing = list(self._objects.values())
objects = self._backend.extract_objects(user, assistant, existing)

# Compute embeddings for extracted objects  # ADD THESE LINES
if objects:
    texts = [obj.content for obj in objects]
    embeddings = self._embedding_backend.embed_batch(texts)
    for obj, embedding in zip(objects, embeddings):
        obj.embedding = embedding

# Store extracted objects
for obj in objects:
    obj.turn_id = self._turn_counter
    self._objects[obj.id] = obj
    self._graph.add_node(obj)
```

### 7. Update retrieve() Signature (Line 134-144)

```python
def retrieve(
    self,
    query: str,
    top_k: int = 5,
    obj_type: Optional[ObjectType] = None,
    method: Literal["semantic", "keyword"] = "semantic",  # ADD THIS LINE
    include_related: bool = False,
):
    """
    Retrieve relevant canvas objects for a query.

    Args:
        query: The search query
        top_k: Maximum number of objects to return
        obj_type: Filter by object type (optional)
        method: Retrieval method ("semantic" or "keyword")  # ADD THIS LINE
        include_related: If True, include 1-hop related objects

    Returns:
        RetrievalResult with matching objects and scores
    """
```

### 8. Replace Retrieval Logic in retrieve() (Lines 164-171)

REPLACE:
```python
# TODO: Implement actual semantic retrieval
# For now, use simple keyword matching
scored = []
query_lower = query.lower()
for obj in candidates:
    score = self._simple_match_score(query_lower, obj.content.lower())
    if score > 0:
        scored.append((obj, score))
```

WITH:
```python
# Choose retrieval method
if method == "semantic":
    scored = self._semantic_retrieve(query, candidates)
else:
    scored = self._keyword_retrieve(query, candidates)
```

### 9. Update add() Method (Line 236-241)

```python
def add(self, obj: CanvasObject, compute_embedding: bool = True) -> None:  # UPDATE
    """
    Manually add an object to the canvas.

    Args:
        obj: CanvasObject to add
        compute_embedding: Whether to compute embedding if missing  # ADD
    """
    # Compute embedding if needed  # ADD THESE LINES
    if compute_embedding and obj.embedding is None and obj.content:
        obj.embedding = self._embedding_backend.embed(obj.content)

    self._objects[obj.id] = obj
    self._graph.add_node(obj)
    if self.storage_path:
        self._save()
```

### 10. Add Helper Methods (Before _mock_extract, around line 496)

```python
def _semantic_retrieve(
    self,
    query: str,
    candidates: List[CanvasObject],
) -> List[tuple[CanvasObject, float]]:
    """
    Retrieve using semantic similarity (cosine similarity of embeddings).

    Args:
        query: Search query
        candidates: Candidate objects to score

    Returns:
        List of (object, score) tuples
    """
    # Compute query embedding
    query_embedding = self._embedding_backend.embed(query)

    # Filter candidates with embeddings
    valid_candidates = [obj for obj in candidates if obj.embedding is not None]

    if not valid_candidates:
        return []

    # Compute similarities
    embeddings = [obj.embedding for obj in valid_candidates]
    similarities = batch_cosine_similarity(query_embedding, embeddings)

    return list(zip(valid_candidates, similarities))

def _keyword_retrieve(
    self,
    query: str,
    candidates: List[CanvasObject],
) -> List[tuple[CanvasObject, float]]:
    """
    Retrieve using keyword matching.

    Args:
        query: Search query
        candidates: Candidate objects to score

    Returns:
        List of (object, score) tuples
    """
    scored = []
    query_lower = query.lower()
    for obj in candidates:
        score = self._simple_match_score(query_lower, obj.content.lower())
        if score > 0:
            scored.append((obj, score))
    return scored

def _infer_relations(self, objects: List[CanvasObject]) -> None:
    """Infer relationships between newly extracted objects."""
    # TODO: Implement relationship inference
    pass
```

## Verification

After making these changes:

1. The embedding tests should pass: `pytest tests/test_embeddings.py -v`
2. The canvas tests should pass: `pytest tests/test_canvas.py -v`
3. Semantic retrieval should work with `method="semantic"`
4. Keyword retrieval should work with `method="keyword"`

## Files Modified

- `/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/cogcanvas/embeddings.py` ✓ Created
- `/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/cogcanvas/canvas.py` - Needs manual modification
- `/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/pyproject.toml` ✓ Already has sentence-transformers
- `/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas/tests/test_embeddings.py` ✓ Already exists

## Usage Example

```python
from cogcanvas import Canvas, CanvasObject, ObjectType

# Create canvas with semantic retrieval
canvas = Canvas(
    embedding_model="all-MiniLM-L6-v2",  # or "mock" for testing
)

# Extract objects (embeddings computed automatically)
canvas.extract(
    user="Let's use PostgreSQL for the database",
    assistant="Great choice! I'll help you set it up."
)

# Semantic retrieval (default)
results = canvas.retrieve(
    query="What database did we choose?",
    method="semantic",  # Uses cosine similarity
    top_k=5
)

# Keyword retrieval (fallback)
results = canvas.retrieve(
    query="database PostgreSQL",
    method="keyword",  # Uses keyword matching
    top_k=5
)

# Manual object with embedding
obj = CanvasObject(
    type=ObjectType.DECISION,
    content="Use Redis for caching layer"
)
canvas.add(obj, compute_embedding=True)  # Embedding computed automatically
```
