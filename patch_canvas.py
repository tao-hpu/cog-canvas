"""Patch script to add embedding support to canvas.py"""
import re

# Read the current file
with open('cogcanvas/canvas.py', 'r') as f:
    content = f.read()

# Track modifications
modified = False

# 1. Add Literal to imports
if 'Literal' not in content:
    content = re.sub(
        r'from typing import List, Optional, Dict, Any, Union',
        'from typing import List, Optional, Dict, Any, Union, Literal',
        content
    )
    modified = True

# 2. Add embeddings imports
if 'from cogcanvas.embeddings import' not in content:
    content = re.sub(
        r'(from cogcanvas\.llm\.base import LLMBackend, MockLLMBackend)',
        r'\1\nfrom cogcanvas.embeddings import (\n    EmbeddingBackend,\n    MockEmbeddingBackend,\n    SentenceTransformerBackend,\n    batch_cosine_similarity,\n)',
        content
    )
    modified = True

# 3. Add embedding_backend parameter
if 'embedding_backend: Optional[EmbeddingBackend]' not in content:
    content = re.sub(
        r'(llm_backend: Optional\[LLMBackend\] = None,)\n(\s+\))',
        r'\1\n        embedding_backend: Optional[EmbeddingBackend] = None,\n\2',
        content
    )
    modified = True

# 4. Update docstring
content = re.sub(
    r'embedding_model: Model for embeddings \(e\.g\., "text-embedding-3-small"\)',
    'embedding_model: Model for embeddings (e.g., "all-MiniLM-L6-v2", "mock")',
    content
)

if 'embedding_backend: Pre-configured embedding backend' not in content:
    content = re.sub(
        r'(llm_backend: Pre-configured LLM backend \(overrides extractor_model\))\n(\s+""")',
        r'\1\n            embedding_backend: Pre-configured embedding backend (overrides embedding_model)\n\2',
        content
    )
    modified = True

# 5. Add _init_embedding_backend method
if '_init_embedding_backend' not in content:
    init_embedding = '''
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
'''
    content = re.sub(
        r'(return MockLLMBackend\(\)\n)(\s+# ===)',
        r'\1' + init_embedding + '\n\2',
        content
    )
    modified = True

# 6. Initialize embedding backend
if 'self._embedding_backend = self._init_embedding_backend' not in content:
    content = re.sub(
        r'(self\._backend = self\._init_backend\(llm_backend\))',
        r'\1\n\n        # Initialize embedding backend\n        self._embedding_backend = self._init_embedding_backend(embedding_backend)',
        content
    )
    modified = True

# 7. Add embedding computation in extract()
if 'Compute embeddings for extracted objects' not in content:
    embed_code = '''        # Compute embeddings for extracted objects
        if objects:
            texts = [obj.content for obj in objects]
            embeddings = self._embedding_backend.embed_batch(texts)
            for obj, embedding in zip(objects, embeddings):
                obj.embedding = embedding

'''
    content = re.sub(
        r'(objects = self\._backend\.extract_objects\(user, assistant, existing\))\n\n',
        r'\1\n\n' + embed_code,
        content
    )
    modified = True

# 8. Add method parameter to retrieve()
if 'method: Literal["semantic", "keyword"]' not in content:
    content = re.sub(
        r'(obj_type: Optional\[ObjectType\] = None,)\n(\s+include_related: bool = False,)',
        r'\1\n        method: Literal["semantic", "keyword"] = "semantic",\n\2',
        content
    )
    content = re.sub(
        r'(obj_type: Filter by object type \(optional\))\n(\s+include_related:)',
        r'\1\n            method: Retrieval method ("semantic" or "keyword")\n\2',
        content
    )
    modified = True

# 9. Replace retrieval logic
if '# Choose retrieval method' not in content:
    old_pattern = r'''# TODO: Implement actual semantic retrieval
        # For now, use simple keyword matching
        scored = \[\]
        query_lower = query\.lower\(\)
        for obj in candidates:
            score = self\._simple_match_score\(query_lower, obj\.content\.lower\(\)\)
            if score > 0:
                scored\.append\(\(obj, score\)\)'''

    new_code = '''        # Choose retrieval method
        if method == "semantic":
            scored = self._semantic_retrieve(query, candidates)
        else:
            scored = self._keyword_retrieve(query, candidates)'''

    content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE | re.DOTALL)
    modified = True

# 10. Update add() method signature
if ', compute_embedding: bool = True' not in content:
    content = re.sub(
        r'def add\(self, obj: CanvasObject\) -> None:\n        """Manually add an object to the canvas\."""',
        '''def add(self, obj: CanvasObject, compute_embedding: bool = True) -> None:
        """
        Manually add an object to the canvas.

        Args:
            obj: CanvasObject to add
            compute_embedding: Whether to compute embedding if missing
        """
        # Compute embedding if needed
        if compute_embedding and obj.embedding is None and obj.content:
            obj.embedding = self._embedding_backend.embed(obj.content)''',
        content
    )
    modified = True

# 11. Add helper methods
if '_semantic_retrieve' not in content:
    helpers = '''    def _semantic_retrieve(
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

'''
    content = re.sub(
        r'(\n    def _mock_extract)',
        '\n' + helpers + r'\1',
        content
    )
    modified = True

# 12. Add _infer_relations stub
if '_infer_relations' not in content:
    infer_stub = '''    def _infer_relations(self, objects: List[CanvasObject]) -> None:
        """Infer relationships between newly extracted objects."""
        # TODO: Implement relationship inference
        pass

'''
    content = re.sub(
        r'(\n    def _simple_match_score)',
        '\n' + infer_stub + r'\1',
        content
    )
    modified = True

if modified:
    with open('cogcanvas/canvas.py', 'w') as f:
        f.write(content)
    print("Successfully patched canvas.py with embedding support")
else:
    print("No modifications needed - file already has embedding support")
