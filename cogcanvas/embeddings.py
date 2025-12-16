"""Embedding backends for vector similarity search."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        pass


class MockEmbeddingBackend(EmbeddingBackend):
    """Mock embedding backend for testing without model loading."""

    def __init__(self, dimension: int = 384):
        """
        Initialize mock backend.

        Args:
            dimension: Dimension of mock embeddings (default 384 to match all-MiniLM-L6-v2)
        """
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        """Generate deterministic mock embedding using hash."""
        import hashlib

        hash_bytes = hashlib.md5(text.encode()).digest()
        embedding = []
        for i in range(self._dimension):
            byte_idx = i % 16
            embedding.append((hash_bytes[byte_idx] / 255.0) - 0.5)  # Center around 0

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch."""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class APIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend using OpenAI-compatible API (e.g., BGE via SiliconFlow)."""

    def __init__(
        self,
        model: str = "bge-large-zh-v1.5",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize API embedding backend.

        Args:
            model: Model name (e.g., "bge-large-zh-v1.5")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            api_base: API base URL (defaults to OPENAI_API_BASE env var)
            batch_size: Max texts per batch request
        """
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.batch_size = batch_size
        self._dimension = None

        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key.")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text via API."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts via API."""
        import requests

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = requests.post(
                    f"{self.api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": batch,
                        "model": self.model,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                # Extract embeddings in order
                batch_embeddings = sorted(data["data"], key=lambda x: x["index"])
                all_embeddings.extend([item["embedding"] for item in batch_embeddings])

                # Cache dimension
                if self._dimension is None and all_embeddings:
                    self._dimension = len(all_embeddings[0])

            except requests.exceptions.HTTPError as e:
                # Log response body for debugging
                print(f"API embedding error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response body: {e.response.text}")
                # Return zero vectors as fallback
                dim = self._dimension or 1024
                all_embeddings.extend([[0.0] * dim for _ in batch])
            except Exception as e:
                print(f"API embedding error: {e}")
                # Return zero vectors as fallback
                dim = self._dimension or 1024
                all_embeddings.extend([[0.0] * dim for _ in batch])

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimension is None:
            # Make a test call to get dimension
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score in [-1, 1]
    """
    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query_vec: List[float],
    vectors: List[List[float]],
) -> List[float]:
    """
    Calculate cosine similarity between query and multiple vectors efficiently.

    Args:
        query_vec: Query vector
        vectors: List of vectors to compare against

    Returns:
        List of cosine similarity scores
    """
    if not vectors:
        return []

    # Ensure query is 1D
    query = np.array(query_vec).flatten()
    matrix = np.array(vectors)

    # Handle single vector case - ensure matrix is 2D
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return [0.0] * len(vectors)
    query_normalized = query / query_norm

    # Normalize vectors
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    matrix_normalized = matrix / norms

    # Compute similarities: (N, D) @ (D,) -> (N,)
    similarities = np.dot(matrix_normalized, query_normalized)

    # Ensure output is a flat list
    if similarities.ndim == 0:
        return [float(similarities)]
    return similarities.flatten().tolist()
