"""Reranker backends for document ranking."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import os


class RerankerBackend(ABC):
    """Abstract base class for reranker backends."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional limit on number of results to return (None = all)

        Returns:
            List of (original_index, score) tuples sorted by score descending
        """
        pass


class MockRerankerBackend(RerankerBackend):
    """Mock reranker backend for testing without model loading."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate deterministic mock reranking scores using simple heuristics.

        Uses a combination of:
        - Query term overlap
        - Document length
        - Hash-based randomization for consistency

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional limit on number of results to return

        Returns:
            List of (original_index, score) tuples sorted by score descending
        """
        import hashlib

        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            doc_terms = set(doc_lower.split())

            # Calculate overlap score (0-1)
            if query_terms:
                overlap = len(query_terms & doc_terms) / len(query_terms)
            else:
                overlap = 0.0

            # Length penalty (prefer medium-length documents)
            doc_len = len(doc)
            length_score = 1.0 / (1.0 + abs(doc_len - 200) / 200.0)

            # Add deterministic randomization based on hash
            hash_input = f"{query}|{doc}".encode()
            hash_bytes = hashlib.md5(hash_input).digest()
            hash_score = hash_bytes[0] / 255.0 * 0.2  # 0-0.2 range

            # Combine scores (overlap is most important)
            score = overlap * 0.6 + length_score * 0.2 + hash_score

            results.append((idx, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]

        return results


class APIRerankerBackend(RerankerBackend):
    """Reranker backend using SiliconFlow API with BAAI/bge-reranker models."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Initialize API reranker backend.

        Args:
            model: Model name (defaults to RERANKER_MODEL env var or BAAI/bge-reranker-v2-m3)
            api_key: API key (defaults to EMBEDDING_API_KEY env var)
            api_base: API base URL (defaults to EMBEDDING_API_BASE env var or https://api.siliconflow.cn/v1)
            timeout: Request timeout in seconds
        """
        self.model = model or os.environ.get(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3"
        )
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY")
        self.api_base = api_base or os.environ.get(
            "EMBEDDING_API_BASE",
            "https://api.siliconflow.cn/v1"
        )
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "API key required. Set EMBEDDING_API_KEY environment variable or pass api_key parameter."
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using the SiliconFlow reranker API.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional limit on number of results to return

        Returns:
            List of (original_index, score) tuples sorted by score descending
        """
        import requests
        import time
        import random

        if not documents:
            return []

        # Filter out empty strings to avoid 400 errors
        filtered_docs = [(i, doc) for i, doc in enumerate(documents) if doc.strip()]
        if not filtered_docs:
            # All documents are empty
            return []

        filtered_indices, filtered_texts = zip(*filtered_docs)

        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                # Prepare request payload
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": list(filtered_texts),
                }

                # Add top_k if specified
                if top_k is not None:
                    payload["top_n"] = top_k

                # Make API request
                response = requests.post(
                    f"{self.api_base}/rerank",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # Extract results
                # API returns: {"results": [{"index": int, "relevance_score": float}, ...]}
                results = []
                for item in data.get("results", []):
                    idx = item["index"]
                    score = item["relevance_score"]
                    # Map back to original document indices
                    original_idx = filtered_indices[idx]
                    results.append((original_idx, float(score)))

                # Results should already be sorted by score, but ensure it
                results.sort(key=lambda x: x[1], reverse=True)

                return results

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else None

                # Don't retry on 400 errors (bad request)
                if status_code == 400:
                    print(f"Reranker API error (400 - bad request): {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"Response body: {e.response.text}")
                    # Fallback to simple ranking
                    return self._fallback_rerank(query, documents, top_k)

                # Retry on 5xx and 429 errors
                if status_code in (429, 500, 502, 503, 504):
                    if attempt < max_retries:
                        delay = (2 ** attempt) * (1 + random.random() * 0.25)
                        print(f"Reranker API error ({status_code}): {e}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue

                # Last attempt or non-retryable error
                print(f"Reranker API HTTP error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}")
                    print(f"Response body: {e.response.text}")
                # Fallback to simple ranking
                return self._fallback_rerank(query, documents, top_k)

            except requests.exceptions.Timeout as e:
                if attempt < max_retries:
                    delay = (2 ** attempt) * (1 + random.random() * 0.25)
                    print(f"Reranker API timeout error: {e}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                print(f"Reranker API timeout error: {e}")
                return self._fallback_rerank(query, documents, top_k)

            except requests.exceptions.RequestException as e:
                # Don't retry on other request errors
                print(f"Reranker API request error: {e}")
                return self._fallback_rerank(query, documents, top_k)

            except Exception as e:
                # Don't retry on unexpected errors
                print(f"Reranker API unexpected error: {e}")
                return self._fallback_rerank(query, documents, top_k)

        # Should not reach here, but fallback just in case
        return self._fallback_rerank(query, documents, top_k)

    def _fallback_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Fallback reranking using simple term overlap scoring.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional limit on number of results

        Returns:
            List of (original_index, score) tuples sorted by score descending
        """
        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            doc_terms = set(doc_lower.split())

            # Simple overlap-based scoring
            if query_terms:
                overlap = len(query_terms & doc_terms) / len(query_terms)
            else:
                overlap = 0.0

            # Bonus for exact query substring match
            if query_lower in doc_lower:
                overlap += 0.5

            results.append((idx, float(overlap)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]

        return results


class Reranker:
    """
    Main reranker class providing a unified interface.

    This class wraps reranker backends and provides a simple API for reranking.
    """

    def __init__(
        self,
        backend: Optional[RerankerBackend] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        Initialize reranker.

        Args:
            backend: Reranker backend instance (if None, creates APIRerankerBackend)
            model: Model name for API backend
            api_key: API key for API backend
            api_base: API base URL for API backend
            use_mock: If True, use MockRerankerBackend instead of API
        """
        if backend is not None:
            self.backend = backend
        elif use_mock:
            self.backend = MockRerankerBackend()
        else:
            self.backend = APIRerankerBackend(
                model=model,
                api_key=api_key,
                api_base=api_base,
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional limit on number of results to return

        Returns:
            List of (original_index, score) tuples sorted by score descending

        Example:
            >>> reranker = Reranker()
            >>> docs = ["Python is great", "Java is popular", "Python programming"]
            >>> results = reranker.rerank("Python coding", docs, top_k=2)
            >>> # Returns: [(2, 0.95), (0, 0.87)] - indices and scores
        """
        return self.backend.rerank(query, documents, top_k)

    def rerank_with_documents(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, int]]:
        """
        Rerank documents and return with document text.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional limit on number of results to return

        Returns:
            List of (document_text, score, original_index) tuples sorted by score descending

        Example:
            >>> reranker = Reranker()
            >>> docs = ["Python is great", "Java is popular", "Python programming"]
            >>> results = reranker.rerank_with_documents("Python coding", docs, top_k=2)
            >>> # Returns: [("Python programming", 0.95, 2), ("Python is great", 0.87, 0)]
        """
        ranked_indices = self.backend.rerank(query, documents, top_k)
        return [
            (documents[idx], score, idx)
            for idx, score in ranked_indices
        ]


# Convenience function for quick usage
def rerank_documents(
    query: str,
    documents: List[str],
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    use_mock: bool = False,
) -> List[Tuple[int, float]]:
    """
    Convenience function to rerank documents without creating a Reranker instance.

    Args:
        query: The search query
        documents: List of documents to rerank
        top_k: Optional limit on number of results to return
        model: Model name for API backend
        api_key: API key for API backend
        api_base: API base URL for API backend
        use_mock: If True, use MockRerankerBackend instead of API

    Returns:
        List of (original_index, score) tuples sorted by score descending

    Example:
        >>> docs = ["Python is great", "Java is popular", "Python programming"]
        >>> results = rerank_documents("Python coding", docs, top_k=2)
        >>> # Returns: [(2, 0.95), (0, 0.87)]
    """
    reranker = Reranker(
        model=model,
        api_key=api_key,
        api_base=api_base,
        use_mock=use_mock,
    )
    return reranker.rerank(query, documents, top_k)
