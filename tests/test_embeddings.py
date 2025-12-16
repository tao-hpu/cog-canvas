"""Tests for embedding backends."""

import pytest
import numpy as np
from cogcanvas.embeddings import (
    MockEmbeddingBackend,
    cosine_similarity,
    batch_cosine_similarity,
)


class TestMockEmbeddingBackend:
    """Test suite for MockEmbeddingBackend."""

    def test_initialization_default(self):
        """Test default initialization."""
        backend = MockEmbeddingBackend()
        assert backend.dimension == 384

    def test_initialization_custom_dimension(self):
        """Test initialization with custom dimension."""
        backend = MockEmbeddingBackend(dimension=512)
        assert backend.dimension == 512

    def test_embed_single_text(self, mock_embedding_backend):
        """Test embedding a single text."""
        text = "Hello, world!"
        embedding = mock_embedding_backend.embed(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_deterministic(self, mock_embedding_backend):
        """Test that same text produces same embedding."""
        text = "Test text"
        emb1 = mock_embedding_backend.embed(text)
        emb2 = mock_embedding_backend.embed(text)

        assert emb1 == emb2

    def test_embed_different_texts(self, mock_embedding_backend):
        """Test that different texts produce different embeddings."""
        text1 = "Hello"
        text2 = "World"

        emb1 = mock_embedding_backend.embed(text1)
        emb2 = mock_embedding_backend.embed(text2)

        assert emb1 != emb2

    def test_embed_normalized(self, mock_embedding_backend):
        """Test that embeddings are normalized to unit vectors."""
        text = "Normalize this text"
        embedding = mock_embedding_backend.embed(text)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6  # Should be unit vector

    def test_embed_empty_string(self, mock_embedding_backend):
        """Test embedding empty string."""
        embedding = mock_embedding_backend.embed("")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_long_text(self, mock_embedding_backend):
        """Test embedding very long text."""
        text = "word " * 1000
        embedding = mock_embedding_backend.embed(text)
        assert len(embedding) == 384

    def test_embed_special_characters(self, mock_embedding_backend):
        """Test embedding text with special characters."""
        text = "Hello! @#$%^&*() 你好 🚀"
        embedding = mock_embedding_backend.embed(text)
        assert len(embedding) == 384

    def test_embed_batch_single(self, mock_embedding_backend):
        """Test batch embedding with single text."""
        texts = ["Hello"]
        embeddings = mock_embedding_backend.embed_batch(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_embed_batch_multiple(self, mock_embedding_backend, test_texts):
        """Test batch embedding with multiple texts."""
        embeddings = mock_embedding_backend.embed_batch(test_texts)

        assert len(embeddings) == len(test_texts)
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embed_batch_consistency(self, mock_embedding_backend):
        """Test that batch embedding matches individual embeddings."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Get embeddings individually
        individual = [mock_embedding_backend.embed(text) for text in texts]

        # Get embeddings in batch
        batch = mock_embedding_backend.embed_batch(texts)

        # Should be identical
        for ind, bat in zip(individual, batch):
            assert ind == bat

    def test_embed_batch_empty_list(self, mock_embedding_backend):
        """Test batch embedding with empty list."""
        embeddings = mock_embedding_backend.embed_batch([])
        assert embeddings == []

    @pytest.mark.parametrize("dimension", [128, 256, 384, 512, 768, 1024])
    def test_various_dimensions(self, dimension):
        """Test different embedding dimensions."""
        backend = MockEmbeddingBackend(dimension=dimension)
        embedding = backend.embed("Test")

        assert len(embedding) == dimension
        assert backend.dimension == dimension

    def test_dimension_property(self):
        """Test dimension property."""
        backend = MockEmbeddingBackend(dimension=256)
        assert backend.dimension == 256

    @pytest.mark.parametrize(
        "text",
        [
            "Short",
            "Medium length sentence here",
            "A much longer piece of text that contains multiple sentences and words.",
            "With\nnewlines\nand\ttabs",
        ],
    )
    def test_various_text_lengths(self, mock_embedding_backend, text):
        """Test embedding texts of various lengths."""
        embedding = mock_embedding_backend.embed(text)
        assert len(embedding) == 384
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6


class TestCosineSimilarity:
    """Test suite for cosine similarity functions."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0, 4.0]
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_similar_vectors(self):
        """Test similarity of similar vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 2.9]
        similarity = cosine_similarity(vec1, vec2)
        assert 0.9 < similarity < 1.0

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_normalized_vectors(self):
        """Test with pre-normalized vectors."""
        vec1 = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        vec2 = [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6

    def test_different_magnitudes(self):
        """Test that magnitude doesn't affect cosine similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [10.0, 20.0, 30.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6

    @pytest.mark.parametrize(
        "vec1,vec2,expected",
        [
            ([1, 0, 0], [1, 0, 0], 1.0),
            ([1, 0, 0], [0, 1, 0], 0.0),
            ([1, 1], [1, 1], 1.0),
            ([1, 2, 3], [4, 5, 6], None),  # Just check it computes
        ],
    )
    def test_parametrized_cases(self, vec1, vec2, expected):
        """Test various vector pairs."""
        similarity = cosine_similarity(vec1, vec2)
        if expected is not None:
            assert abs(similarity - expected) < 1e-6
        else:
            assert -1.0 <= similarity <= 1.0


class TestBatchCosineSimilarity:
    """Test suite for batch cosine similarity."""

    def test_single_vector(self):
        """Test batch similarity with single vector."""
        query = [1.0, 2.0, 3.0]
        vectors = [[1.0, 2.0, 3.0]]
        similarities = batch_cosine_similarity(query, vectors)

        assert len(similarities) == 1
        assert abs(similarities[0] - 1.0) < 1e-6

    def test_multiple_vectors(self):
        """Test batch similarity with multiple vectors."""
        query = [1.0, 0.0, 0.0]
        vectors = [
            [1.0, 0.0, 0.0],  # Same direction
            [0.0, 1.0, 0.0],  # Orthogonal
            [-1.0, 0.0, 0.0],  # Opposite
        ]
        similarities = batch_cosine_similarity(query, vectors)

        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6
        assert abs(similarities[1] - 0.0) < 1e-6
        assert abs(similarities[2] - (-1.0)) < 1e-6

    def test_empty_vectors_list(self):
        """Test with empty vectors list."""
        query = [1.0, 2.0, 3.0]
        similarities = batch_cosine_similarity(query, [])
        assert similarities == []

    def test_zero_query_vector(self):
        """Test with zero query vector."""
        query = [0.0, 0.0, 0.0]
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        similarities = batch_cosine_similarity(query, vectors)

        assert len(similarities) == 2
        assert all(s == 0.0 for s in similarities)

    def test_zero_in_vectors(self):
        """Test with zero vector in list."""
        query = [1.0, 2.0, 3.0]
        vectors = [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],  # Zero vector
            [4.0, 5.0, 6.0],
        ]
        similarities = batch_cosine_similarity(query, vectors)

        assert len(similarities) == 3
        # Zero vector should have 0 similarity but not crash
        assert isinstance(similarities[1], float)

    def test_consistency_with_single(self):
        """Test that batch results match individual computations."""
        query = [1.0, 2.0, 3.0, 4.0]
        vectors = [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]

        batch_sims = batch_cosine_similarity(query, vectors)
        individual_sims = [cosine_similarity(query, v) for v in vectors]

        for batch, individual in zip(batch_sims, individual_sims):
            assert abs(batch - individual) < 1e-6

    def test_large_batch(self):
        """Test with large batch of vectors."""
        query = [1.0] * 100
        vectors = [[1.0] * 100 for _ in range(1000)]
        similarities = batch_cosine_similarity(query, vectors)

        assert len(similarities) == 1000
        assert all(abs(s - 1.0) < 1e-6 for s in similarities)

    def test_with_mock_embeddings(self, mock_embedding_backend, test_texts):
        """Test batch similarity with real mock embeddings."""
        # Generate embeddings
        embeddings = mock_embedding_backend.embed_batch(test_texts)

        # Compute similarity of first text to all others
        query_emb = embeddings[0]
        similarities = batch_cosine_similarity(query_emb, embeddings)

        assert len(similarities) == len(test_texts)
        # First should be identical to itself
        assert abs(similarities[0] - 1.0) < 1e-6
        # All should be in valid range
        assert all(-1.0 <= s <= 1.0 for s in similarities)

    @pytest.mark.parametrize("num_vectors", [1, 5, 10, 50, 100])
    def test_various_batch_sizes(self, num_vectors):
        """Test with various batch sizes."""
        query = [1.0, 2.0, 3.0]
        vectors = [[1.0, 2.0, 3.0] for _ in range(num_vectors)]
        similarities = batch_cosine_similarity(query, vectors)

        assert len(similarities) == num_vectors
        assert all(abs(s - 1.0) < 1e-6 for s in similarities)

    def test_similarity_symmetry(self):
        """Test that similarity is symmetric."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]

        sim1 = batch_cosine_similarity(vec1, [vec2])[0]
        sim2 = batch_cosine_similarity(vec2, [vec1])[0]

        assert abs(sim1 - sim2) < 1e-6

    def test_returns_list_of_floats(self):
        """Test that result is list of floats."""
        query = [1.0, 2.0]
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        similarities = batch_cosine_similarity(query, vectors)

        assert isinstance(similarities, list)
        assert all(isinstance(s, float) for s in similarities)
