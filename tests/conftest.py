"""Shared pytest fixtures for CogCanvas tests."""

import pytest
from cogcanvas import Canvas, CanvasObject, ObjectType
from cogcanvas.llm.base import MockLLMBackend
from cogcanvas.embeddings import MockEmbeddingBackend


@pytest.fixture
def sample_decision():
    """Create a sample decision object."""
    return CanvasObject(
        type=ObjectType.DECISION,
        content="Use PostgreSQL for the database",
        context="Team decided after evaluating options",
        confidence=0.9,
        turn_id=1,
    )


@pytest.fixture
def sample_todo():
    """Create a sample TODO object."""
    return CanvasObject(
        type=ObjectType.TODO,
        content="Implement user authentication",
        context="Required for MVP launch",
        confidence=0.95,
        turn_id=2,
    )


@pytest.fixture
def sample_key_fact():
    """Create a sample key fact object."""
    return CanvasObject(
        type=ObjectType.KEY_FACT,
        content="API rate limit is 100 requests per minute",
        context="From API documentation",
        confidence=1.0,
        turn_id=3,
    )


@pytest.fixture
def sample_reminder():
    """Create a sample reminder object."""
    return CanvasObject(
        type=ObjectType.REMINDER,
        content="User prefers TypeScript over JavaScript",
        context="Mentioned in initial discussion",
        confidence=0.85,
        turn_id=4,
    )


@pytest.fixture
def sample_insight():
    """Create a sample insight object."""
    return CanvasObject(
        type=ObjectType.INSIGHT,
        content="The database query is the main performance bottleneck",
        context="Discovered during profiling",
        confidence=0.8,
        turn_id=5,
    )


@pytest.fixture
def sample_objects(
    sample_decision, sample_todo, sample_key_fact, sample_reminder, sample_insight
):
    """Return a list of sample objects covering all types."""
    return [
        sample_decision,
        sample_todo,
        sample_key_fact,
        sample_reminder,
        sample_insight,
    ]


@pytest.fixture
def empty_canvas():
    """Create an empty canvas with mock backend."""
    return Canvas(extractor_model="mock")


@pytest.fixture
def populated_canvas(empty_canvas, sample_objects):
    """Create a canvas populated with sample objects."""
    for obj in sample_objects:
        empty_canvas.add(obj)
    return empty_canvas


@pytest.fixture
def mock_llm_backend():
    """Create a mock LLM backend."""
    return MockLLMBackend()


@pytest.fixture
def mock_embedding_backend():
    """Create a mock embedding backend."""
    return MockEmbeddingBackend(dimension=384)


@pytest.fixture
def test_texts():
    """Sample texts for embedding tests."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Data structures and algorithms are fundamental",
    ]
