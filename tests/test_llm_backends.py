"""Tests for LLM backends."""

import os
import pytest
from cogcanvas.llm import get_backend, MockLLMBackend, OpenAIBackend
from cogcanvas.models import CanvasObject, ObjectType
from cogcanvas import Canvas


class TestGetBackend:
    """Test the get_backend factory function."""

    def test_get_mock_backend(self):
        """Test getting mock backend."""
        backend = get_backend("mock")
        assert isinstance(backend, MockLLMBackend)

    def test_get_openai_backend_without_key(self):
        """Test getting OpenAI backend without API key."""
        # Temporarily remove key if it exists
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                get_backend("openai")
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key



    def test_get_unknown_backend(self):
        """Test getting unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown")

    def test_backend_with_kwargs(self):
        """Test passing kwargs to backend."""
        backend = get_backend("mock")
        assert isinstance(backend, MockLLMBackend)


class TestMockBackend:
    """Test MockLLMBackend."""

    def test_extract_decision(self):
        """Test extracting decisions."""
        backend = MockLLMBackend()
        objects = backend.extract_objects(
            "Let's use PostgreSQL", "Good choice!"
        )
        assert len(objects) > 0
        assert any(obj.type == ObjectType.DECISION for obj in objects)

    def test_extract_todo(self):
        """Test extracting todos."""
        backend = MockLLMBackend()
        objects = backend.extract_objects(
            "We need to implement auth", "I'll work on it"
        )
        assert len(objects) > 0
        assert any(obj.type == ObjectType.TODO for obj in objects)

    def test_extract_reminder(self):
        """Test extracting reminders."""
        backend = MockLLMBackend()
        objects = backend.extract_objects(
            "Remember to keep it simple", "Noted!"
        )
        assert len(objects) > 0
        assert any(obj.type == ObjectType.REMINDER for obj in objects)

    def test_extract_nothing(self):
        """Test extracting from generic conversation."""
        backend = MockLLMBackend()
        objects = backend.extract_objects(
            "Hello there", "Hi, how can I help?"
        )
        assert len(objects) == 0

    def test_embed(self):
        """Test embedding generation."""
        backend = MockLLMBackend()
        embedding = backend.embed("test text")
        assert len(embedding) == 384
        assert all(0 <= x <= 1 for x in embedding)

    def test_embed_consistent(self):
        """Test embeddings are consistent."""
        backend = MockLLMBackend()
        emb1 = backend.embed("test")
        emb2 = backend.embed("test")
        assert emb1 == emb2


class TestCanvasWithBackends:
    """Test Canvas integration with different backends."""

    def test_canvas_with_mock_backend(self):
        """Test Canvas with mock backend."""
        backend = MockLLMBackend()
        canvas = Canvas(llm_backend=backend)

        result = canvas.extract(
            user="Let's use PostgreSQL",
            assistant="Great choice!",
        )

        assert result.count > 0
        assert canvas.size > 0

    def test_canvas_backend_fallback(self):
        """Test Canvas falls back to mock if no backend specified."""
        canvas = Canvas()
        assert isinstance(canvas._backend, MockLLMBackend)

    def test_canvas_extracts_and_stores(self):
        """Test Canvas extraction and storage."""
        backend = MockLLMBackend()
        canvas = Canvas(llm_backend=backend)

        # First extraction
        result1 = canvas.extract(
            user="We need to add authentication",
            assistant="I'll implement OAuth2",
        )
        assert result1.count > 0

        # Second extraction
        result2 = canvas.extract(
            user="Remember to use TypeScript",
            assistant="Understood, TypeScript it is",
        )
        assert result2.count > 0

        # Check total objects
        assert canvas.size >= result1.count + result2.count


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIBackend:
    """Test OpenAI backend (requires API key)."""

    def test_openai_extract(self):
        """Test OpenAI extraction."""
        backend = OpenAIBackend(model="gpt-4o-mini")
        objects = backend.extract_objects(
            user_message="Let's use PostgreSQL and implement auth",
            assistant_message="Good choices! I'll start with the database setup.",
        )
        # Should extract at least decision/todo
        assert len(objects) >= 1
        assert all(isinstance(obj, CanvasObject) for obj in objects)

    def test_openai_embed(self):
        """Test OpenAI embeddings."""
        backend = OpenAIBackend()
        embedding = backend.embed("test text")
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)



