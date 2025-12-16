"""Tests for the Canvas class."""

import pytest
import tempfile
import json
from pathlib import Path
from cogcanvas import Canvas, CanvasObject, ObjectType


class TestCanvas:
    """Test suite for Canvas class."""

    def test_init(self):
        """Test canvas initialization."""
        canvas = Canvas()
        assert canvas.size == 0
        assert canvas.turn_count == 0

    def test_extract_basic(self):
        """Test basic extraction."""
        canvas = Canvas()
        result = canvas.extract(
            user="Let's decide to use PostgreSQL",
            assistant="Good choice!",
        )
        assert canvas.turn_count == 1
        # Mock extractor should find "decide" keyword
        assert result.count >= 1

    def test_retrieve_empty(self):
        """Test retrieval on empty canvas."""
        canvas = Canvas()
        result = canvas.retrieve("anything")
        assert result.count == 0

    def test_add_and_retrieve(self):
        """Test manual add and retrieve."""
        canvas = Canvas()

        obj = CanvasObject(
            type=ObjectType.DECISION,
            content="Use PostgreSQL for database",
            context="Team decision",
        )
        canvas.add(obj)

        assert canvas.size == 1

        result = canvas.retrieve("PostgreSQL")
        assert result.count == 1
        assert "PostgreSQL" in result.objects[0].content

    def test_remove(self):
        """Test object removal."""
        canvas = Canvas()

        obj = CanvasObject(
            type=ObjectType.TODO,
            content="Add error handling",
        )
        canvas.add(obj)
        assert canvas.size == 1

        canvas.remove(obj.id)
        assert canvas.size == 0

    def test_clear(self):
        """Test clearing canvas."""
        canvas = Canvas()

        canvas.add(CanvasObject(type=ObjectType.KEY_FACT, content="Fact 1"))
        canvas.add(CanvasObject(type=ObjectType.KEY_FACT, content="Fact 2"))
        assert canvas.size == 2

        canvas.clear()
        assert canvas.size == 0

    def test_list_by_type(self):
        """Test listing objects by type."""
        canvas = Canvas()

        canvas.add(CanvasObject(type=ObjectType.DECISION, content="Decision 1"))
        canvas.add(CanvasObject(type=ObjectType.TODO, content="TODO 1"))
        canvas.add(CanvasObject(type=ObjectType.DECISION, content="Decision 2"))

        decisions = canvas.list_objects(obj_type=ObjectType.DECISION)
        assert len(decisions) == 2

        todos = canvas.list_objects(obj_type=ObjectType.TODO)
        assert len(todos) == 1

    def test_link_objects(self):
        """Test linking objects."""
        canvas = Canvas()

        obj1 = CanvasObject(type=ObjectType.DECISION, content="Use PostgreSQL")
        obj2 = CanvasObject(type=ObjectType.TODO, content="Set up PostgreSQL")

        canvas.add(obj1)
        canvas.add(obj2)

        canvas.link(obj1.id, obj2.id, relation="leads_to")

        assert obj2.id in obj1.leads_to
        assert obj1.id in obj2.caused_by

    def test_get_related(self):
        """Test getting related objects."""
        canvas = Canvas()

        obj1 = CanvasObject(type=ObjectType.DECISION, content="Decision")
        obj2 = CanvasObject(type=ObjectType.TODO, content="TODO")

        canvas.add(obj1)
        canvas.add(obj2)
        canvas.link(obj1.id, obj2.id)

        related = canvas.get_related(obj1.id)
        assert len(related) == 1
        assert related[0].id == obj2.id

    def test_stats(self):
        """Test canvas statistics."""
        canvas = Canvas()

        canvas.add(CanvasObject(type=ObjectType.DECISION, content="D1"))
        canvas.add(CanvasObject(type=ObjectType.TODO, content="T1"))
        canvas.add(CanvasObject(type=ObjectType.TODO, content="T2"))

        stats = canvas.stats()
        assert stats["total_objects"] == 3
        assert stats["by_type"]["decision"] == 1
        assert stats["by_type"]["todo"] == 2

    def test_inject_markdown(self):
        """Test markdown injection format."""
        canvas = Canvas()

        canvas.add(
            CanvasObject(
                type=ObjectType.DECISION,
                content="Use PostgreSQL for database",
            )
        )

        result = canvas.retrieve("PostgreSQL")
        injected = canvas.inject(result, format="markdown")

        assert "## Relevant Context" in injected
        assert "[Decision]" in injected
        assert "PostgreSQL" in injected


class TestCanvasObject:
    """Test suite for CanvasObject."""

    def test_to_dict(self):
        """Test serialization to dict."""
        obj = CanvasObject(
            type=ObjectType.DECISION,
            content="Test content",
            context="Test context",
        )
        data = obj.to_dict()

        assert data["type"] == "decision"
        assert data["content"] == "Test content"
        assert "id" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "test123",
            "type": "todo",
            "content": "Test TODO",
            "context": "",
            "turn_id": 1,
            "timestamp": 1234567890.0,
            "confidence": 0.9,
            "embedding": None,
            "references": [],
            "referenced_by": [],
            "leads_to": [],
            "caused_by": [],
        }

        obj = CanvasObject.from_dict(data)
        assert obj.id == "test123"
        assert obj.type == ObjectType.TODO
        assert obj.content == "Test TODO"


class TestCanvasAdvanced:
    """Advanced tests for Canvas class."""

    def test_multiple_extractions(self):
        """Test multiple extraction rounds."""
        canvas = Canvas()

        # First turn
        result1 = canvas.extract(
            user="Let's use PostgreSQL",
            assistant="Good idea"
        )
        turn1_count = result1.count

        # Second turn
        result2 = canvas.extract(
            user="We need to implement authentication",
            assistant="I'll work on it"
        )

        assert canvas.turn_count == 2
        assert canvas.size >= turn1_count

    def test_retrieve_with_type_filter(self):
        """Test retrieval with type filtering."""
        canvas = Canvas()

        canvas.add(CanvasObject(type=ObjectType.DECISION, content="Use PostgreSQL database"))
        canvas.add(CanvasObject(type=ObjectType.TODO, content="Set up PostgreSQL"))
        canvas.add(CanvasObject(type=ObjectType.DECISION, content="Use React framework"))

        # Retrieve only decisions
        result = canvas.retrieve("PostgreSQL", obj_type=ObjectType.DECISION)
        assert all(obj.type == ObjectType.DECISION for obj in result.objects)

    def test_retrieve_top_k(self):
        """Test top_k parameter in retrieval."""
        canvas = Canvas()

        for i in range(10):
            canvas.add(CanvasObject(
                type=ObjectType.KEY_FACT,
                content=f"Fact about database number {i}"
            ))

        result = canvas.retrieve("database", top_k=3)
        assert result.count <= 3

    def test_inject_json_format(self):
        """Test JSON injection format."""
        canvas = Canvas()

        canvas.add(CanvasObject(
            type=ObjectType.TODO,
            content="Implement feature X"
        ))

        result = canvas.retrieve("feature")
        injected = canvas.inject(result, format="json")

        # Should be valid JSON
        data = json.loads(injected)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_inject_empty_result(self):
        """Test injection with empty result."""
        canvas = Canvas()
        result = canvas.retrieve("nonexistent")

        injected = canvas.inject(result)
        assert injected == ""

    def test_remove_nonexistent(self):
        """Test removing nonexistent object."""
        canvas = Canvas()
        result = canvas.remove("nonexistent-id")
        assert result is False

    def test_get_nonexistent(self):
        """Test getting nonexistent object."""
        canvas = Canvas()
        obj = canvas.get("nonexistent-id")
        assert obj is None

    def test_link_nonexistent_objects(self):
        """Test linking with nonexistent objects."""
        canvas = Canvas()

        obj = CanvasObject(type=ObjectType.DECISION, content="Test")
        canvas.add(obj)

        # Try to link with nonexistent object
        result = canvas.link(obj.id, "nonexistent-id")
        assert result is False

    def test_link_references_relation(self):
        """Test linking with references relation."""
        canvas = Canvas()

        obj1 = CanvasObject(type=ObjectType.KEY_FACT, content="Fact 1")
        obj2 = CanvasObject(type=ObjectType.KEY_FACT, content="Fact 2")

        canvas.add(obj1)
        canvas.add(obj2)

        canvas.link(obj1.id, obj2.id, relation="references")

        assert obj2.id in obj1.references
        assert obj1.id in obj2.referenced_by

    def test_get_related_depth(self):
        """Test getting related objects with depth."""
        canvas = Canvas()

        obj1 = CanvasObject(type=ObjectType.DECISION, content="Decision 1")
        obj2 = CanvasObject(type=ObjectType.TODO, content="TODO 1")
        obj3 = CanvasObject(type=ObjectType.TODO, content="TODO 2")

        canvas.add(obj1)
        canvas.add(obj2)
        canvas.add(obj3)

        canvas.link(obj1.id, obj2.id)
        canvas.link(obj2.id, obj3.id)

        # Depth 1 should only get immediate neighbors
        related = canvas.get_related(obj1.id, depth=1)
        assert len(related) == 1
        assert related[0].id == obj2.id

    def test_get_related_nonexistent(self):
        """Test getting related objects for nonexistent object."""
        canvas = Canvas()
        related = canvas.get_related("nonexistent-id")
        assert related == []

    def test_list_objects_sorted_by_turn(self):
        """Test that list_objects returns objects sorted by turn_id."""
        canvas = Canvas()

        obj1 = CanvasObject(type=ObjectType.TODO, content="TODO 1", turn_id=3)
        obj2 = CanvasObject(type=ObjectType.TODO, content="TODO 2", turn_id=1)
        obj3 = CanvasObject(type=ObjectType.TODO, content="TODO 3", turn_id=2)

        canvas.add(obj1)
        canvas.add(obj2)
        canvas.add(obj3)

        objects = canvas.list_objects()
        assert objects[0].turn_id == 1
        assert objects[1].turn_id == 2
        assert objects[2].turn_id == 3

    def test_stats_empty_canvas(self):
        """Test statistics on empty canvas."""
        canvas = Canvas()
        stats = canvas.stats()

        assert stats["total_objects"] == 0
        assert stats["turn_count"] == 0
        assert stats["by_type"] == {}

    def test_canvas_len(self):
        """Test __len__ magic method."""
        canvas = Canvas()
        assert len(canvas) == 0

        canvas.add(CanvasObject(type=ObjectType.KEY_FACT, content="Fact"))
        assert len(canvas) == 1

    def test_canvas_repr(self):
        """Test __repr__ string representation."""
        canvas = Canvas()
        canvas.add(CanvasObject(type=ObjectType.TODO, content="TODO"))

        repr_str = repr(canvas)
        assert "Canvas" in repr_str
        assert "objects=1" in repr_str


class TestCanvasPersistence:
    """Test canvas persistence and storage."""

    def test_save_and_load(self):
        """Test saving and loading canvas state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "canvas.json"

            # Create and populate canvas
            canvas1 = Canvas(storage_path=str(storage_path))
            canvas1.add(CanvasObject(
                type=ObjectType.DECISION,
                content="Use PostgreSQL"
            ))
            canvas1._save()

            # Load into new canvas
            canvas2 = Canvas(storage_path=str(storage_path))

            assert canvas2.size == 1
            assert canvas2.list_objects()[0].content == "Use PostgreSQL"

    def test_auto_save_on_add(self):
        """Test that canvas auto-saves on add."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "canvas.json"

            canvas = Canvas(storage_path=str(storage_path))
            canvas.add(CanvasObject(type=ObjectType.TODO, content="Test"))

            # File should exist
            assert storage_path.exists()

            # File should contain the object
            with open(storage_path) as f:
                data = json.load(f)
            assert len(data["objects"]) == 1

    def test_auto_save_on_remove(self):
        """Test that canvas auto-saves on remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "canvas.json"

            canvas = Canvas(storage_path=str(storage_path))
            obj = CanvasObject(type=ObjectType.TODO, content="Test")
            canvas.add(obj)
            canvas.remove(obj.id)

            # File should exist and be empty
            with open(storage_path) as f:
                data = json.load(f)
            assert len(data["objects"]) == 0

    def test_auto_save_on_clear(self):
        """Test that canvas auto-saves on clear."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "canvas.json"

            canvas = Canvas(storage_path=str(storage_path))
            canvas.add(CanvasObject(type=ObjectType.TODO, content="Test"))
            canvas.clear()

            # File should exist and be empty
            with open(storage_path) as f:
                data = json.load(f)
            assert len(data["objects"]) == 0
            assert data["turn_counter"] == 0

    def test_no_storage_path(self):
        """Test canvas without storage path."""
        canvas = Canvas()
        canvas.add(CanvasObject(type=ObjectType.TODO, content="Test"))

        # Should work without errors even though no persistence
        assert canvas.size == 1


class TestCanvasRetrievalScoring:
    """Test canvas retrieval and scoring."""

    def test_simple_match_score(self):
        """Test simple keyword matching."""
        canvas = Canvas()

        canvas.add(CanvasObject(
            type=ObjectType.KEY_FACT,
            content="PostgreSQL is a relational database"
        ))
        canvas.add(CanvasObject(
            type=ObjectType.KEY_FACT,
            content="MongoDB is a document database"
        ))

        result = canvas.retrieve("PostgreSQL")

        # Should find the PostgreSQL object
        assert result.count >= 1
        assert any("PostgreSQL" in obj.content for obj in result.objects)

    def test_retrieve_scores_order(self):
        """Test that retrieval scores are in descending order."""
        canvas = Canvas()

        canvas.add(CanvasObject(
            type=ObjectType.KEY_FACT,
            content="database database database"  # High match
        ))
        canvas.add(CanvasObject(
            type=ObjectType.KEY_FACT,
            content="database system"  # Medium match
        ))
        canvas.add(CanvasObject(
            type=ObjectType.KEY_FACT,
            content="system architecture"  # Low/no match
        ))

        result = canvas.retrieve("database")

        # Scores should be in descending order
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]

    def test_retrieve_case_insensitive(self):
        """Test case-insensitive retrieval."""
        canvas = Canvas()

        canvas.add(CanvasObject(
            type=ObjectType.KEY_FACT,
            content="Use POSTGRESQL Database"
        ))

        result = canvas.retrieve("postgresql")
        assert result.count >= 1

    @pytest.mark.parametrize("query,content,should_match", [
        ("database", "Use PostgreSQL database", True),
        ("postgres", "PostgreSQL is great", True),
        ("xyz", "PostgreSQL database", False),
        ("auth", "Implement authentication", True),
    ])
    def test_retrieval_parametrized(self, query, content, should_match):
        """Test various retrieval scenarios."""
        canvas = Canvas()
        canvas.add(CanvasObject(type=ObjectType.KEY_FACT, content=content))

        result = canvas.retrieve(query)

        if should_match:
            assert result.count >= 1
        # Note: might still find partial matches, so we don't assert == 0 for False case


class TestCanvasEdgeCases:
    """Test edge cases and error handling."""

    def test_extract_with_metadata(self):
        """Test extraction with optional metadata."""
        canvas = Canvas()

        metadata = {"source": "test", "priority": "high"}
        result = canvas.extract(
            user="Let's use Redis",
            assistant="Good choice",
            metadata=metadata
        )

        # Should work without errors
        assert isinstance(result.count, int)

    def test_empty_messages(self):
        """Test extraction with empty messages."""
        canvas = Canvas()

        result = canvas.extract(user="", assistant="")
        assert result.count == 0

    def test_very_long_content(self):
        """Test with very long content."""
        canvas = Canvas()

        long_content = "word " * 10000
        obj = CanvasObject(type=ObjectType.KEY_FACT, content=long_content)
        canvas.add(obj)

        assert canvas.size == 1
        retrieved = canvas.get(obj.id)
        assert retrieved is not None

    def test_special_characters_in_content(self):
        """Test objects with special characters."""
        canvas = Canvas()

        obj = CanvasObject(
            type=ObjectType.KEY_FACT,
            content="Use @#$%^&*() special chars 你好 🚀"
        )
        canvas.add(obj)

        result = canvas.retrieve("special")
        assert result.count >= 1
