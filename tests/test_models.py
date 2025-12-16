"""Tests for CogCanvas data models."""

import pytest
import time
from cogcanvas.models import (
    ObjectType,
    CanvasObject,
    ExtractionResult,
    RetrievalResult,
)


class TestObjectType:
    """Test suite for ObjectType enum."""

    def test_enum_values(self):
        """Test all enum values are correct."""
        assert ObjectType.DECISION.value == "decision"
        assert ObjectType.TODO.value == "todo"
        assert ObjectType.KEY_FACT.value == "key_fact"
        assert ObjectType.REMINDER.value == "reminder"
        assert ObjectType.INSIGHT.value == "insight"

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        assert ObjectType("decision") == ObjectType.DECISION
        assert ObjectType("todo") == ObjectType.TODO
        assert ObjectType("key_fact") == ObjectType.KEY_FACT
        assert ObjectType("reminder") == ObjectType.REMINDER
        assert ObjectType("insight") == ObjectType.INSIGHT

    def test_enum_invalid_value(self):
        """Test invalid enum value raises error."""
        with pytest.raises(ValueError):
            ObjectType("invalid_type")

    def test_enum_equality(self):
        """Test enum equality comparison."""
        obj_type = ObjectType.DECISION
        assert obj_type == ObjectType.DECISION
        assert obj_type != ObjectType.TODO


class TestCanvasObject:
    """Test suite for CanvasObject dataclass."""

    def test_default_initialization(self):
        """Test object creation with defaults."""
        obj = CanvasObject()
        assert obj.id is not None
        assert len(obj.id) == 8  # UUID first 8 chars
        assert obj.type == ObjectType.KEY_FACT
        assert obj.content == ""
        assert obj.context == ""
        assert obj.turn_id == 0
        assert obj.confidence == 1.0
        assert obj.embedding is None
        assert obj.references == []
        assert obj.referenced_by == []
        assert obj.leads_to == []
        assert obj.caused_by == []

    def test_custom_initialization(self, sample_decision):
        """Test object creation with custom values."""
        assert sample_decision.type == ObjectType.DECISION
        assert sample_decision.content == "Use PostgreSQL for the database"
        assert sample_decision.context == "Team decided after evaluating options"
        assert sample_decision.confidence == 0.9
        assert sample_decision.turn_id == 1

    def test_timestamp_generation(self):
        """Test timestamp is automatically generated."""
        before = time.time()
        obj = CanvasObject()
        after = time.time()
        assert before <= obj.timestamp <= after

    def test_unique_ids(self):
        """Test that each object gets a unique ID."""
        obj1 = CanvasObject()
        obj2 = CanvasObject()
        assert obj1.id != obj2.id

    def test_to_dict_serialization(self, sample_decision):
        """Test serialization to dictionary."""
        data = sample_decision.to_dict()

        assert isinstance(data, dict)
        assert data["type"] == "decision"
        assert data["content"] == sample_decision.content
        assert data["context"] == sample_decision.context
        assert data["turn_id"] == 1
        assert data["confidence"] == 0.9
        assert "id" in data
        assert "timestamp" in data
        assert "embedding" in data
        assert "references" in data
        assert "referenced_by" in data
        assert "leads_to" in data
        assert "caused_by" in data

    def test_to_dict_with_embedding(self):
        """Test serialization with embedding vector."""
        obj = CanvasObject(
            type=ObjectType.TODO,
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
        )
        data = obj.to_dict()
        assert data["embedding"] == [0.1, 0.2, 0.3]

    def test_to_dict_with_relationships(self):
        """Test serialization with graph relationships."""
        obj = CanvasObject(
            type=ObjectType.DECISION,
            content="Test",
            references=["ref1", "ref2"],
            referenced_by=["refby1"],
            leads_to=["lead1"],
            caused_by=["cause1", "cause2"],
        )
        data = obj.to_dict()
        assert data["references"] == ["ref1", "ref2"]
        assert data["referenced_by"] == ["refby1"]
        assert data["leads_to"] == ["lead1"]
        assert data["caused_by"] == ["cause1", "cause2"]

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test123",
            "type": "todo",
            "content": "Test TODO",
            "context": "Test context",
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
        assert obj.context == "Test context"
        assert obj.turn_id == 1
        assert obj.timestamp == 1234567890.0
        assert obj.confidence == 0.9

    def test_from_dict_with_embedding(self):
        """Test deserialization with embedding vector."""
        data = {
            "id": "test456",
            "type": "key_fact",
            "content": "Test fact",
            "context": "",
            "turn_id": 2,
            "timestamp": 1234567890.0,
            "confidence": 1.0,
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "references": [],
            "referenced_by": [],
            "leads_to": [],
            "caused_by": [],
        }

        obj = CanvasObject.from_dict(data)
        assert obj.embedding == [0.1, 0.2, 0.3, 0.4]

    def test_roundtrip_serialization(self, sample_objects):
        """Test that to_dict -> from_dict preserves data."""
        for original in sample_objects:
            data = original.to_dict()
            restored = CanvasObject.from_dict(data)

            assert restored.id == original.id
            assert restored.type == original.type
            assert restored.content == original.content
            assert restored.context == original.context
            assert restored.turn_id == original.turn_id
            assert restored.confidence == original.confidence

    def test_repr_output(self, sample_decision):
        """Test string representation."""
        repr_str = repr(sample_decision)
        assert "CanvasObject" in repr_str
        assert "decision" in repr_str
        assert "PostgreSQL" in repr_str


class TestExtractionResult:
    """Test suite for ExtractionResult."""

    def test_empty_result(self):
        """Test empty extraction result."""
        result = ExtractionResult()
        assert result.objects == []
        assert result.raw_response == ""
        assert result.extraction_time == 0.0
        assert result.model_used == ""
        assert result.count == 0

    def test_result_with_objects(self, sample_objects):
        """Test result with extracted objects."""
        result = ExtractionResult(
            objects=sample_objects,
            raw_response="Mock response",
            extraction_time=0.5,
            model_used="gpt-4o-mini",
        )
        assert result.count == 5
        assert len(result.objects) == 5
        assert result.extraction_time == 0.5
        assert result.model_used == "gpt-4o-mini"

    def test_count_property(self, sample_decision, sample_todo):
        """Test count property."""
        result = ExtractionResult(objects=[sample_decision, sample_todo])
        assert result.count == 2

    def test_by_type_filtering(self, sample_objects):
        """Test filtering objects by type."""
        result = ExtractionResult(objects=sample_objects)

        decisions = result.by_type(ObjectType.DECISION)
        assert len(decisions) == 1
        assert decisions[0].type == ObjectType.DECISION

        todos = result.by_type(ObjectType.TODO)
        assert len(todos) == 1
        assert todos[0].type == ObjectType.TODO

        facts = result.by_type(ObjectType.KEY_FACT)
        assert len(facts) == 1

    def test_by_type_empty(self, sample_decision):
        """Test filtering when no objects match."""
        result = ExtractionResult(objects=[sample_decision])
        todos = result.by_type(ObjectType.TODO)
        assert len(todos) == 0

    @pytest.mark.parametrize(
        "obj_type,expected_count",
        [
            (ObjectType.DECISION, 1),
            (ObjectType.TODO, 1),
            (ObjectType.KEY_FACT, 1),
            (ObjectType.REMINDER, 1),
            (ObjectType.INSIGHT, 1),
        ],
    )
    def test_by_type_parametrized(self, sample_objects, obj_type, expected_count):
        """Test filtering with parametrized types."""
        result = ExtractionResult(objects=sample_objects)
        filtered = result.by_type(obj_type)
        assert len(filtered) == expected_count


class TestRetrievalResult:
    """Test suite for RetrievalResult."""

    def test_empty_result(self):
        """Test empty retrieval result."""
        result = RetrievalResult()
        assert result.objects == []
        assert result.scores == []
        assert result.query == ""
        assert result.retrieval_time == 0.0
        assert result.count == 0

    def test_result_with_objects(self, sample_objects):
        """Test result with retrieved objects."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = RetrievalResult(
            objects=sample_objects,
            scores=scores,
            query="test query",
            retrieval_time=0.1,
        )
        assert result.count == 5
        assert len(result.objects) == 5
        assert len(result.scores) == 5
        assert result.query == "test query"
        assert result.retrieval_time == 0.1

    def test_count_property(self, sample_decision, sample_todo):
        """Test count property."""
        result = RetrievalResult(
            objects=[sample_decision, sample_todo],
            scores=[0.9, 0.8],
        )
        assert result.count == 2

    def test_top_k_retrieval(self, sample_objects):
        """Test getting top k objects."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = RetrievalResult(objects=sample_objects, scores=scores)

        top_3 = result.top_k(3)
        assert len(top_3) == 3
        assert top_3[0] == sample_objects[0]
        assert top_3[1] == sample_objects[1]
        assert top_3[2] == sample_objects[2]

    def test_top_k_more_than_available(self, sample_decision, sample_todo):
        """Test top_k when k > available objects."""
        result = RetrievalResult(
            objects=[sample_decision, sample_todo],
            scores=[0.9, 0.8],
        )
        top_10 = result.top_k(10)
        assert len(top_10) == 2

    def test_top_k_zero(self, sample_objects):
        """Test top_k with k=0."""
        result = RetrievalResult(objects=sample_objects, scores=[0.9] * 5)
        top_0 = result.top_k(0)
        assert len(top_0) == 0

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_top_k_parametrized(self, sample_objects, k):
        """Test top_k with various k values."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = RetrievalResult(objects=sample_objects, scores=scores)
        top = result.top_k(k)
        assert len(top) == k
        assert all(obj in sample_objects for obj in top)

    def test_scores_alignment(self, sample_objects):
        """Test that scores align with objects."""
        scores = [0.95, 0.85, 0.75, 0.65, 0.55]
        result = RetrievalResult(objects=sample_objects, scores=scores)

        # Scores should correspond to objects at same index
        for i, (obj, score) in enumerate(zip(result.objects, result.scores)):
            assert obj == sample_objects[i]
            assert score == scores[i]
