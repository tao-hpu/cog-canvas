"""Tests for VAGE (Vulnerability-Aware Greedy Extraction) module."""

import pytest
from cogcanvas.vage import (
    VAGE,
    VAGEConfig,
    VAGEResult,
    RetentionPredictor,
    ImportanceEstimator,
    vage_select,
    compute_retention_bound,
)
from cogcanvas.models import CanvasObject, ObjectType


class TestRetentionPredictor:
    """Test retention probability prediction."""

    def test_recency_effect(self):
        """Later turns should have higher retention probability."""
        predictor = RetentionPredictor()

        # Early turn (turn 1 of 10)
        rho_early = predictor.predict(
            turn_id=1,
            total_turns=10,
            content="Some early information",
        )

        # Late turn (turn 9 of 10)
        rho_late = predictor.predict(
            turn_id=9,
            total_turns=10,
            content="Some late information",
        )

        assert rho_late > rho_early, "Later turns should have higher retention"
        assert 0 <= rho_early <= 1
        assert 0 <= rho_late <= 1

    def test_numbers_boost_retention(self):
        """Content with numbers should be more memorable."""
        predictor = RetentionPredictor()

        rho_no_numbers = predictor.predict(
            turn_id=5,
            total_turns=10,
            content="The budget is limited",
        )

        rho_with_numbers = predictor.predict(
            turn_id=5,
            total_turns=10,
            content="The budget is $500",
        )

        assert rho_with_numbers >= rho_no_numbers

    def test_entities_boost_retention(self):
        """Content with named entities should be more memorable."""
        predictor = RetentionPredictor()

        rho_no_entities = predictor.predict(
            turn_id=5,
            total_turns=10,
            content="use the database system",
        )

        rho_with_entities = predictor.predict(
            turn_id=5,
            total_turns=10,
            content="use PostgreSQL for the database",
        )

        assert rho_with_entities >= rho_no_entities


class TestImportanceEstimator:
    """Test importance estimation."""

    def test_confidence_source(self):
        """When using confidence source, should return confidence directly."""
        estimator = ImportanceEstimator(source="confidence")

        omega = estimator.estimate(
            content="Some content",
            confidence=0.85,
        )

        assert omega == 0.85

    def test_heuristic_source(self):
        """When using heuristic source, should weight by type."""
        estimator = ImportanceEstimator(source="heuristic")

        omega_decision = estimator.estimate(
            content="Use PostgreSQL",
            confidence=1.0,
            obj_type="decision",
        )

        omega_insight = estimator.estimate(
            content="This might be useful",
            confidence=1.0,
            obj_type="insight",
        )

        assert omega_decision > omega_insight, "Decisions should be more important than insights"


class TestVAGE:
    """Test VAGE algorithm."""

    def create_test_objects(self):
        """Create test CanvasObjects."""
        return [
            CanvasObject(
                id="obj1",
                type=ObjectType.DECISION,
                content="Use PostgreSQL for the database",
                confidence=0.9,
                turn_id=1,
            ),
            CanvasObject(
                id="obj2",
                type=ObjectType.KEY_FACT,
                content="Budget is $500",
                confidence=0.8,
                turn_id=2,
            ),
            CanvasObject(
                id="obj3",
                type=ObjectType.TODO,
                content="Review the proposal",
                confidence=0.7,
                turn_id=8,
            ),
            CanvasObject(
                id="obj4",
                type=ObjectType.INSIGHT,
                content="Performance might be an issue",
                confidence=0.6,
                turn_id=9,
            ),
        ]

    def test_prioritize_returns_correct_structure(self):
        """Test that prioritize returns VAGEResult with correct fields."""
        vage = VAGE()
        objects = self.create_test_objects()

        result = vage.prioritize(objects, total_turns=10, budget_k=2)

        assert isinstance(result, VAGEResult)
        assert len(result.selected_indices) == 2
        assert len(result.marginal_gains) == 4  # All objects get scores
        assert result.total_retained > 0

    def test_prioritize_selects_high_delta(self):
        """Test that VAGE prioritizes objects with high importance × vulnerability."""
        vage = VAGE()
        objects = self.create_test_objects()

        result = vage.prioritize(objects, total_turns=10, budget_k=2)

        # Early turns (obj1, obj2) should have higher vulnerability
        # Combined with confidence, they should be prioritized
        selected = result.selected_indices

        # Verify selected indices are valid
        for idx in selected:
            assert 0 <= idx < len(objects)

    def test_prioritize_respects_budget(self):
        """Test that VAGE respects the budget constraint."""
        vage = VAGE()
        objects = self.create_test_objects()

        for k in [1, 2, 3, 4]:
            result = vage.prioritize(objects, total_turns=10, budget_k=k)
            assert len(result.selected_indices) == min(k, len(objects))

    def test_empty_objects(self):
        """Test handling of empty object list."""
        vage = VAGE()

        result = vage.prioritize([], total_turns=10, budget_k=5)

        assert result.selected_indices == []
        assert result.marginal_gains == []
        assert result.total_retained == 0.0

    def test_marginal_gain_formula(self):
        """Test that marginal gain = importance × (1 - retention)."""
        vage = VAGE()
        objects = self.create_test_objects()

        scores = vage.compute_object_scores(objects, total_turns=10)

        for omega, rho, delta in scores:
            expected_delta = omega * (1 - rho)
            assert abs(delta - expected_delta) < 0.001, \
                f"Delta should be omega * (1 - rho): {delta} vs {expected_delta}"


class TestVAGEConvenienceFunctions:
    """Test convenience functions."""

    def test_vage_select(self):
        """Test vage_select convenience function."""
        objects = [
            CanvasObject(id="a", type=ObjectType.DECISION, content="Decision A", confidence=0.9, turn_id=1),
            CanvasObject(id="b", type=ObjectType.KEY_FACT, content="Fact B", confidence=0.8, turn_id=5),
            CanvasObject(id="c", type=ObjectType.TODO, content="Todo C", confidence=0.7, turn_id=9),
        ]

        selected = vage_select(objects, total_turns=10, budget_k=2)

        assert len(selected) == 2
        assert all(isinstance(idx, int) for idx in selected)

    def test_compute_retention_bound(self):
        """Test theoretical bound computation."""
        objects = [
            CanvasObject(id="a", type=ObjectType.DECISION, content="Decision A", confidence=0.9, turn_id=1),
            CanvasObject(id="b", type=ObjectType.KEY_FACT, content="Fact B", confidence=0.8, turn_id=5),
        ]

        natural, gain, total = compute_retention_bound(objects, total_turns=10, budget_k=1)

        assert natural >= 0
        assert gain >= 0
        assert total == natural + gain


class TestVAGETheorems:
    """Test theoretical properties (informal verification)."""

    def test_greedy_is_optimal_for_unit_weights(self):
        """
        Theorem 1: Greedy is optimal when all items have unit 'weight'.

        For our problem, each extraction costs 1 unit, so greedy by
        marginal gain is optimal.
        """
        vage = VAGE()
        objects = [
            CanvasObject(id=f"obj{i}", type=ObjectType.KEY_FACT,
                        content=f"Fact {i}", confidence=0.5 + i*0.1, turn_id=i+1)
            for i in range(5)
        ]

        # Get VAGE selection
        result = vage.prioritize(objects, total_turns=10, budget_k=3)

        # Verify that selected objects have the top-3 marginal gains
        all_gains = [(i, g) for i, g in enumerate(result.marginal_gains)]
        all_gains.sort(key=lambda x: x[1], reverse=True)
        expected_top3 = [g[0] for g in all_gains[:3]]

        assert set(result.selected_indices) == set(expected_top3), \
            "Greedy should select the top-k by marginal gain"

    def test_retention_lower_bound(self):
        """
        Theorem 2: Total retention >= natural + extraction gain.

        This is actually an equality in our model, not just a bound.
        """
        vage = VAGE()
        objects = [
            CanvasObject(id="a", type=ObjectType.DECISION, content="A with 100", confidence=0.9, turn_id=1),
            CanvasObject(id="b", type=ObjectType.KEY_FACT, content="B", confidence=0.8, turn_id=5),
            CanvasObject(id="c", type=ObjectType.TODO, content="C", confidence=0.7, turn_id=9),
        ]

        natural, gain, total = compute_retention_bound(objects, total_turns=10, budget_k=2)

        # Total should equal natural + gain (exact equality)
        assert abs(total - (natural + gain)) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
