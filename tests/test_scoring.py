"""Tests for confidence scoring system."""

import pytest
from cogcanvas.models import CanvasObject, ObjectType
from cogcanvas.scoring import RuleScorer, ConfidenceScorer, DEFAULT_TRIGGER_WEIGHTS


class TestRuleScorer:
    """Test rule-based confidence scoring."""

    def test_decision_with_triggers(self):
        """Test decision object with trigger words gets high score."""
        scorer = RuleScorer()
        obj = CanvasObject(
            type=ObjectType.DECISION,
            content="We decided to use PostgreSQL for the database",
            context="Database selection discussion",
        )

        score, components = scorer.score(obj)

        # Should have positive trigger score
        assert components["trigger"] > 0
        # Should have context bonus
        assert components["context"] > 0
        # Overall score should be reasonable
        assert 0.5 <= score <= 1.0

    def test_decision_without_triggers(self):
        """Test decision object without trigger words gets lower score."""
        scorer = RuleScorer()
        obj = CanvasObject(
            type=ObjectType.DECISION,
            content="PostgreSQL is the database",  # No decision trigger words
            context="",
        )

        score, components = scorer.score(obj)

        # Should have negative trigger score (penalty)
        assert components["trigger"] < 0
        # No context bonus
        assert components["context"] == 0
        # Overall score should be lower
        assert score < 0.5

    def test_todo_with_triggers(self):
        """Test TODO object with action verbs."""
        scorer = RuleScorer()
        obj = CanvasObject(
            type=ObjectType.TODO,
            content="Need to implement user authentication",
            context="Security requirements",
        )

        score, components = scorer.score(obj)

        assert components["trigger"] > 0
        assert components["context"] > 0
        assert score > 0.5

    def test_length_penalty_too_short(self):
        """Test that very short content gets penalized."""
        scorer = RuleScorer(min_length=10)
        obj = CanvasObject(
            type=ObjectType.KEY_FACT,
            content="DB",  # Too short
            context="",
        )

        score, components = scorer.score(obj)

        # Length should be negative
        assert components["length"] < 0

    def test_length_penalty_too_long(self):
        """Test that very long content gets penalized."""
        scorer = RuleScorer(max_length=100)
        obj = CanvasObject(
            type=ObjectType.KEY_FACT,
            content="A" * 500,  # Way too long
            context="",
        )

        score, components = scorer.score(obj)

        # Length should be negative
        assert components["length"] < 0

    def test_length_ideal(self):
        """Test ideal length range."""
        scorer = RuleScorer(min_length=10, max_length=200)
        obj = CanvasObject(
            type=ObjectType.KEY_FACT,
            content="The API rate limit is 1000 requests per hour",
            context="",
        )

        score, components = scorer.score(obj)

        # Length should be positive or zero
        assert components["length"] >= 0

    def test_context_bonus(self):
        """Test that having context gives a bonus."""
        scorer = RuleScorer(context_bonus=0.1)

        obj_with_context = CanvasObject(
            type=ObjectType.KEY_FACT,
            content="Rate limit is 1000/hour",
            context="API documentation review",
        )

        obj_without_context = CanvasObject(
            type=ObjectType.KEY_FACT,
            content="Rate limit is 1000/hour",
            context="",
        )

        score_with, components_with = scorer.score(obj_with_context)
        score_without, components_without = scorer.score(obj_without_context)

        assert components_with["context"] == 0.1
        assert components_without["context"] == 0.0
        assert score_with > score_without

    def test_key_fact_with_numbers(self):
        """Test key facts with numbers get bonus."""
        scorer = RuleScorer()
        obj = CanvasObject(
            type=ObjectType.KEY_FACT,
            content="The API rate limit is 1000 requests per hour",
            context="",
        )

        score, components = scorer.score(obj)

        # Type-specific should be positive
        assert components["type_specific"] >= 0

    def test_todo_with_action_verbs(self):
        """Test TODOs with action verbs get bonus."""
        scorer = RuleScorer()
        obj = CanvasObject(
            type=ObjectType.TODO,
            content="Implement user authentication and fix login bug",
            context="",
        )

        score, components = scorer.score(obj)

        # Type-specific should be positive
        assert components["type_specific"] > 0

    def test_custom_trigger_weights(self):
        """Test using custom trigger word configuration."""
        custom_weights = {
            "decision": {
                "triggers": ["confirmed", "approved"],
                "weight": 0.5,
            }
        }

        scorer = RuleScorer(trigger_weights=custom_weights)
        obj = CanvasObject(
            type=ObjectType.DECISION,
            content="We confirmed the use of Redis",
            context="",
        )

        score, components = scorer.score(obj)

        # Should match custom trigger
        assert components["trigger"] > 0


class TestConfidenceScorer:
    """Test hybrid confidence scoring."""

    def test_rule_only_scoring(self):
        """Test scoring with rules only (no LLM)."""
        scorer = ConfidenceScorer(use_llm=False)

        obj = CanvasObject(
            type=ObjectType.DECISION,
            content="We decided to use PostgreSQL",
            context="Database selection",
        )

        score, details = scorer.score(obj)

        # Should have rule score
        assert "rule_score" in details
        assert details["rule_score"] > 0

        # Should not have LLM score
        assert details["llm_score"] is None

        # Final score should equal rule score
        assert score == details["rule_score"]

    def test_hybrid_weighting(self):
        """Test that hybrid scoring uses correct weights."""
        # Create scorer with custom weights
        scorer = ConfidenceScorer(
            rule_weight=0.3,
            llm_weight=0.7,
            use_llm=False,  # We'll test weights conceptually
        )

        assert scorer.rule_weight == 0.3
        assert scorer.llm_weight == 0.7

    def test_batch_scoring(self):
        """Test scoring multiple objects at once."""
        scorer = ConfidenceScorer(use_llm=False)

        objects = [
            CanvasObject(
                type=ObjectType.DECISION,
                content="We decided to use PostgreSQL",
                context="Database selection",
            ),
            CanvasObject(
                type=ObjectType.TODO,
                content="Need to implement auth",
                context="Security tasks",
            ),
            CanvasObject(
                type=ObjectType.KEY_FACT,
                content="API rate limit is 1000/hour",
                context="API docs",
            ),
        ]

        results = scorer.score_batch(objects)

        # Should return same number of results
        assert len(results) == len(objects)

        # Each result should have score and details
        for score, details in results:
            assert 0.0 <= score <= 1.0
            assert "rule_score" in details
            assert "final_score" in details

    def test_threshold_filtering(self):
        """Test that scores can be used for filtering."""
        scorer = ConfidenceScorer(use_llm=False)
        threshold = 0.5

        good_obj = CanvasObject(
            type=ObjectType.DECISION,
            content="We decided to use PostgreSQL for reliability",
            context="Database selection after thorough evaluation",
        )

        poor_obj = CanvasObject(
            type=ObjectType.DECISION,
            content="DB",  # Too short, no triggers
            context="",
        )

        good_score, _ = scorer.score(good_obj)
        poor_score, _ = scorer.score(poor_obj)

        # Good object should pass threshold
        assert good_score >= threshold

        # Poor object should fail threshold
        assert poor_score < threshold


class TestDefaultTriggerWeights:
    """Test default trigger weight configuration."""

    def test_all_types_have_configs(self):
        """Test that all object types have trigger configurations."""
        expected_types = ["decision", "todo", "key_fact", "reminder", "insight"]

        for obj_type in expected_types:
            assert obj_type in DEFAULT_TRIGGER_WEIGHTS
            config = DEFAULT_TRIGGER_WEIGHTS[obj_type]
            assert "triggers" in config
            assert "weight" in config
            assert isinstance(config["triggers"], list)
            assert isinstance(config["weight"], (int, float))
            assert len(config["triggers"]) > 0

    def test_reasonable_weights(self):
        """Test that weights are in reasonable range."""
        for config in DEFAULT_TRIGGER_WEIGHTS.values():
            weight = config["weight"]
            # Weights should be positive and not too large
            assert 0.0 < weight <= 0.5
