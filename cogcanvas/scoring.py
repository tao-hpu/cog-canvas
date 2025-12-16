"""Confidence scoring system for canvas object extraction."""

from typing import Dict, List, Optional, Tuple
import logging
import re

from cogcanvas.models import CanvasObject, ObjectType

logger = logging.getLogger(__name__)


# Default trigger words for each object type with their weight contributions
DEFAULT_TRIGGER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "decision": {
        "triggers": ["decide", "choose", "use", "go with", "let's", "select", "pick"],
        "weight": 0.3,
    },
    "todo": {
        "triggers": ["todo", "need to", "should", "must", "will", "have to", "got to"],
        "weight": 0.3,
    },
    "key_fact": {
        "triggers": ["is", "are", "equals", "limit", "rate", "value", "number"],
        "weight": 0.2,
    },
    "reminder": {
        "triggers": ["remember", "don't forget", "prefer", "always", "never", "note that"],
        "weight": 0.3,
    },
    "insight": {
        "triggers": ["realize", "notice", "found that", "turns out", "discovered", "learned"],
        "weight": 0.3,
    },
}


class RuleScorer:
    """
    Rule-based confidence scorer for canvas objects.

    Uses heuristics like trigger words, content length, and context presence
    to estimate confidence without requiring LLM calls.
    """

    def __init__(
        self,
        trigger_weights: Optional[Dict[str, Dict[str, float]]] = None,
        min_length: int = 10,
        max_length: int = 500,
        context_bonus: float = 0.1,
    ):
        """
        Initialize the rule scorer.

        Args:
            trigger_weights: Custom trigger word configurations
            min_length: Minimum content length (shorter gets penalized)
            max_length: Maximum ideal length (longer gets penalized)
            context_bonus: Bonus score for having context
        """
        self.trigger_weights = trigger_weights or DEFAULT_TRIGGER_WEIGHTS
        self.min_length = min_length
        self.max_length = max_length
        self.context_bonus = context_bonus

    def score(self, obj: CanvasObject) -> Tuple[float, Dict[str, float]]:
        """
        Calculate rule-based confidence score for an object.

        Args:
            obj: The canvas object to score

        Returns:
            Tuple of (final_score, component_scores_dict)
        """
        components = {}

        # 1. Trigger word matching
        trigger_score = self._score_triggers(obj)
        components["trigger"] = trigger_score

        # 2. Length scoring
        length_score = self._score_length(obj)
        components["length"] = length_score

        # 3. Context bonus
        context_score = self.context_bonus if obj.context.strip() else 0.0
        components["context"] = context_score

        # 4. Type-specific heuristics
        type_score = self._score_type_specific(obj)
        components["type_specific"] = type_score

        # Combine components
        base_score = 0.5  # Start at neutral
        final_score = min(1.0, max(0.0, base_score + trigger_score + length_score + context_score + type_score))

        logger.debug(
            f"Rule scoring for {obj.id}: trigger={trigger_score:.2f}, "
            f"length={length_score:.2f}, context={context_score:.2f}, "
            f"type={type_score:.2f} -> {final_score:.2f}"
        )

        return final_score, components

    def _score_triggers(self, obj: CanvasObject) -> float:
        """Score based on trigger word presence."""
        obj_type_name = obj.type.value
        if obj_type_name not in self.trigger_weights:
            return 0.0

        config = self.trigger_weights[obj_type_name]
        triggers = config.get("triggers", [])
        weight = config.get("weight", 0.2)

        content_lower = obj.content.lower()

        # Check if any trigger words are present
        found_triggers = [t for t in triggers if t in content_lower]

        if not found_triggers:
            return -0.2  # Penalty for missing trigger words

        # More trigger words = higher confidence (but capped)
        trigger_count = len(found_triggers)
        return min(weight, weight * (trigger_count / 2))

    def _score_length(self, obj: CanvasObject) -> float:
        """Score based on content length."""
        length = len(obj.content)

        # Too short - likely incomplete
        if length < self.min_length:
            return -0.3

        # Ideal length range
        if self.min_length <= length <= self.max_length:
            return 0.1

        # Too long - might be overly verbose or multiple items
        if length > self.max_length:
            penalty = min(0.3, (length - self.max_length) / 1000)
            return -penalty

        return 0.0

    def _score_type_specific(self, obj: CanvasObject) -> float:
        """Apply type-specific scoring heuristics."""
        if obj.type == ObjectType.KEY_FACT:
            # Key facts should contain numbers, names, or specific values
            if re.search(r'\d+', obj.content) or re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', obj.content):
                return 0.1
            return -0.1

        elif obj.type == ObjectType.TODO:
            # TODOs should be action-oriented (verbs)
            action_verbs = ["create", "build", "implement", "fix", "test", "deploy", "write", "update"]
            if any(verb in obj.content.lower() for verb in action_verbs):
                return 0.1
            return 0.0

        elif obj.type == ObjectType.DECISION:
            # Decisions should be definitive
            if any(word in obj.content.lower() for word in ["will", "going to", "decided"]):
                return 0.1
            return 0.0

        return 0.0


class LLMScorer:
    """
    LLM-based confidence scorer for canvas objects.

    Uses an LLM to evaluate extraction quality, relevance, and accuracy.
    This is more accurate but slower than rule-based scoring.
    """

    def __init__(self, llm_backend=None):
        """
        Initialize LLM scorer.

        Args:
            llm_backend: LLM backend for scoring (optional)
        """
        self.llm_backend = llm_backend

    def score(self, obj: CanvasObject, dialogue_context: str = "") -> float:
        """
        Calculate LLM-based confidence score.

        Args:
            obj: The canvas object to score
            dialogue_context: Original dialogue for context

        Returns:
            Confidence score [0, 1]
        """
        if not self.llm_backend:
            logger.warning("No LLM backend configured for LLM scoring")
            return 0.5

        # TODO: Implement actual LLM scoring
        # This would prompt the LLM to evaluate:
        # - Is this extraction accurate?
        # - Is the type classification correct?
        # - Is this information important enough to store?

        logger.debug(f"LLM scoring for {obj.id} (not yet implemented)")
        return 0.5


class ConfidenceScorer:
    """
    Hybrid confidence scorer combining rule-based and LLM-based scoring.

    Uses fast rule-based heuristics as a baseline, optionally enhanced
    with LLM-based scoring for higher accuracy.
    """

    def __init__(
        self,
        rule_weight: float = 0.3,
        llm_weight: float = 0.7,
        use_llm: bool = False,
        llm_backend=None,
        **rule_scorer_kwargs,
    ):
        """
        Initialize the hybrid confidence scorer.

        Args:
            rule_weight: Weight for rule-based score in final calculation
            llm_weight: Weight for LLM score in final calculation
            use_llm: Whether to use LLM scoring
            llm_backend: LLM backend for scoring
            **rule_scorer_kwargs: Arguments for RuleScorer
        """
        self.rule_weight = rule_weight
        self.llm_weight = llm_weight
        self.use_llm = use_llm

        self.rule_scorer = RuleScorer(**rule_scorer_kwargs)
        self.llm_scorer = LLMScorer(llm_backend) if use_llm else None

    def score(
        self,
        obj: CanvasObject,
        dialogue_context: str = "",
    ) -> Tuple[float, Dict[str, any]]:
        """
        Calculate hybrid confidence score for an object.

        Args:
            obj: The canvas object to score
            dialogue_context: Original dialogue for LLM context

        Returns:
            Tuple of (final_score, details_dict)
        """
        details = {}

        # Always get rule-based score
        rule_score, rule_components = self.rule_scorer.score(obj)
        details["rule_score"] = rule_score
        details["rule_components"] = rule_components

        # Optionally get LLM score
        if self.use_llm and self.llm_scorer:
            llm_score = self.llm_scorer.score(obj, dialogue_context)
            details["llm_score"] = llm_score

            # Weighted combination
            final_score = (self.rule_weight * rule_score) + (self.llm_weight * llm_score)
        else:
            # Use only rule score
            final_score = rule_score
            details["llm_score"] = None

        details["final_score"] = final_score

        logger.info(
            f"Scored {obj.id} ({obj.type.value}): "
            f"rule={rule_score:.2f}, llm={details.get('llm_score', 'N/A')}, "
            f"final={final_score:.2f}"
        )

        return final_score, details

    def score_batch(
        self,
        objects: List[CanvasObject],
        dialogue_context: str = "",
    ) -> List[Tuple[float, Dict[str, any]]]:
        """
        Score multiple objects efficiently.

        Args:
            objects: List of canvas objects to score
            dialogue_context: Original dialogue for context

        Returns:
            List of (score, details) tuples matching input order
        """
        return [self.score(obj, dialogue_context) for obj in objects]
