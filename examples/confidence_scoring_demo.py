"""
Demonstration of confidence scoring in CogCanvas.

This example shows how to:
1. Use confidence scoring to filter low-quality extractions
2. Adjust confidence thresholds
3. Access scoring details
4. Debug filtered objects
"""

from cogcanvas.models import CanvasObject, ObjectType
from cogcanvas.scoring import ConfidenceScorer, RuleScorer

def demo_rule_scorer():
    """Demonstrate rule-based confidence scoring."""
    print("=" * 60)
    print("DEMO 1: Rule-Based Confidence Scoring")
    print("=" * 60)

    scorer = RuleScorer()

    # Example 1: Good quality decision
    good_decision = CanvasObject(
        type=ObjectType.DECISION,
        content="We decided to use PostgreSQL for the database after evaluating MySQL and MongoDB",
        context="Database selection meeting - discussed scalability and ACID requirements",
    )

    score, components = scorer.score(good_decision)
    print(f"\n✓ Good Decision Object:")
    print(f"  Content: {good_decision.content}")
    print(f"  Score: {score:.2f}")
    print(f"  Components: trigger={components['trigger']:.2f}, "
          f"length={components['length']:.2f}, context={components['context']:.2f}")

    # Example 2: Poor quality decision
    poor_decision = CanvasObject(
        type=ObjectType.DECISION,
        content="DB",  # Too short, no triggers
        context="",
    )

    score, components = scorer.score(poor_decision)
    print(f"\n✗ Poor Decision Object:")
    print(f"  Content: {poor_decision.content}")
    print(f"  Score: {score:.2f}")
    print(f"  Components: trigger={components['trigger']:.2f}, "
          f"length={components['length']:.2f}, context={components['context']:.2f}")

    # Example 3: TODO with action verbs
    todo = CanvasObject(
        type=ObjectType.TODO,
        content="Need to implement user authentication and fix the login bug",
        context="Security sprint planning",
    )

    score, components = scorer.score(todo)
    print(f"\n✓ TODO Object:")
    print(f"  Content: {todo.content}")
    print(f"  Score: {score:.2f}")
    print(f"  Components: trigger={components['trigger']:.2f}, "
          f"length={components['length']:.2f}, context={components['context']:.2f}, "
          f"type_specific={components['type_specific']:.2f}")

    # Example 4: Key fact with numbers
    fact = CanvasObject(
        type=ObjectType.KEY_FACT,
        content="The API rate limit is 1000 requests per hour",
        context="API documentation review",
    )

    score, components = scorer.score(fact)
    print(f"\n✓ Key Fact Object:")
    print(f"  Content: {fact.content}")
    print(f"  Score: {score:.2f}")
    print(f"  Components: trigger={components['trigger']:.2f}, "
          f"length={components['length']:.2f}, context={components['context']:.2f}, "
          f"type_specific={components['type_specific']:.2f}")


def demo_confidence_scorer():
    """Demonstrate hybrid confidence scoring."""
    print("\n" + "=" * 60)
    print("DEMO 2: Hybrid Confidence Scoring")
    print("=" * 60)

    # Create scorer with rule-only mode (no LLM)
    scorer = ConfidenceScorer(use_llm=False)

    # Create a batch of objects
    objects = [
        CanvasObject(
            type=ObjectType.DECISION,
            content="We decided to use PostgreSQL for reliability and performance",
            context="Database selection after thorough evaluation",
        ),
        CanvasObject(
            type=ObjectType.TODO,
            content="Implement OAuth2 authentication",
            context="Security requirements",
        ),
        CanvasObject(
            type=ObjectType.INSIGHT,
            content="We realized that caching reduces response time by 80%",
            context="Performance optimization findings",
        ),
        CanvasObject(
            type=ObjectType.KEY_FACT,
            content="DB",  # Poor quality
            context="",
        ),
    ]

    # Score all objects
    results = scorer.score_batch(objects)

    print(f"\nScored {len(objects)} objects:")
    for obj, (score, details) in zip(objects, results):
        status = "✓ PASS" if score >= 0.5 else "✗ FAIL"
        print(f"\n{status} [{obj.type.value}] - Score: {score:.2f}")
        print(f"  Content: {obj.content[:60]}...")
        print(f"  Rule Score: {details['rule_score']:.2f}")


def demo_threshold_filtering():
    """Demonstrate threshold-based filtering."""
    print("\n" + "=" * 60)
    print("DEMO 3: Threshold-Based Filtering")
    print("=" * 60)

    scorer = ConfidenceScorer(use_llm=False)

    # Create objects with varying quality
    objects = [
        ("High Quality", CanvasObject(
            type=ObjectType.DECISION,
            content="We decided to use PostgreSQL for the database after careful evaluation of requirements",
            context="Database selection meeting with detailed analysis",
        )),
        ("Medium Quality", CanvasObject(
            type=ObjectType.TODO,
            content="implement auth",  # No capital, short
            context="",
        )),
        ("Low Quality", CanvasObject(
            type=ObjectType.KEY_FACT,
            content="db",  # Very short
            context="",
        )),
    ]

    thresholds = [0.3, 0.5, 0.7]

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        passed = 0
        failed = 0

        for label, obj in objects:
            score, _ = scorer.score(obj)
            if score >= threshold:
                print(f"  ✓ PASS: {label} (score: {score:.2f})")
                passed += 1
            else:
                print(f"  ✗ FAIL: {label} (score: {score:.2f})")
                failed += 1

        print(f"  Summary: {passed} passed, {failed} failed")


def demo_custom_triggers():
    """Demonstrate custom trigger word configuration."""
    print("\n" + "=" * 60)
    print("DEMO 4: Custom Trigger Words")
    print("=" * 60)

    # Custom trigger configuration for a specific domain
    custom_triggers = {
        "decision": {
            "triggers": ["confirmed", "approved", "selected", "finalized"],
            "weight": 0.4,
        },
        "key_fact": {
            "triggers": ["metric", "KPI", "baseline", "target", "actual"],
            "weight": 0.3,
        },
    }

    scorer = RuleScorer(trigger_weights=custom_triggers)

    # Test with custom triggers
    obj1 = CanvasObject(
        type=ObjectType.DECISION,
        content="The team confirmed the use of microservices architecture",
        context="Architecture review meeting",
    )

    obj2 = CanvasObject(
        type=ObjectType.KEY_FACT,
        content="The baseline response time is 200ms, target is 100ms",
        context="Performance metrics discussion",
    )

    score1, comp1 = scorer.score(obj1)
    score2, comp2 = scorer.score(obj2)

    print(f"\nCustom Decision Trigger ('confirmed'):")
    print(f"  Content: {obj1.content}")
    print(f"  Score: {score1:.2f} (trigger: {comp1['trigger']:.2f})")

    print(f"\nCustom Fact Trigger ('baseline', 'target'):")
    print(f"  Content: {obj2.content}")
    print(f"  Score: {score2:.2f} (trigger: {comp2['trigger']:.2f})")


def demo_scoring_components():
    """Demonstrate detailed scoring components."""
    print("\n" + "=" * 60)
    print("DEMO 5: Detailed Scoring Components")
    print("=" * 60)

    scorer = RuleScorer()

    test_cases = [
        ("With Context", CanvasObject(
            type=ObjectType.DECISION,
            content="We decided to use Redis for caching",
            context="Performance optimization discussion",
        )),
        ("Without Context", CanvasObject(
            type=ObjectType.DECISION,
            content="We decided to use Redis for caching",
            context="",
        )),
        ("Too Short", CanvasObject(
            type=ObjectType.TODO,
            content="Fix",
            context="Bug tracking",
        )),
        ("Too Long", CanvasObject(
            type=ObjectType.REMINDER,
            content="A" * 600,  # Very long
            context="Notes",
        )),
        ("With Numbers", CanvasObject(
            type=ObjectType.KEY_FACT,
            content="The server has 16GB RAM and 8 CPU cores",
            context="Infrastructure specs",
        )),
    ]

    print(f"\n{'Case':<20} {'Final':<8} {'Trigger':<8} {'Length':<8} {'Context':<8} {'Type':<8}")
    print("-" * 68)

    for label, obj in test_cases:
        score, comp = scorer.score(obj)
        print(f"{label:<20} {score:<8.2f} {comp['trigger']:<8.2f} "
              f"{comp['length']:<8.2f} {comp['context']:<8.2f} "
              f"{comp['type_specific']:<8.2f}")


if __name__ == "__main__":
    # Run all demos
    demo_rule_scorer()
    demo_confidence_scorer()
    demo_threshold_filtering()
    demo_custom_triggers()
    demo_scoring_components()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
