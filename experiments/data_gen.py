"""
Synthetic Conversation Generator for CogCanvas Evaluation.

Generates controlled conversations with:
1. Planted key facts at specific turns (for retention testing)
2. Filler dialogue (to create "distance" between facts and tests)
3. Test questions with ground truth answers
4. Adversarial modes (needles in haystack, distractor injection)

Design based on DOC.md Section 4.1 (Truncation Simulation):
- Turn 1-10: Plant key facts
- Turn 11-40: Filler dialogue
- Turn 40: Compression point (truncation)
- Turn 41-50: Test questions

Usage:
    python -m experiments.data_gen --output experiments/data/eval_set.json --num_conversations 100
"""

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum


class DifficultyLevel(Enum):
    """Difficulty levels for fact planting."""
    EASY = "easy"           # Explicit, memorable facts
    MEDIUM = "medium"       # Embedded in context
    HARD = "hard"           # Needles in haystack (subtle)
    ADVERSARIAL = "adversarial"  # With contradicting distractors


@dataclass
class PlantedFact:
    """A key fact planted in the conversation for later testing."""
    id: str
    turn_id: int                    # Which turn it appears in
    fact_type: str                  # decision, key_fact, reminder, etc.
    content: str                    # The actual fact content
    quote: str                      # Exact quote in the conversation
    difficulty: DifficultyLevel
    test_question: str              # Question to ask about this fact
    ground_truth: str               # Expected answer
    distractors: List[str] = field(default_factory=list)  # Contradicting info
    distractor_turn_id: Optional[int] = None  # Turn where distractor appears


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: int
    user: str
    assistant: str
    contains_fact: Optional[str] = None  # ID of planted fact, if any
    is_distractor: bool = False          # True if contains contradicting info
    distractor_for_fact: Optional[str] = None  # ID of fact this distracts from
    session_datetime: Optional[str] = None  # Session occurrence time (for LoCoMo)


@dataclass
class SyntheticConversation:
    """A complete synthetic conversation for evaluation."""
    id: str
    turns: List[ConversationTurn]
    planted_facts: List[PlantedFact]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "turns": [asdict(t) for t in self.turns],
            "planted_facts": [
                {**asdict(f), "difficulty": f.difficulty.value}
                for f in self.planted_facts
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SyntheticConversation":
        turns = [ConversationTurn(**t) for t in data["turns"]]
        facts = []
        for f in data["planted_facts"]:
            f = f.copy()
            f["difficulty"] = DifficultyLevel(f["difficulty"])
            facts.append(PlantedFact(**f))
        return cls(
            id=data["id"],
            turns=turns,
            planted_facts=facts,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Fact Templates
# =============================================================================

DECISION_TEMPLATES = [
    {
        "topic": "database",
        "user": "What database should we use for the project?",
        "assistant": "After considering our requirements for scalability and ACID compliance, I recommend using {value}. It offers excellent support for complex queries and has proven reliability.",
        "values": ["PostgreSQL", "MySQL", "MongoDB", "SQLite"],
        "question": "What database was chosen for the project?",
    },
    {
        "topic": "framework",
        "user": "Which web framework would be best for our API?",
        "assistant": "Given our need for high performance and async support, I suggest we go with {value}. It's well-documented and has great community support.",
        "values": ["FastAPI", "Django", "Flask", "Express.js"],
        "question": "What web framework was selected for the API?",
    },
    {
        "topic": "auth",
        "user": "How should we handle authentication?",
        "assistant": "For security and scalability, let's implement {value}. This approach provides stateless authentication and works well with microservices.",
        "values": ["JWT tokens", "OAuth 2.0", "Session-based auth", "API keys"],
        "question": "What authentication method was decided upon?",
    },
    {
        "topic": "hosting",
        "user": "Where should we deploy our application?",
        "assistant": "Considering our budget and scaling needs, I recommend deploying on {value}. It offers good pricing and easy container orchestration.",
        "values": ["AWS", "Google Cloud", "Azure", "DigitalOcean"],
        "question": "What hosting platform was chosen?",
    },
    {
        "topic": "cache",
        "user": "Do we need a caching layer?",
        "assistant": "Yes, definitely. We should use {value} for caching frequently accessed data. This will significantly reduce database load.",
        "values": ["Redis", "Memcached", "Varnish", "local in-memory cache"],
        "question": "What caching solution was selected?",
    },
]

KEY_FACT_TEMPLATES = [
    {
        "topic": "api_limit",
        "user": "What's the rate limit on the external API we're using?",
        "assistant": "The API has a rate limit of {value}. We need to implement throttling to stay within this limit.",
        "values": ["100 requests per minute", "500 requests per hour", "10,000 requests per day", "50 requests per second"],
        "question": "What is the rate limit of the external API?",
    },
    {
        "topic": "deadline",
        "user": "When is the project deadline?",
        "assistant": "The deadline is {value}. We should plan our sprints accordingly to deliver on time.",
        "values": ["March 15th", "end of Q2", "in 6 weeks", "December 1st"],
        "question": "What is the project deadline?",
    },
    {
        "topic": "budget",
        "user": "What's our infrastructure budget?",
        "assistant": "We have a monthly budget of {value} for cloud infrastructure. This should cover our compute and storage needs.",
        "values": ["$500", "$2,000", "$10,000", "$50,000"],
        "question": "What is the monthly infrastructure budget?",
    },
    {
        "topic": "team_size",
        "user": "How many developers will be working on this?",
        "assistant": "We have a team of {value} working on this project, including backend and frontend developers.",
        "values": ["3 developers", "5 engineers", "a team of 8", "12 people"],
        "question": "How many developers are on the team?",
    },
    {
        "topic": "data_size",
        "user": "How much data are we expecting to handle?",
        "assistant": "Based on projections, we'll be handling approximately {value} of data. Our architecture needs to scale accordingly.",
        "values": ["50GB", "500GB", "2TB", "10TB"],
        "question": "How much data is the system expected to handle?",
    },
]

REMINDER_TEMPLATES = [
    {
        "topic": "code_style",
        "user": "Any preferences for code style?",
        "assistant": "Yes, please {value}. This keeps our codebase consistent and maintainable.",
        "values": [
            "use type hints everywhere",
            "keep functions under 20 lines",
            "write docstrings for all public methods",
            "follow PEP 8 strictly",
        ],
        "question": "What coding style preference was mentioned?",
    },
    {
        "topic": "testing",
        "user": "What's our testing strategy?",
        "assistant": "We should {value}. This ensures reliability and catches regressions early.",
        "values": [
            "maintain at least 80% test coverage",
            "write tests before implementing features",
            "focus on integration tests",
            "use property-based testing",
        ],
        "question": "What was the testing strategy requirement?",
    },
]

# =============================================================================
# Filler Dialogue Templates
# =============================================================================

FILLER_EXCHANGES = [
    ("Can you explain how that works?", "Sure! The basic concept is..."),
    ("What about error handling?", "Good question. We should implement proper try-catch blocks..."),
    ("How do we handle edge cases?", "For edge cases, I suggest we add validation..."),
    ("Can you show me an example?", "Of course, here's a simple example..."),
    ("What are the alternatives?", "There are several alternatives we could consider..."),
    ("Is there documentation for this?", "Yes, the official documentation covers this in detail..."),
    ("How does this scale?", "In terms of scalability, this approach..."),
    ("What about security?", "Security is important here. We should..."),
    ("Can we optimize this?", "Yes, there are several optimizations we can make..."),
    ("What's the timeline for this?", "Given our resources, I estimate..."),
    ("Are there any risks?", "The main risks to consider are..."),
    ("How do we test this?", "For testing, I recommend..."),
    ("What dependencies does this need?", "We'll need to install several packages..."),
    ("Can you refactor this?", "Sure, here's a cleaner version..."),
    ("What's the best practice here?", "The industry best practice is to..."),
    ("How do we deploy this?", "For deployment, we can use..."),
    ("What about monitoring?", "We should set up monitoring for..."),
    ("Can we automate this?", "Yes, we can automate this using..."),
    ("What's the performance like?", "Performance benchmarks show..."),
    ("How do we maintain this?", "For maintenance, we should..."),
]


# =============================================================================
# Conversation Generator
# =============================================================================

class ConversationGenerator:
    """
    Generates synthetic conversations for evaluation.

    Following the Truncation Simulation design from DOC.md:
    - Turn 1-10: Plant key facts
    - Turn 11-40: Filler dialogue
    - Turn 40: Compression point
    - Turn 41-50: Test questions
    """

    def __init__(
        self,
        total_turns: int = 50,
        fact_turns: List[int] = None,
        compression_turn: int = 40,
        seed: Optional[int] = None,
    ):
        self.total_turns = total_turns
        self.fact_turns = fact_turns or [3, 5, 7, 9]
        self.compression_turn = compression_turn
        self.rng = random.Random(seed)

    def generate(
        self,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        num_facts: int = 4,
        include_distractors: bool = False,
    ) -> SyntheticConversation:
        """Generate a single synthetic conversation."""
        conversation_id = str(uuid.uuid4())[:8]
        turns: List[ConversationTurn] = []
        planted_facts: List[PlantedFact] = []

        # Select fact templates
        all_templates = (
            DECISION_TEMPLATES + KEY_FACT_TEMPLATES + REMINDER_TEMPLATES
        )
        selected_templates = self.rng.sample(
            all_templates, min(num_facts, len(all_templates))
        )

        # Map fact turns to templates
        fact_schedule = dict(zip(self.fact_turns[:num_facts], selected_templates))

        # Distractor turns (if adversarial mode)
        distractor_turns = []
        if include_distractors:
            # Place distractors after facts but before compression
            distractor_turns = self.rng.sample(
                range(max(self.fact_turns) + 5, self.compression_turn - 5),
                min(num_facts, 3),
            )

        # Generate turns
        for turn_id in range(1, self.total_turns + 1):
            if turn_id in fact_schedule:
                # Generate a fact turn
                template = fact_schedule[turn_id]
                turn, fact = self._generate_fact_turn(
                    turn_id, template, difficulty
                )
                turns.append(turn)
                planted_facts.append(fact)

            elif turn_id in distractor_turns and planted_facts:
                # Generate a distractor turn
                target_fact = self.rng.choice(planted_facts)
                turn = self._generate_distractor_turn(turn_id, target_fact)
                turns.append(turn)
                target_fact.distractors.append(turn.assistant)

            else:
                # Generate filler turn
                turn = self._generate_filler_turn(turn_id)
                turns.append(turn)

        return SyntheticConversation(
            id=conversation_id,
            turns=turns,
            planted_facts=planted_facts,
            metadata={
                "total_turns": self.total_turns,
                "fact_turns": self.fact_turns[:num_facts],
                "compression_turn": self.compression_turn,
                "difficulty": difficulty.value,
                "has_distractors": include_distractors,
            },
        )

    def _generate_fact_turn(
        self,
        turn_id: int,
        template: dict,
        difficulty: DifficultyLevel,
    ) -> Tuple[ConversationTurn, PlantedFact]:
        """Generate a turn containing a planted fact."""
        value = self.rng.choice(template["values"])
        user_msg = template["user"]
        assistant_msg = template["assistant"].format(value=value)

        # For harder difficulties, embed the fact less obviously
        if difficulty == DifficultyLevel.HARD:
            # Add more context around the fact
            prefix = self.rng.choice([
                "Based on our discussion, ",
                "Considering all factors, ",
                "After careful analysis, ",
            ])
            suffix = self.rng.choice([
                " This aligns with our overall strategy.",
                " Let me know if you want to explore alternatives.",
                " We can revisit this decision later if needed.",
            ])
            assistant_msg = prefix + assistant_msg + suffix

        fact_id = str(uuid.uuid4())[:8]
        turn = ConversationTurn(
            turn_id=turn_id,
            user=user_msg,
            assistant=assistant_msg,
            contains_fact=fact_id,
        )

        # Determine fact type from template
        if "decision" in template.get("topic", "") or "use" in template["user"].lower():
            fact_type = "decision"
        elif "reminder" in template.get("topic", "") or "prefer" in template["user"].lower():
            fact_type = "reminder"
        else:
            fact_type = "key_fact"

        fact = PlantedFact(
            id=fact_id,
            turn_id=turn_id,
            fact_type=fact_type,
            content=f"{template['topic']}: {value}",
            quote=assistant_msg,
            difficulty=difficulty,
            test_question=template["question"],
            ground_truth=value,
        )

        return turn, fact

    def _generate_distractor_turn(
        self,
        turn_id: int,
        target_fact: PlantedFact,
    ) -> ConversationTurn:
        """Generate a turn with contradicting information (adversarial)."""
        # Create a contradicting statement
        user_msg = f"Wait, I'm confused about the {target_fact.fact_type}..."

        # Generate wrong answer
        wrong_values = [
            "something completely different",
            "the opposite approach",
            "a different solution",
        ]
        wrong_value = self.rng.choice(wrong_values)

        assistant_msg = (
            f"You might be thinking of {wrong_value}, but let me clarify... "
            f"Actually, there was some earlier discussion about alternatives. "
            f"However, we should stick with our original decision."
        )

        return ConversationTurn(
            turn_id=turn_id,
            user=user_msg,
            assistant=assistant_msg,
            is_distractor=True,
        )

    def _generate_filler_turn(self, turn_id: int) -> ConversationTurn:
        """Generate a filler turn (no important facts)."""
        user_msg, assistant_base = self.rng.choice(FILLER_EXCHANGES)

        # Expand assistant response
        assistant_msg = assistant_base + " " + self.rng.choice([
            "This is a common pattern in software development.",
            "Let me know if you need more details.",
            "We can discuss this further if needed.",
            "This should work well for our use case.",
            "I can provide more examples if helpful.",
        ])

        return ConversationTurn(
            turn_id=turn_id,
            user=user_msg,
            assistant=assistant_msg,
        )


# =============================================================================
# Dataset Generation
# =============================================================================

@dataclass
class EvaluationDataset:
    """A complete evaluation dataset."""
    conversations: List[SyntheticConversation]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "conversations": [c.to_dict() for c in self.conversations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationDataset":
        conversations = [
            SyntheticConversation.from_dict(c)
            for c in data["conversations"]
        ]
        return cls(
            conversations=conversations,
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        """Save dataset to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved {len(self.conversations)} conversations to {path}")

    @classmethod
    def load(cls, path: str) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_dataset(
    num_conversations: int = 100,
    difficulty_distribution: Dict[DifficultyLevel, float] = None,
    seed: int = 42,
    max_workers: int = 10,
) -> EvaluationDataset:
    """
    Generate a complete evaluation dataset.

    Args:
        num_conversations: Number of conversations to generate
        difficulty_distribution: Distribution of difficulties (default: balanced)
        seed: Random seed for reproducibility
        max_workers: Number of parallel workers (default 10)

    Returns:
        EvaluationDataset with generated conversations
    """
    if difficulty_distribution is None:
        difficulty_distribution = {
            DifficultyLevel.EASY: 0.25,
            DifficultyLevel.MEDIUM: 0.35,
            DifficultyLevel.HARD: 0.25,
            DifficultyLevel.ADVERSARIAL: 0.15,
        }

    conversations = [None] * num_conversations

    def generate_one(idx: int) -> Tuple[int, SyntheticConversation]:
        # Independent RNG per worker for deterministic sampling
        rng = random.Random(seed + idx)
        difficulty = rng.choices(
            list(difficulty_distribution.keys()),
            weights=list(difficulty_distribution.values()),
        )[0]

        include_distractors = difficulty == DifficultyLevel.ADVERSARIAL

        generator = ConversationGenerator(seed=seed + idx)
        conv = generator.generate(
            difficulty=difficulty,
            num_facts=4,
            include_distractors=include_distractors,
        )
        return idx, conv

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_one, i): i for i in range(num_conversations)
        }

        completed = 0
        for future in as_completed(futures):
            idx, conv = future.result()
            conversations[idx] = conv
            completed += 1
            print(f"Generated conversation {completed}/{num_conversations}: {conv.id}")

    return EvaluationDataset(
        conversations=conversations,
        metadata={
            "num_conversations": num_conversations,
            "difficulty_distribution": {
                k.value: v for k, v in difficulty_distribution.items()
            },
            "seed": seed,
            "max_workers": max_workers,
            "generator_version": "1.0",
        },
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic conversations for CogCanvas evaluation"
    )
    parser.add_argument(
        "--output", "-o",
        default="experiments/data/eval_set.json",
        help="Output file path",
    )
    parser.add_argument(
        "--num-conversations", "-n",
        type=int,
        default=100,
        help="Number of conversations to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview one conversation without saving",
    )

    args = parser.parse_args()

    if args.preview:
        # Generate and preview one conversation
        generator = ConversationGenerator(seed=args.seed)
        conv = generator.generate(
            difficulty=DifficultyLevel.MEDIUM,
            num_facts=4,
        )

        print("=" * 60)
        print(f"CONVERSATION {conv.id}")
        print("=" * 60)

        for turn in conv.turns[:15]:  # Show first 15 turns
            print(f"\n[Turn {turn.turn_id}]")
            print(f"User: {turn.user}")
            print(f"Assistant: {turn.assistant[:200]}...")
            if turn.contains_fact:
                print(f"  *** CONTAINS FACT: {turn.contains_fact} ***")

        print("\n" + "=" * 60)
        print("PLANTED FACTS:")
        print("=" * 60)
        for fact in conv.planted_facts:
            print(f"\n- [{fact.fact_type}] {fact.content}")
            print(f"  Turn: {fact.turn_id}")
            print(f"  Question: {fact.test_question}")
            print(f"  Answer: {fact.ground_truth}")
    else:
        # Generate full dataset
        print(f"Generating {args.num_conversations} conversations...")
        dataset = generate_dataset(
            num_conversations=args.num_conversations,
            seed=args.seed,
            max_workers=args.max_workers,
        )
        dataset.save(args.output)

        # Print summary
        print("\nDataset Summary:")
        print(f"  Total conversations: {len(dataset.conversations)}")
        total_facts = sum(len(c.planted_facts) for c in dataset.conversations)
        print(f"  Total planted facts: {total_facts}")
        print(f"  Seed: {args.seed}")


if __name__ == "__main__":
    main()
