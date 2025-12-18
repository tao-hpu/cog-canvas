"""
Multi-Hop Benchmark Data Generator for CogCanvas Evaluation.

This generator creates conversations with:
1. Planted facts with EXPLICIT RELATIONS between them
2. Multi-hop questions that require graph traversal to answer
3. Ground truth relation chains for evaluation

The goal is to prove that graph structure matters for certain types of questions,
specifically those requiring:
- Causal reasoning: "Why did we choose X?"
- Impact analysis: "What decisions were affected by Y?"
- Decision chains: "Trace the path from A to B"

Usage:
    python -m experiments.data_gen_multihop --output experiments/data/multihop_eval.json --num_conversations 50
"""

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from enum import Enum


class QuestionType(Enum):
    """Types of multi-hop questions."""
    CAUSAL = "causal"           # Why was X chosen? (needs: X <- caused_by <- [Y, Z])
    IMPACT = "impact"           # What was affected by X? (needs: X -> leads_to -> [A, B])
    CHAIN = "chain"             # Path from X to Y? (needs: X -> ... -> Y)


class RelationType(Enum):
    """Types of relations between facts."""
    CAUSED_BY = "caused_by"     # Decision X was caused by Fact Y
    LEADS_TO = "leads_to"       # Fact X leads to Decision Y
    REQUIRES = "requires"       # Decision X requires Fact Y


@dataclass
class FactRelation:
    """A relation between two facts."""
    source_id: str              # Source fact ID
    target_id: str              # Target fact ID
    relation_type: RelationType
    verbalized: str             # How this relation was expressed in conversation

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "verbalized": self.verbalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FactRelation":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            verbalized=data["verbalized"],
        )


@dataclass
class PlantedFactWithRelations:
    """A fact that can have relations to other facts."""
    id: str
    turn_id: int
    fact_type: str              # "decision", "constraint", "requirement", "outcome"
    content: str                # e.g., "database: PostgreSQL"
    quote: str                  # Full assistant response

    # Relations are stored separately but referenced here
    caused_by: List[str] = field(default_factory=list)  # IDs of facts that caused this
    leads_to: List[str] = field(default_factory=list)   # IDs of facts this leads to

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "fact_type": self.fact_type,
            "content": self.content,
            "quote": self.quote,
            "caused_by": self.caused_by,
            "leads_to": self.leads_to,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlantedFactWithRelations":
        return cls(**data)


@dataclass
class MultiHopQuestion:
    """A question requiring multi-hop reasoning."""
    id: str
    question: str               # The actual question text
    question_type: QuestionType
    required_fact_ids: List[str]  # All facts needed to answer
    required_hops: int          # Number of relation hops needed
    ground_truth_keywords: List[str]  # Keywords expected in answer
    explanation: str            # Why this answer (for debugging)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "question_type": self.question_type.value,
            "required_fact_ids": self.required_fact_ids,
            "required_hops": self.required_hops,
            "ground_truth_keywords": self.ground_truth_keywords,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultiHopQuestion":
        data = data.copy()
        data["question_type"] = QuestionType(data["question_type"])
        return cls(**data)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: int
    user: str
    assistant: str
    contains_fact_id: Optional[str] = None
    contains_relation: Optional[str] = None  # Verbalized relation if any


@dataclass
class MultiHopConversation:
    """A conversation designed for multi-hop testing."""
    id: str
    turns: List[ConversationTurn]
    facts: List[PlantedFactWithRelations]
    relations: List[FactRelation]
    questions: List[MultiHopQuestion]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "turns": [asdict(t) for t in self.turns],
            "facts": [f.to_dict() for f in self.facts],
            "relations": [r.to_dict() for r in self.relations],
            "questions": [q.to_dict() for q in self.questions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultiHopConversation":
        return cls(
            id=data["id"],
            turns=[ConversationTurn(**t) for t in data["turns"]],
            facts=[PlantedFactWithRelations.from_dict(f) for f in data["facts"]],
            relations=[FactRelation.from_dict(r) for r in data["relations"]],
            questions=[MultiHopQuestion.from_dict(q) for q in data["questions"]],
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Templates for Multi-Hop Scenarios
# =============================================================================

# Constraint → Decision patterns
# IMPORTANT: Decision templates DO NOT mention constraint values directly
# This forces multi-hop reasoning - agents must trace back to find the cause
CONSTRAINT_DECISION_TEMPLATES = [
    {
        "constraint": {
            "topic": "budget",
            "user": "What's our infrastructure budget?",
            "assistant": "We have a monthly budget of {value}. This will be a key constraint for our technical decisions.",
            "values": ["$500", "$2,000", "$10,000", "$50,000"],
        },
        "decision": {
            "topic": "hosting",
            "user": "So what's the hosting decision?",
            # NOTE: No mention of budget here - just the decision
            "assistant": "After considering our constraints, I recommend {value} for hosting. It meets our requirements well.",
            "values": {
                "$500": ["DigitalOcean", "Heroku"],
                "$2,000": ["AWS", "Google Cloud"],
                "$10,000": ["AWS", "Azure"],
                "$50,000": ["AWS", "Google Cloud", "Azure"],
            },
        },
        "relation_template": "Our {constraint_value} budget constraint led us to choose {decision_value}.",
    },
    {
        "constraint": {
            "topic": "team_size",
            "user": "How many developers do we have?",
            "assistant": "We have {value} on the team. This will influence our technology choices.",
            "values": ["2 developers", "5 engineers", "a team of 10", "15 people"],
        },
        "decision": {
            "topic": "framework",
            "user": "What framework should we use?",
            # NOTE: No mention of team size here
            "assistant": "I recommend we go with {value} for our backend framework. It's well-suited for our needs.",
            "values": {
                "2 developers": ["Flask", "FastAPI"],
                "5 engineers": ["Django", "FastAPI"],
                "a team of 10": ["Django", "Spring Boot"],
                "15 people": ["Django", "Spring Boot", "Rails"],
            },
        },
        "relation_template": "Having {constraint_value} led us to select {decision_value} for its learning curve.",
    },
    {
        "constraint": {
            "topic": "deadline",
            "user": "When do we need to launch?",
            "assistant": "The deadline is {value}. We need to plan accordingly.",
            "values": ["in 2 weeks", "next month", "end of Q2", "by year end"],
        },
        "decision": {
            "topic": "approach",
            "user": "What development approach should we take?",
            # NOTE: No mention of deadline here
            "assistant": "For our development approach, I suggest we {value}. This aligns with our project goals.",
            "values": {
                "in 2 weeks": ["use existing templates", "go with MVP only"],
                "next month": ["prioritize core features", "use a proven stack"],
                "end of Q2": ["build properly with tests", "follow best practices"],
                "by year end": ["build a robust solution", "include comprehensive testing"],
            },
        },
        "relation_template": "The {constraint_value} deadline requires us to {decision_value}.",
    },
]

# Decision → Outcome patterns
DECISION_OUTCOME_TEMPLATES = [
    {
        "decision": {
            "topic": "database",
            "user": "What database should we use?",
            "assistant": "I recommend {value} for our project. It offers the features we need.",
            "values": ["PostgreSQL", "MongoDB", "MySQL"],
        },
        "outcome": {
            "topic": "schema_approach",
            "user": "How should we structure our data?",
            "assistant": "Since we're using {decision_value}, we should {value}. This aligns with our database choice.",
            "values": {
                "PostgreSQL": ["use strict schemas with migrations", "leverage relational features"],
                "MongoDB": ["use flexible documents", "embrace schema-less design"],
                "MySQL": ["use traditional tables", "follow relational patterns"],
            },
        },
        "relation_template": "Choosing {decision_value} means we will {outcome_value}.",
    },
]

# Filler exchanges
FILLER_EXCHANGES = [
    ("Can you explain that further?", "Sure, let me elaborate on that point..."),
    ("What about error handling?", "For errors, we should implement proper logging..."),
    ("How does this scale?", "In terms of scalability, this approach handles..."),
    ("Any concerns about this?", "The main concern would be..."),
    ("What's the alternative?", "An alternative approach would be..."),
]


# =============================================================================
# Multi-Hop Conversation Generator
# =============================================================================

class MultiHopConversationGenerator:
    """Generates conversations with explicit relations between facts."""

    def __init__(
        self,
        total_turns: int = 50,
        compression_turn: int = 40,
        seed: Optional[int] = None,
    ):
        self.total_turns = total_turns
        self.compression_turn = compression_turn
        self.rng = random.Random(seed)

    def generate(self) -> MultiHopConversation:
        """Generate a conversation with related facts and multi-hop questions."""
        conv_id = str(uuid.uuid4())[:8]
        turns: List[ConversationTurn] = []
        facts: List[PlantedFactWithRelations] = []
        relations: List[FactRelation] = []

        # Select a constraint → decision pattern
        pattern = self.rng.choice(CONSTRAINT_DECISION_TEMPLATES)

        # Generate constraint fact (early turn)
        constraint_turn = 3
        constraint_value = self.rng.choice(pattern["constraint"]["values"])
        constraint_id = str(uuid.uuid4())[:8]

        constraint_fact = PlantedFactWithRelations(
            id=constraint_id,
            turn_id=constraint_turn,
            fact_type="constraint",
            content=f"{pattern['constraint']['topic']}: {constraint_value}",
            quote=pattern["constraint"]["assistant"].format(value=constraint_value),
        )
        facts.append(constraint_fact)

        # Generate decision fact (later turn, but before compression)
        decision_turn = 7
        decision_values = pattern["decision"]["values"].get(constraint_value, ["default option"])
        decision_value = self.rng.choice(decision_values)
        decision_id = str(uuid.uuid4())[:8]

        decision_fact = PlantedFactWithRelations(
            id=decision_id,
            turn_id=decision_turn,
            fact_type="decision",
            content=f"{pattern['decision']['topic']}: {decision_value}",
            quote=pattern["decision"]["assistant"].format(value=decision_value),
            caused_by=[constraint_id],  # Explicit relation
        )
        facts.append(decision_fact)

        # Update constraint's leads_to
        constraint_fact.leads_to.append(decision_id)

        # Create the relation object
        relation_verbalized = pattern["relation_template"].format(
            constraint_value=constraint_value,
            decision_value=decision_value
        )
        relation = FactRelation(
            source_id=constraint_id,
            target_id=decision_id,
            relation_type=RelationType.LEADS_TO,
            verbalized=relation_verbalized,
        )
        relations.append(relation)

        # Optionally add a second constraint → decision pair
        if self.rng.random() > 0.3:  # 70% chance
            pattern2 = self.rng.choice([p for p in CONSTRAINT_DECISION_TEMPLATES if p != pattern])

            constraint2_turn = 5
            constraint2_value = self.rng.choice(pattern2["constraint"]["values"])
            constraint2_id = str(uuid.uuid4())[:8]

            constraint2_fact = PlantedFactWithRelations(
                id=constraint2_id,
                turn_id=constraint2_turn,
                fact_type="constraint",
                content=f"{pattern2['constraint']['topic']}: {constraint2_value}",
                quote=pattern2["constraint"]["assistant"].format(value=constraint2_value),
            )
            facts.append(constraint2_fact)

            decision2_turn = 9
            decision2_values = pattern2["decision"]["values"].get(constraint2_value, ["default"])
            decision2_value = self.rng.choice(decision2_values)
            decision2_id = str(uuid.uuid4())[:8]

            decision2_fact = PlantedFactWithRelations(
                id=decision2_id,
                turn_id=decision2_turn,
                fact_type="decision",
                content=f"{pattern2['decision']['topic']}: {decision2_value}",
                quote=pattern2["decision"]["assistant"].format(value=decision2_value),
                caused_by=[constraint2_id],
            )
            facts.append(decision2_fact)
            constraint2_fact.leads_to.append(decision2_id)

            relation2 = FactRelation(
                source_id=constraint2_id,
                target_id=decision2_id,
                relation_type=RelationType.LEADS_TO,
                verbalized=pattern2["relation_template"].format(
                    constraint_value=constraint2_value,
                    decision_value=decision2_value
                ),
            )
            relations.append(relation2)

        # Sort facts by turn_id
        facts.sort(key=lambda f: f.turn_id)

        # Build turn schedule
        fact_turns = {f.turn_id: f for f in facts}

        # Generate all turns
        for turn_id in range(1, self.total_turns + 1):
            if turn_id in fact_turns:
                fact = fact_turns[turn_id]
                # Find the appropriate pattern
                if fact.fact_type == "constraint":
                    for p in CONSTRAINT_DECISION_TEMPLATES:
                        if p["constraint"]["topic"] in fact.content:
                            turn = ConversationTurn(
                                turn_id=turn_id,
                                user=p["constraint"]["user"],
                                assistant=fact.quote,
                                contains_fact_id=fact.id,
                            )
                            break
                    else:
                        turn = ConversationTurn(
                            turn_id=turn_id,
                            user="What about this constraint?",
                            assistant=fact.quote,
                            contains_fact_id=fact.id,
                        )
                else:  # decision
                    for p in CONSTRAINT_DECISION_TEMPLATES:
                        if p["decision"]["topic"] in fact.content:
                            # Find the causing constraint
                            causing_fact = next((f for f in facts if f.id in fact.caused_by), None)
                            constraint_val = causing_fact.content.split(": ")[1] if causing_fact else "constraints"
                            turn = ConversationTurn(
                                turn_id=turn_id,
                                user=p["decision"]["user"].replace("{constraint_value}", constraint_val),
                                assistant=fact.quote,
                                contains_fact_id=fact.id,
                            )
                            break
                    else:
                        turn = ConversationTurn(
                            turn_id=turn_id,
                            user="What's the decision here?",
                            assistant=fact.quote,
                            contains_fact_id=fact.id,
                        )
            else:
                # Filler turn
                user_msg, assistant_msg = self.rng.choice(FILLER_EXCHANGES)
                turn = ConversationTurn(
                    turn_id=turn_id,
                    user=user_msg,
                    assistant=assistant_msg + " This is common in software development.",
                )
            turns.append(turn)

        # Generate multi-hop questions
        questions = self._generate_questions(facts, relations)

        return MultiHopConversation(
            id=conv_id,
            turns=turns,
            facts=facts,
            relations=relations,
            questions=questions,
            metadata={
                "total_turns": self.total_turns,
                "compression_turn": self.compression_turn,
                "num_facts": len(facts),
                "num_relations": len(relations),
            },
        )

    def _generate_questions(
        self,
        facts: List[PlantedFactWithRelations],
        relations: List[FactRelation],
    ) -> List[MultiHopQuestion]:
        """Generate multi-hop questions based on the planted facts and relations."""
        questions = []

        # Find decision facts (they have caused_by relations)
        decisions = [f for f in facts if f.fact_type == "decision" and f.caused_by]
        constraints = [f for f in facts if f.fact_type == "constraint"]

        for decision in decisions:
            # CAUSAL question: Why was this decision made?
            causing_facts = [f for f in facts if f.id in decision.caused_by]
            if causing_facts:
                decision_value = decision.content.split(": ")[1]
                decision_topic = decision.content.split(": ")[0]

                causal_q = MultiHopQuestion(
                    id=str(uuid.uuid4())[:8],
                    question=f"Why did we choose {decision_value} for {decision_topic}?",
                    question_type=QuestionType.CAUSAL,
                    required_fact_ids=[decision.id] + [f.id for f in causing_facts],
                    required_hops=1,
                    ground_truth_keywords=[
                        decision_value,
                        *[f.content.split(": ")[1] for f in causing_facts]
                    ],
                    explanation=f"Decision '{decision_value}' was caused by: {[f.content for f in causing_facts]}",
                )
                questions.append(causal_q)

        # IMPACT question: What did this constraint affect?
        for constraint in constraints:
            if constraint.leads_to:
                affected_decisions = [f for f in facts if f.id in constraint.leads_to]
                if affected_decisions:
                    constraint_value = constraint.content.split(": ")[1]
                    constraint_topic = constraint.content.split(": ")[0]

                    impact_q = MultiHopQuestion(
                        id=str(uuid.uuid4())[:8],
                        question=f"What decisions were affected by our {constraint_topic} ({constraint_value})?",
                        question_type=QuestionType.IMPACT,
                        required_fact_ids=[constraint.id] + [f.id for f in affected_decisions],
                        required_hops=1,
                        ground_truth_keywords=[
                            constraint_value,
                            *[f.content.split(": ")[1] for f in affected_decisions]
                        ],
                        explanation=f"Constraint '{constraint_value}' affected: {[f.content for f in affected_decisions]}",
                    )
                    questions.append(impact_q)

        return questions


# =============================================================================
# Dataset Generation
# =============================================================================

@dataclass
class MultiHopDataset:
    """A complete multi-hop evaluation dataset."""
    conversations: List[MultiHopConversation]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "conversations": [c.to_dict() for c in self.conversations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultiHopDataset":
        return cls(
            conversations=[MultiHopConversation.from_dict(c) for c in data["conversations"]],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved {len(self.conversations)} multi-hop conversations to {path}")

    @classmethod
    def load(cls, path: str) -> "MultiHopDataset":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_multihop_dataset(
    num_conversations: int = 50,
    seed: int = 42,
    max_workers: int = 10,
) -> MultiHopDataset:
    """Generate a multi-hop evaluation dataset."""
    conversations = [None] * num_conversations

    def generate_one(idx: int) -> Tuple[int, MultiHopConversation]:
        generator = MultiHopConversationGenerator(seed=seed + idx)
        conv = generator.generate()
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

    total_facts = sum(len(c.facts) for c in conversations)
    total_relations = sum(len(c.relations) for c in conversations)
    total_questions = sum(len(c.questions) for c in conversations)

    return MultiHopDataset(
        conversations=conversations,
        metadata={
            "num_conversations": num_conversations,
            "total_facts": total_facts,
            "total_relations": total_relations,
            "total_questions": total_questions,
            "seed": seed,
            "max_workers": max_workers,
            "benchmark_type": "multi-hop",
        },
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multi-hop benchmark for CogCanvas evaluation"
    )
    parser.add_argument(
        "--output", "-o",
        default="experiments/data/multihop_eval.json",
        help="Output file path",
    )
    parser.add_argument(
        "--num-conversations", "-n",
        type=int,
        default=50,
        help="Number of conversations to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview one conversation",
    )

    args = parser.parse_args()

    if args.preview:
        generator = MultiHopConversationGenerator(seed=args.seed)
        conv = generator.generate()

        print("=" * 70)
        print(f"MULTI-HOP CONVERSATION: {conv.id}")
        print("=" * 70)

        print("\n📋 FACTS:")
        for fact in conv.facts:
            print(f"  [{fact.turn_id}] {fact.fact_type}: {fact.content}")
            if fact.caused_by:
                print(f"       ← caused_by: {fact.caused_by}")
            if fact.leads_to:
                print(f"       → leads_to: {fact.leads_to}")

        print("\n🔗 RELATIONS:")
        for rel in conv.relations:
            print(f"  {rel.source_id} --{rel.relation_type.value}--> {rel.target_id}")
            print(f"       \"{rel.verbalized}\"")

        print("\n❓ MULTI-HOP QUESTIONS:")
        for q in conv.questions:
            print(f"\n  [{q.question_type.value}] {q.question}")
            print(f"       Requires {q.required_hops} hop(s)")
            print(f"       Facts needed: {q.required_fact_ids}")
            print(f"       Keywords: {q.ground_truth_keywords}")
            print(f"       Explanation: {q.explanation}")

        print("\n📝 SAMPLE TURNS:")
        for turn in conv.turns[:12]:
            marker = "📌" if turn.contains_fact_id else "  "
            print(f"{marker} [Turn {turn.turn_id}]")
            print(f"   User: {turn.user}")
            print(f"   Asst: {turn.assistant[:100]}...")
    else:
        print(f"Generating {args.num_conversations} multi-hop conversations...")
        dataset = generate_multihop_dataset(
            num_conversations=args.num_conversations,
            seed=args.seed,
            max_workers=args.workers,
        )
        dataset.save(args.output)

        print("\n📊 Dataset Summary:")
        print(f"  Conversations: {len(dataset.conversations)}")
        print(f"  Total facts: {dataset.metadata['total_facts']}")
        print(f"  Total relations: {dataset.metadata['total_relations']}")
        print(f"  Total questions: {dataset.metadata['total_questions']}")


if __name__ == "__main__":
    main()
