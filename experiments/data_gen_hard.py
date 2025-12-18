#!/usr/bin/env python3
"""
Hard Mode Dataset Generator for CogCanvas Evaluation.

Generates synthetic conversations with:
1. Planted Facts/Decisions at specific turns (using SAME structure as data_gen.py)
2. Distractors ~20 turns later that contain confusing keywords
   but semantically CONFIRM the original decision

Purpose: Test if CogCanvas can distinguish 'decision' from 'discussion'
         and avoid retriever confusion from keyword overlap.

Output format matches eval_set.json for compatibility with the evaluation runner.
"""

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum


# Import shared types from data_gen for consistency
class DifficultyLevel(Enum):
    """Difficulty levels for fact planting."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


@dataclass
class PlantedFact:
    """A key fact planted in the conversation for later testing."""
    id: str
    turn_id: int
    fact_type: str
    content: str
    quote: str
    difficulty: DifficultyLevel
    test_question: str
    ground_truth: str
    distractors: List[str] = field(default_factory=list)
    distractor_turn_id: Optional[int] = None  # Turn where distractor appears


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: int
    user: str
    assistant: str
    contains_fact: Optional[str] = None
    is_distractor: bool = False
    distractor_for_fact: Optional[str] = None  # ID of the fact this distracts from


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


# ============================================
# Decision Templates with Distractors
# ============================================

HARD_MODE_TEMPLATES = [
    {
        "topic": "database",
        "decision": {
            "user": "What database should we use for this project?",
            "assistant": "I recommend PostgreSQL for this project. It offers robust ACID compliance, excellent JSON support, and scales well for our use case.",
        },
        "values": ["PostgreSQL"],
        "question": "What database was chosen for the project?",
        "distractor": {
            "user": "I heard MySQL is pretty popular too. Should we consider switching to MySQL?",
            "assistant": "MySQL is indeed popular and has its strengths. However, we've already committed to PostgreSQL for good reasons - its advanced features like JSONB, better concurrency handling, and richer indexing options. Let's stick with our PostgreSQL decision.",
            "confusing_keywords": ["MySQL", "database", "switch", "consider"]
        }
    },
    {
        "topic": "framework",
        "decision": {
            "user": "Which web framework should we go with?",
            "assistant": "Let's use FastAPI. It's modern, async-first, has automatic OpenAPI docs, and excellent type hints support with Pydantic.",
        },
        "values": ["FastAPI"],
        "question": "What web framework was selected for the API?",
        "distractor": {
            "user": "My colleague mentioned Django is more mature. Should we switch to Django instead?",
            "assistant": "Django is certainly mature and battle-tested. But FastAPI fits our async-heavy microservices architecture better. We made the right call with FastAPI - let's continue with it.",
            "confusing_keywords": ["Django", "framework", "switch", "mature"]
        }
    },
    {
        "topic": "auth",
        "decision": {
            "user": "How should we handle authentication?",
            "assistant": "We should implement JWT-based authentication. It's stateless, works great for APIs, and scales horizontally without session storage concerns.",
        },
        "values": ["JWT-based authentication"],
        "question": "What authentication method was decided upon?",
        "distractor": {
            "user": "What about using session-based auth? Some say sessions are more secure than JWT.",
            "assistant": "Session-based auth does have security benefits in some scenarios. However, for our distributed microservices, JWT remains the better choice. Sessions would require shared state or sticky sessions. We'll stick with JWT as planned.",
            "confusing_keywords": ["session", "auth", "secure", "JWT alternative"]
        }
    },
    {
        "topic": "hosting",
        "decision": {
            "user": "Which cloud provider should we deploy to?",
            "assistant": "AWS is our best choice. Our team has deep AWS expertise, and their managed services like RDS, ECS, and Lambda align with our architecture.",
        },
        "values": ["AWS"],
        "question": "What hosting platform was chosen?",
        "distractor": {
            "user": "GCP has some cool features. Would it make sense to migrate to Google Cloud?",
            "assistant": "GCP does have impressive offerings, especially BigQuery and their ML services. But migrating would be costly and our AWS infrastructure is working well. Let's stay on AWS.",
            "confusing_keywords": ["GCP", "Google Cloud", "migrate", "cloud provider"]
        }
    },
    {
        "topic": "frontend",
        "decision": {
            "user": "What frontend framework should we use?",
            "assistant": "React with TypeScript. It has the largest ecosystem, our team knows it well, and TypeScript will help us maintain code quality at scale.",
        },
        "values": ["React with TypeScript"],
        "question": "What frontend framework was selected?",
        "distractor": {
            "user": "Vue.js seems simpler and has a gentler learning curve. Should we consider Vue instead?",
            "assistant": "Vue is elegant and developer-friendly. However, React's ecosystem and our team's expertise make it the pragmatic choice. We're committed to React + TypeScript.",
            "confusing_keywords": ["Vue", "Vue.js", "frontend", "simpler"]
        }
    },
    {
        "topic": "testing",
        "decision": {
            "user": "What testing strategy should we adopt?",
            "assistant": "We'll use pytest with 80% coverage target. Focus on unit tests for business logic, integration tests for APIs, and a few E2E tests for critical paths.",
        },
        "values": ["pytest with 80% coverage target"],
        "question": "What was the testing strategy requirement?",
        "distractor": {
            "user": "unittest is built into Python. Why not use unittest instead of pytest?",
            "assistant": "unittest works fine for basic cases. But pytest's fixture system, parametrization, and plugin ecosystem make it far more productive. Our pytest decision stands.",
            "confusing_keywords": ["unittest", "testing", "built-in", "pytest alternative"]
        }
    },
    {
        "topic": "containerization",
        "decision": {
            "user": "Should we containerize our services?",
            "assistant": "Yes, we'll use Docker containers orchestrated by Kubernetes. This gives us consistent environments, easy scaling, and deployment flexibility.",
        },
        "values": ["Docker with Kubernetes"],
        "question": "What containerization approach was decided?",
        "distractor": {
            "user": "Docker Swarm is simpler than Kubernetes. Should we use Swarm instead?",
            "assistant": "Docker Swarm is indeed simpler to set up. However, Kubernetes' advanced features like auto-scaling, rolling updates, and service mesh support justify the complexity. We're going with K8s.",
            "confusing_keywords": ["Swarm", "Docker Swarm", "simpler", "orchestration"]
        }
    },
    {
        "topic": "messaging",
        "decision": {
            "user": "We need async messaging between services. What should we use?",
            "assistant": "RabbitMQ is perfect for our needs. It's reliable, supports multiple messaging patterns, and has excellent Python client libraries.",
        },
        "values": ["RabbitMQ"],
        "question": "What messaging system was selected?",
        "distractor": {
            "user": "Kafka seems more scalable. Should we switch to Apache Kafka?",
            "assistant": "Kafka excels at high-throughput streaming scenarios. For our request-response and task queue patterns, RabbitMQ is actually better suited. Let's keep RabbitMQ.",
            "confusing_keywords": ["Kafka", "Apache Kafka", "streaming", "switch"]
        }
    },
    {
        "topic": "cache",
        "decision": {
            "user": "What should we use for caching?",
            "assistant": "Redis is our go-to. It supports various data structures, has pub/sub capabilities, and can serve as both cache and session store if needed.",
        },
        "values": ["Redis"],
        "question": "What caching solution was selected?",
        "distractor": {
            "user": "Memcached is lighter weight than Redis. Should we consider Memcached?",
            "assistant": "Memcached is simpler and slightly faster for basic key-value caching. But Redis's versatility with sorted sets, lists, and pub/sub makes it more valuable for us. Redis stays.",
            "confusing_keywords": ["Memcached", "lighter", "caching", "faster"]
        }
    },
    {
        "topic": "api_style",
        "decision": {
            "user": "REST or GraphQL for our API?",
            "assistant": "REST with OpenAPI spec. It's simpler, well-understood, and our frontend needs are straightforward. GraphQL would be overkill.",
        },
        "values": ["REST with OpenAPI spec"],
        "question": "What API style was chosen?",
        "distractor": {
            "user": "GraphQL would reduce over-fetching. Should we migrate to GraphQL?",
            "assistant": "GraphQL does solve over-fetching elegantly. But our REST endpoints are well-designed, and adding GraphQL complexity isn't worth it for our use case. REST remains our choice.",
            "confusing_keywords": ["GraphQL", "over-fetching", "migrate", "API"]
        }
    },
    {
        "topic": "logging",
        "decision": {
            "user": "How should we handle logging and monitoring?",
            "assistant": "We'll use the ELK stack - Elasticsearch for storage, Logstash for processing, Kibana for visualization. It's proven and our ops team knows it well.",
        },
        "values": ["ELK stack"],
        "question": "What logging solution was selected?",
        "distractor": {
            "user": "Datadog seems easier to set up than ELK. Should we switch to Datadog?",
            "assistant": "Datadog is definitely more turnkey and has great integrations. However, the ELK stack gives us more control and avoids vendor lock-in costs. We'll continue with ELK.",
            "confusing_keywords": ["Datadog", "monitoring", "easier", "switch"]
        }
    },
    {
        "topic": "ci_cd",
        "decision": {
            "user": "What should we use for CI/CD?",
            "assistant": "GitHub Actions. It's tightly integrated with our repo, has great marketplace actions, and the pricing is reasonable for our scale.",
        },
        "values": ["GitHub Actions"],
        "question": "What CI/CD platform was chosen?",
        "distractor": {
            "user": "Jenkins gives more control. Should we consider setting up Jenkins instead?",
            "assistant": "Jenkins is highly customizable, true. But maintaining a Jenkins server is overhead we don't need. GitHub Actions handles our workflows perfectly. Let's stick with it.",
            "confusing_keywords": ["Jenkins", "CI/CD", "control", "customizable"]
        }
    },
    {
        "topic": "orm",
        "decision": {
            "user": "Should we use an ORM or raw SQL?",
            "assistant": "SQLAlchemy ORM for most operations, with the option for raw SQL when we need performance. Best of both worlds.",
        },
        "values": ["SQLAlchemy ORM"],
        "question": "What ORM was selected?",
        "distractor": {
            "user": "Tortoise ORM is async-native. Should we use Tortoise instead of SQLAlchemy?",
            "assistant": "Tortoise ORM's async design is appealing. However, SQLAlchemy 2.0 now has good async support, and its maturity and features outweigh Tortoise's async advantages. We're keeping SQLAlchemy.",
            "confusing_keywords": ["Tortoise", "ORM", "async", "native"]
        }
    },
    {
        "topic": "versioning",
        "decision": {
            "user": "How should we version our API?",
            "assistant": "URL path versioning like /api/v1/, /api/v2/. It's explicit, easy to route, and clear for consumers.",
        },
        "values": ["URL path versioning"],
        "question": "What API versioning strategy was chosen?",
        "distractor": {
            "user": "Header-based versioning is cleaner. Should we switch to Accept header versioning?",
            "assistant": "Header versioning does keep URLs cleaner. But URL versioning is more discoverable and easier to debug. Our path-based approach is the pragmatic choice.",
            "confusing_keywords": ["header", "Accept header", "versioning", "cleaner"]
        }
    },
    {
        "topic": "secret_management",
        "decision": {
            "user": "How do we manage secrets in production?",
            "assistant": "AWS Secrets Manager. It integrates well with our AWS infrastructure, provides rotation, and has fine-grained IAM access control.",
        },
        "values": ["AWS Secrets Manager"],
        "question": "What secret management solution was chosen?",
        "distractor": {
            "user": "HashiCorp Vault is more feature-rich. Should we migrate to Vault?",
            "assistant": "Vault is incredibly powerful with dynamic secrets and detailed audit logs. But it's also complex to operate. AWS Secrets Manager meets our needs without operational overhead. We'll stay with it.",
            "confusing_keywords": ["Vault", "HashiCorp", "secrets", "migrate"]
        }
    },
    {
        "topic": "search",
        "decision": {
            "user": "We need full-text search. What should we use?",
            "assistant": "Elasticsearch. It's battle-tested, has excellent query capabilities, and we're already using it for logs so we have operational experience.",
        },
        "values": ["Elasticsearch"],
        "question": "What search solution was selected?",
        "distractor": {
            "user": "Meilisearch is faster and simpler. Should we switch to Meilisearch?",
            "assistant": "Meilisearch is impressive for developer experience and speed. But Elasticsearch's advanced features and our existing familiarity make it the better choice. We're committed to Elasticsearch.",
            "confusing_keywords": ["Meilisearch", "search", "faster", "simpler"]
        }
    },
    {
        "topic": "time_series",
        "decision": {
            "user": "We need to store metrics time series data. What database?",
            "assistant": "InfluxDB. It's purpose-built for time series, has good retention policies, and integrates well with Grafana for visualization.",
        },
        "values": ["InfluxDB"],
        "question": "What time series database was chosen?",
        "distractor": {
            "user": "TimescaleDB works on top of PostgreSQL. Should we use TimescaleDB instead?",
            "assistant": "TimescaleDB's PostgreSQL compatibility is attractive. However, InfluxDB's query language and specialized optimizations work better for our metrics use case. Let's continue with InfluxDB.",
            "confusing_keywords": ["TimescaleDB", "PostgreSQL", "time series", "compatible"]
        }
    },
    {
        "topic": "package_manager",
        "decision": {
            "user": "Which Python package manager should we standardize on?",
            "assistant": "Poetry. It handles dependencies, virtual environments, and publishing in one tool. The lockfile ensures reproducible builds.",
        },
        "values": ["Poetry"],
        "question": "What Python package manager was selected?",
        "distractor": {
            "user": "pip-tools is simpler and more standard. Should we just use pip with pip-tools?",
            "assistant": "pip-tools is lightweight and widely compatible. But Poetry's unified workflow and better dependency resolution save us time. We're sticking with Poetry.",
            "confusing_keywords": ["pip", "pip-tools", "simpler", "standard"]
        }
    },
    {
        "topic": "feature_flags",
        "decision": {
            "user": "How should we implement feature flags?",
            "assistant": "LaunchDarkly. It's feature-rich, has great SDKs, and provides targeting, A/B testing, and gradual rollouts out of the box.",
        },
        "values": ["LaunchDarkly"],
        "question": "What feature flag solution was chosen?",
        "distractor": {
            "user": "Unleash is open source and self-hostable. Should we switch to Unleash?",
            "assistant": "Unleash being open source is appealing for cost and control. But LaunchDarkly's managed service, analytics, and enterprise features justify the cost for us. We'll keep LaunchDarkly.",
            "confusing_keywords": ["Unleash", "open source", "feature flags", "self-host"]
        }
    },
    {
        "topic": "task_queue",
        "decision": {
            "user": "What should we use for background task processing?",
            "assistant": "Celery with Redis as the broker. It's mature, well-documented, and handles our async task needs perfectly.",
        },
        "values": ["Celery with Redis broker"],
        "question": "What task queue was selected?",
        "distractor": {
            "user": "Dramatiq seems more modern than Celery. Should we consider Dramatiq?",
            "assistant": "Dramatiq has nice defaults and simpler API. But Celery's extensive documentation, monitoring tools, and our team's experience make it the safer choice. We're committed to Celery.",
            "confusing_keywords": ["Dramatiq", "task queue", "modern", "simpler"]
        }
    },
]

# Filler conversation templates (expanded for variety)
FILLER_EXCHANGES = [
    ("Can you explain how that component works?", "Sure! Let me walk you through the architecture and how the pieces fit together. The main idea is to keep components loosely coupled."),
    ("What's the status on the current sprint?", "We're on track. The main features are code complete and we're in testing phase. Should be ready for review by end of week."),
    ("Can you review this pull request?", "I'll take a look at the PR. Send me the link and I'll provide detailed feedback on the implementation."),
    ("How should we structure the unit tests?", "Let's follow the Arrange-Act-Assert pattern and group tests by functionality. This makes tests more readable and maintainable."),
    ("What's the best way to handle errors here?", "We should use try-catch blocks with specific exception types and proper logging. Don't catch generic exceptions."),
    ("Can you help debug this issue?", "Of course. Let's start by looking at the logs and stack trace. Can you reproduce it consistently?"),
    ("How do we deploy the new changes?", "Follow the standard CI/CD pipeline. Merge to main and it auto-deploys to staging first for validation."),
    ("What documentation do we need?", "At minimum, API docs, setup guide, and architecture overview. Let's also add runbook for operations."),
    ("Can you explain the data model?", "The core entities are Users, Projects, and Tasks with relationships defined in the schema. Let me draw it out."),
    ("What security measures should we implement?", "Input validation, parameterized queries, HTTPS everywhere, and proper auth checks. Standard OWASP guidelines."),
    ("How do we handle rate limiting?", "We'll use token bucket algorithm with Redis for distributed state. This handles bursty traffic well."),
    ("What's the plan for the next milestone?", "Focus on performance optimization and adding the reporting module. Then we'll tackle the mobile API."),
    ("Can you explain the authentication flow?", "User submits credentials, we validate against the store, issue tokens, and client stores them securely."),
    ("How should we handle database migrations?", "Use Alembic for schema changes with proper versioning and rollback support. Always test migrations first."),
    ("What metrics should we track?", "Request latency, error rates, throughput, and resource utilization. Also business metrics like daily active users."),
    ("How do we handle concurrent requests?", "Our framework handles concurrency with async handlers. We use connection pooling for database access."),
    ("What about data validation?", "Pydantic models for request/response validation. Schema validation catches most issues at the boundary."),
    ("Can you optimize this query?", "Let me analyze the execution plan. We might need an index here, or we could restructure the join."),
    ("How do we handle file uploads?", "Stream directly to object storage, don't buffer in memory. Set reasonable size limits and validate file types."),
    ("What's the retry strategy?", "Exponential backoff with jitter, max 3 retries. Circuit breaker pattern for external services."),
    ("How do we handle timezones?", "Store everything in UTC internally. Convert to user timezone only at display time."),
    ("Can you review the API design?", "The endpoints look RESTful. A few suggestions on naming conventions and response structure."),
    ("What about backwards compatibility?", "We version the API. Deprecate endpoints gradually with migration guides for clients."),
    ("How do we handle long-running tasks?", "Queue them for background processing. Return a job ID immediately and let clients poll for status."),
    ("What's our backup strategy?", "Daily automated backups to a separate region. Weekly restore tests to verify integrity."),
    ("How do we handle feature toggles?", "Configuration-driven toggles. Can enable/disable features without deployment."),
    ("Can you explain the caching strategy?", "Cache at multiple levels - CDN for static assets, Redis for computed data, query cache for database."),
    ("What about data privacy?", "GDPR compliance, data encryption at rest and in transit, audit logging for sensitive operations."),
    ("How do we handle schema evolution?", "Backwards-compatible changes when possible. For breaking changes, version the schema."),
    ("What's the incident response process?", "On-call rotation, runbooks for common issues, post-mortems for significant incidents."),
]


@dataclass
class HardModeConfig:
    """Configuration for hard mode data generation."""
    num_conversations: int = 20
    facts_per_conversation: int = 4
    turns_before_first_fact: Tuple[int, int] = (2, 5)
    turns_between_facts: Tuple[int, int] = (3, 6)
    turns_between_fact_and_distractor: int = 20
    turns_after_last_distractor: Tuple[int, int] = (3, 8)
    output_path: str = "experiments/data/eval_set_hard.json"
    seed: int = 42


class HardModeGenerator:
    """Generate hard mode evaluation datasets with distractors."""

    def __init__(self, config: HardModeConfig = None):
        self.config = config or HardModeConfig()
        self.rng = random.Random(self.config.seed)
        self.fillers = FILLER_EXCHANGES.copy()
        self.templates = HARD_MODE_TEMPLATES.copy()

    def _get_random_filler(self) -> Tuple[str, str]:
        """Get a random filler conversation (with replacement but tracking to reduce repetition)."""
        return self.rng.choice(self.fillers)

    def _generate_fact_turn(
        self,
        turn_id: int,
        template: dict,
    ) -> Tuple[ConversationTurn, PlantedFact]:
        """Generate a turn containing a planted fact."""
        fact_id = str(uuid.uuid4())[:8]
        user_msg = template["decision"]["user"]
        assistant_msg = template["decision"]["assistant"]
        ground_truth = template["values"][0]

        turn = ConversationTurn(
            turn_id=turn_id,
            user=user_msg,
            assistant=assistant_msg,
            contains_fact=fact_id,
            is_distractor=False,
        )

        fact = PlantedFact(
            id=fact_id,
            turn_id=turn_id,
            fact_type="decision",
            content=f"{template['topic']}: {ground_truth}",
            quote=assistant_msg,
            difficulty=DifficultyLevel.HARD,
            test_question=template["question"],
            ground_truth=ground_truth,
        )

        return turn, fact

    def _generate_distractor_turn(
        self,
        turn_id: int,
        template: dict,
        fact: PlantedFact,
    ) -> ConversationTurn:
        """Generate a distractor turn that CONFIRMS the original decision."""
        user_msg = template["distractor"]["user"]
        assistant_msg = template["distractor"]["assistant"]

        # Record the distractor in the fact
        fact.distractors.append(assistant_msg)
        fact.distractor_turn_id = turn_id

        return ConversationTurn(
            turn_id=turn_id,
            user=user_msg,
            assistant=assistant_msg,
            contains_fact=None,
            is_distractor=True,
            distractor_for_fact=fact.id,
        )

    def _generate_filler_turn(self, turn_id: int, used_fillers: set) -> ConversationTurn:
        """Generate a filler turn, trying to avoid repetition."""
        # Try to get an unused filler
        available = [f for i, f in enumerate(self.fillers) if i not in used_fillers]
        if not available:
            # Reset if we've used all fillers
            used_fillers.clear()
            available = self.fillers

        idx = self.fillers.index(self.rng.choice(available))
        used_fillers.add(idx)
        user_msg, assistant_msg = self.fillers[idx]

        return ConversationTurn(
            turn_id=turn_id,
            user=user_msg,
            assistant=assistant_msg,
            contains_fact=None,
            is_distractor=False,
        )

    def generate_conversation(self, conversation_idx: int) -> SyntheticConversation:
        """Generate a single hard mode conversation.

        Structure:
        1. Some filler turns
        2. Fact 1 planted
        3. More fillers
        4. Fact 2 planted
        5. ... (repeat for all facts)
        6. ~20 turns of fillers
        7. Distractor for Fact 1
        8. More fillers
        9. Distractor for Fact 2
        10. ... (repeat for all distractors)
        11. Final filler turns
        """
        conversation_id = f"hard_{conversation_idx:03d}"
        turns: List[ConversationTurn] = []
        planted_facts: List[PlantedFact] = []
        used_fillers: set = set()

        # Select templates for this conversation
        num_facts = min(self.config.facts_per_conversation, len(self.templates))
        selected_templates = self.rng.sample(self.templates, num_facts)

        current_turn = 1
        fact_turns = []
        distractor_turns = []

        # Phase 1: Plant facts with fillers between them
        for i, template in enumerate(selected_templates):
            # Add fillers before this fact
            if i == 0:
                num_fillers = self.rng.randint(*self.config.turns_before_first_fact)
            else:
                num_fillers = self.rng.randint(*self.config.turns_between_facts)

            for _ in range(num_fillers):
                turn = self._generate_filler_turn(current_turn, used_fillers)
                turns.append(turn)
                current_turn += 1

            # Plant the fact
            turn, fact = self._generate_fact_turn(current_turn, template)
            turns.append(turn)
            planted_facts.append(fact)
            fact_turns.append(current_turn)
            current_turn += 1

        # Phase 2: Add filler turns before distractors (~20 turns)
        filler_count = self.config.turns_between_fact_and_distractor
        for _ in range(filler_count):
            turn = self._generate_filler_turn(current_turn, used_fillers)
            turns.append(turn)
            current_turn += 1

        # Phase 3: Plant distractors (one for each fact)
        for i, (template, fact) in enumerate(zip(selected_templates, planted_facts)):
            # Add a couple fillers between distractors
            if i > 0:
                num_fillers = self.rng.randint(1, 3)
                for _ in range(num_fillers):
                    turn = self._generate_filler_turn(current_turn, used_fillers)
                    turns.append(turn)
                    current_turn += 1

            # Plant the distractor
            turn = self._generate_distractor_turn(current_turn, template, fact)
            turns.append(turn)
            distractor_turns.append(current_turn)
            current_turn += 1

        # Phase 4: Final filler turns
        num_final_fillers = self.rng.randint(*self.config.turns_after_last_distractor)
        for _ in range(num_final_fillers):
            turn = self._generate_filler_turn(current_turn, used_fillers)
            turns.append(turn)
            current_turn += 1

        return SyntheticConversation(
            id=conversation_id,
            turns=turns,
            planted_facts=planted_facts,
            metadata={
                "total_turns": len(turns),
                "fact_turns": fact_turns,
                "distractor_turns": distractor_turns,
                "turns_between_fact_and_distractor": self.config.turns_between_fact_and_distractor,
                "difficulty": "hard",
                "has_distractors": True,
                "generator_version": "2.0",
            },
        )

    def generate_dataset(self, max_workers: int = 10) -> List[SyntheticConversation]:
        """Generate full hard mode dataset with parallel execution.

        Args:
            max_workers: Number of parallel workers (default 10; tune down if rate limits hit)
        """
        conversations = [None] * self.config.num_conversations

        def generate_one(idx: int) -> Tuple[int, SyntheticConversation]:
            conv = self.generate_conversation(idx)
            return idx, conv

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_one, i): i
                for i in range(self.config.num_conversations)
            }

            completed = 0
            for future in as_completed(futures):
                idx, conv = future.result()
                conversations[idx] = conv
                completed += 1
                print(f"Generated conversation {completed}/{self.config.num_conversations}: {conv.id}")

        return conversations


@dataclass
class EvaluationDataset:
    """A complete evaluation dataset (same structure as data_gen.py)."""
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


def main():
    """Generate hard mode evaluation dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Hard Mode Evaluation Dataset")
    parser.add_argument("--num-conversations", "-n", type=int, default=20,
                        help="Number of conversations to generate")
    parser.add_argument("--output", "-o", type=str,
                        default="experiments/data/eval_set_hard.json",
                        help="Output path")
    parser.add_argument("--distractor-gap", type=int, default=20,
                        help="Number of turns between facts and distractors")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--facts-per-conversation", type=int, default=4,
                        help="Number of facts to plant per conversation")

    args = parser.parse_args()

    config = HardModeConfig(
        num_conversations=args.num_conversations,
        output_path=args.output,
        turns_between_fact_and_distractor=args.distractor_gap,
        seed=args.seed,
        facts_per_conversation=args.facts_per_conversation,
    )

    generator = HardModeGenerator(config)
    conversations = generator.generate_dataset()

    # Create dataset with metadata
    dataset = EvaluationDataset(
        conversations=conversations,
        metadata={
            "num_conversations": len(conversations),
            "facts_per_conversation": config.facts_per_conversation,
            "distractor_gap": config.turns_between_fact_and_distractor,
            "seed": config.seed,
            "difficulty": "hard",
            "generator_version": "2.0",
        },
    )

    dataset.save(args.output)

    # Print summary
    print("\n" + "="*60)
    print("HARD MODE DATASET SUMMARY")
    print("="*60)
    print(f"Total conversations: {len(conversations)}")
    print(f"Output file: {args.output}")

    total_facts = sum(len(c.planted_facts) for c in conversations)
    total_distractors = sum(
        sum(1 for t in c.turns if t.is_distractor)
        for c in conversations
    )

    print(f"\nTotal planted facts: {total_facts}")
    print(f"Total distractors: {total_distractors}")

    # Show topics covered
    topics = set()
    for conv in conversations:
        for fact in conv.planted_facts:
            topic = fact.content.split(":")[0]
            topics.add(topic)

    print(f"\nTopics covered: {len(topics)}")
    for topic in sorted(topics):
        print(f"  - {topic}")

    print("\nConversation structure:")
    print("  1. Pre-fact filler (2-5 turns)")
    print("  2. FACT 1 planted (with contains_fact field)")
    print("  3. Inter-fact filler (3-6 turns)")
    print("  4. FACT 2, 3, 4... planted")
    print(f"  5. Mid-filler (~{args.distractor_gap} turns)")
    print("  6. DISTRACTOR for Fact 1 (is_distractor=True, confirms original)")
    print("  7. DISTRACTOR for Fact 2, 3, 4...")
    print("  8. Post-distractor filler (3-8 turns)")

    print("\nTest: Can the system retrieve FACT turns, not DISTRACTOR turns?")
    print("Ground truth is in planted_facts[].ground_truth")


if __name__ == "__main__":
    main()
