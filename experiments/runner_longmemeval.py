"""
LongMemEval Experiment Runner for CogCanvas Evaluation.

This runner evaluates agents on the LongMemEval (ICLR 2025) benchmark.

LongMemEval Characteristics:
- 500 questions, each with its own haystack of ~48 multi-session conversations
- One QA per conversation: 500 conversations x 1 question each
- 5 memory ability categories across 6 question types

Evaluation Strategy:
1. Process conversation turns with rolling/single/dynamic compression
2. Ask the single question associated with each conversation
3. Score using LongMemEval's official LLM Judge binary evaluation

Question Categories (5 memory abilities):
- Category 1: Information Extraction (single-session-user, single-session-assistant, single-session-preference)
- Category 2: Multi-Session Reasoning (multi-session)
- Category 3: Knowledge Update (knowledge-update)
- Category 4: Temporal Reasoning (temporal-reasoning)
- Category 5: Abstention (unanswerable questions)

Scoring:
- LLM Judge binary evaluation (yes/no) using official LongMemEval prompts
- Different judge prompts per question_type
- No F1 fallback (LLM Judge is the official protocol)
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn


# =============================================================================
# Cache Utilities
# =============================================================================

def get_extraction_config_hash(config: dict, extraction_mode: str = "batch") -> str:
    """
    Compute a hash for extraction configuration.

    This hash is used to identify cached Canvas states.
    Only extraction-relevant parameters are included.

    Args:
        config: Agent configuration dict
        extraction_mode: "batch" or "per_turn"

    Returns:
        8-character hex hash
    """
    extraction_config = {
        "extractor_model": config.get("extractor_model", "gpt-4o-mini"),
        "embedding_model": config.get("embedding_model", "bge-large-zh-v1.5"),
        "enable_temporal_heuristic": config.get("enable_temporal_heuristic", True),
        "enable_gleaning": config.get("enable_gleaning", True),
        "extraction_mode": extraction_mode,
        "rolling_interval": config.get("rolling_interval", 40),
        # P1-7 chunks ablation: must not share cache with LLM-extracted artifacts
        "chunks_mode": config.get("chunks_mode", False),
        "chunks_chunk_size": config.get("chunks_chunk_size", 512),
        "chunks_overlap": config.get("chunks_overlap", 100),
        # W4 union storage: chunks+artifacts canvas must not share either cache
        "union_mode": config.get("union_mode", False),
    }
    config_str = json.dumps(extraction_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_cache_path(conv_id: str, config_hash: str) -> Path:
    """Get the cache file path for a LongMemEval conversation.

    Uses a separate namespace from LoCoMo to avoid cache mixing.
    """
    cache_dir = Path("experiments/cache/extraction_longmemeval") / config_hash
    return cache_dir / f"{conv_id}.json"


from experiments.longmemeval_adapter import (
    load_longmemeval,
    convert_to_eval_format,
    LongMemEvalConversation,
    LongMemEvalQAPair,
    CATEGORY_NAMES,
)


# =============================================================================
# LLM Judge Prompts (Official LongMemEval evaluate_qa.py)
# =============================================================================


def get_longmemeval_judge_prompt(
    question_type: str,
    question: str,
    answer: str,
    response: str,
    is_abstention: bool = False,
) -> str:
    """
    Get the official LongMemEval LLM Judge prompt for a given question type.

    These prompts are EXACT copies from the official LongMemEval evaluate_qa.py.
    Each question type uses a tailored evaluation template.

    Args:
        question_type: One of the 6 LongMemEval question types
        question: The original question
        answer: The ground truth answer (or explanation for abstention)
        response: The model's response to judge
        is_abstention: Whether this is an abstention (unanswerable) question

    Returns:
        The formatted judge prompt string

    Raises:
        ValueError: If question_type is not recognized (non-abstention only)
    """
    if not is_abstention:
        if question_type in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}"
                "\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif question_type == 'temporal-reasoning':
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes "
                "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
                "response is still correct. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}"
                "\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif question_type == 'knowledge-update':
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the "
                "required answer."
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}"
                "\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif question_type == 'single-session-preference':
            template = (
                "I will give you a question, a rubric for desired personalized response, and a "
                "response from a model. Please answer yes if the response satisfies the desired "
                "response. Otherwise, answer no. The model does not need to reflect all the points "
                "in the rubric. The response is correct as long as it recalls and utilizes the user's "
                "personal information correctly."
                "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}"
                "\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        else:
            raise ValueError(f"Unknown question_type: {question_type}")
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is "
            "given but the asked information is not."
            "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}"
            "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)


# =============================================================================
# Scoring
# =============================================================================


@dataclass
class LongMemEvalScoreResult:
    """Result of scoring a LongMemEval answer (official LLM Judge binary evaluation)."""

    correct: bool       # Binary: yes/no from LLM judge
    judge_response: str  # Raw judge response
    answer: str         # Model's answer
    ground_truth: str   # Expected answer

    @property
    def passed(self) -> bool:
        """Whether the answer was judged correct."""
        return self.correct

    @property
    def f1_score(self) -> float:
        """Binary score for compatibility with LoCoMo-style aggregation."""
        return 1.0 if self.correct else 0.0

    @property
    def exact_match(self) -> bool:
        """Alias for correct (binary judge = exact match equivalent)."""
        return self.correct


def score_longmemeval_answer(
    answer: str,
    ground_truth: str,
    question: str,
    question_type: str,
    is_abstention: bool,
    client,
    model: str = "gpt-4o-mini",
) -> LongMemEvalScoreResult:
    """
    Score an answer using LongMemEval's official LLM Judge protocol.

    Uses the official evaluation prompts from LongMemEval evaluate_qa.py,
    tailored to each question type. The judge outputs yes/no and we parse
    the binary result.

    Args:
        answer: The model's answer to score
        ground_truth: Expected answer (or rubric/explanation for preference/abstention)
        question: The original question
        question_type: One of the 6 LongMemEval question types
        is_abstention: Whether this is an abstention question
        client: OpenAI-compatible client for the judge LLM
        model: Model to use for judging (default: gpt-4o-mini)

    Returns:
        LongMemEvalScoreResult with binary judgment
    """
    from experiments.llm_utils import call_llm_with_retry

    prompt = get_longmemeval_judge_prompt(
        question_type, question, ground_truth, answer, is_abstention
    )

    try:
        response = call_llm_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
            verbose=False,
            max_retries=5,
        )
        correct = 'yes' in response.strip().lower()
    except Exception as e:
        print(f"[WARN] LLM Judge failed: {e}, marking as incorrect")
        response = f"Error: {type(e).__name__}: {e}"
        correct = False

    return LongMemEvalScoreResult(
        correct=correct,
        judge_response=response,
        answer=answer,
        ground_truth=ground_truth,
    )


# =============================================================================
# Results
# =============================================================================


@dataclass
class LongMemEvalQuestionResult:
    """Result for a single LongMemEval question."""

    question_id: str
    question: str
    question_type: str
    category: int
    category_name: str
    is_abstention: bool
    ground_truth: str
    answer: str
    score: LongMemEvalScoreResult
    latency_ms: float


@dataclass
class LongMemEvalConversationResult:
    """Result for a single LongMemEval conversation (1 question per conversation)."""

    conversation_id: str
    num_turns: int
    compression_turn: int
    question_results: List[LongMemEvalQuestionResult]
    total_time_ms: float

    @property
    def accuracy(self) -> float:
        """Fraction of questions passed."""
        if not self.question_results:
            return 0.0
        return sum(1 for r in self.question_results if r.score.passed) / len(
            self.question_results
        )

    @property
    def exact_match_rate(self) -> float:
        """Fraction of questions with exact match."""
        if not self.question_results:
            return 0.0
        return sum(1 for r in self.question_results if r.score.exact_match) / len(
            self.question_results
        )

    @property
    def avg_f1_score(self) -> float:
        """Average F1 score across all questions (binary for LongMemEval)."""
        if not self.question_results:
            return 0.0
        return sum(r.score.f1_score for r in self.question_results) / len(
            self.question_results
        )

    @property
    def avg_keyword_overlap(self) -> float:
        """Alias for avg_f1_score (backward compatibility)."""
        return self.avg_f1_score

    def accuracy_by_category(self, category: int) -> float:
        """Accuracy for specific category."""
        category_results = [r for r in self.question_results if r.category == category]
        if not category_results:
            return 0.0
        return sum(1 for r in category_results if r.score.passed) / len(
            category_results
        )


@dataclass
class LongMemEvalExperimentResult:
    """Result for complete LongMemEval experiment."""

    agent_name: str
    conversation_results: List[LongMemEvalConversationResult]
    config: Dict[str, Any]
    timestamp: str

    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy across all conversations."""
        if not self.conversation_results:
            return 0.0
        return sum(c.accuracy for c in self.conversation_results) / len(
            self.conversation_results
        )

    @property
    def overall_exact_match_rate(self) -> float:
        """Overall exact match rate."""
        if not self.conversation_results:
            return 0.0
        return sum(c.exact_match_rate for c in self.conversation_results) / len(
            self.conversation_results
        )

    @property
    def overall_f1_score(self) -> float:
        """Overall average F1 score (binary for LongMemEval)."""
        if not self.conversation_results:
            return 0.0
        return sum(c.avg_f1_score for c in self.conversation_results) / len(
            self.conversation_results
        )

    @property
    def overall_keyword_overlap(self) -> float:
        """Alias for overall_f1_score (backward compatibility)."""
        return self.overall_f1_score

    def accuracy_by_category(self, category: int) -> float:
        """Overall accuracy for specific category."""
        category_results = []
        for conv in self.conversation_results:
            category_results.extend(
                [r for r in conv.question_results if r.category == category]
            )
        if not category_results:
            return 0.0
        return sum(1 for r in category_results if r.score.passed) / len(
            category_results
        )

    def task_averaged_accuracy(self) -> float:
        """Task-averaged accuracy: mean of 5 category accuracies.

        This is the primary metric for LongMemEval, treating each memory
        ability equally regardless of the number of questions per category.
        """
        category_accs = []
        for cat in range(1, 6):
            acc = self.accuracy_by_category(cat)
            # Only include categories that have questions
            category_results = []
            for conv in self.conversation_results:
                category_results.extend(
                    [r for r in conv.question_results if r.category == cat]
                )
            if category_results:
                category_accs.append(acc)

        if not category_accs:
            return 0.0
        return sum(category_accs) / len(category_accs)

    def summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            "agent": self.agent_name,
            "num_conversations": len(self.conversation_results),
            "overall_accuracy": f"{self.overall_accuracy:.1%}",
            "task_averaged_accuracy": f"{self.task_averaged_accuracy():.1%}",
            "information_extraction_accuracy": f"{self.accuracy_by_category(1):.1%}",
            "multi_session_reasoning_accuracy": f"{self.accuracy_by_category(2):.1%}",
            "knowledge_update_accuracy": f"{self.accuracy_by_category(3):.1%}",
            "temporal_reasoning_accuracy": f"{self.accuracy_by_category(4):.1%}",
            "abstention_accuracy": f"{self.accuracy_by_category(5):.1%}",
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "summary": self.summary(),
            "conversations": [
                {
                    "id": c.conversation_id,
                    "num_turns": c.num_turns,
                    "compression_turn": c.compression_turn,
                    "accuracy": c.accuracy,
                    "exact_match_rate": c.exact_match_rate,
                    "avg_f1_score": c.avg_f1_score,
                    "information_extraction_accuracy": c.accuracy_by_category(1),
                    "multi_session_reasoning_accuracy": c.accuracy_by_category(2),
                    "knowledge_update_accuracy": c.accuracy_by_category(3),
                    "temporal_reasoning_accuracy": c.accuracy_by_category(4),
                    "abstention_accuracy": c.accuracy_by_category(5),
                    "questions": [
                        {
                            "question_id": q.question_id,
                            "question": q.question,
                            "question_type": q.question_type,
                            "category": q.category,
                            "category_name": q.category_name,
                            "is_abstention": q.is_abstention,
                            "ground_truth": q.ground_truth,
                            "answer": q.answer,
                            "correct": q.score.correct,
                            "judge_response": q.score.judge_response,
                            "f1_score": q.score.f1_score,
                            "exact_match": q.score.exact_match,
                            "passed": q.score.passed,
                            "latency_ms": q.latency_ms,
                        }
                        for q in c.question_results
                    ],
                }
                for c in self.conversation_results
            ],
        }


# =============================================================================
# Runner
# =============================================================================


class LongMemEvalExperimentRunner:
    """
    Runs LongMemEval evaluation experiments.

    Flow:
    1. Load LongMemEval conversations (500 questions, each with haystack sessions)
    2. For each conversation:
       a. Process turns with rolling/single/dynamic compression
       b. Fix temporal resolution after extraction
       c. Cache save/load for Canvas states
       d. Ask the single QA question
       e. Score using LLM Judge (official LongMemEval protocol)
    """

    def __init__(
        self,
        dataset_path: str,
        compression_at_middle: bool = True,
        compression_turn: Optional[int] = None,
        retain_recent: int = 5,
        rolling_interval: int = 0,  # 0 means disabled (single compression)
        max_turns: int = 0,  # 0 means all turns
        dynamic_compression: bool = False,  # Letta-inspired dynamic triggering
        extraction_mode: str = "batch",  # "batch" (40-turn) or "per_turn" (legacy)
        load_cache: bool = True,  # Load cached Canvas state if available
        save_cache: bool = True,  # Save Canvas state to cache after extraction
        extract_only: bool = False,  # Only extract, skip QA (for cache warmup)
        qa_parallel: int = 1,  # Number of parallel QA workers per conversation (1 = sequential)
        score_model: Optional[str] = None,  # LLM judge model (default from env or gpt-4o-mini)
    ):
        """
        Initialize LongMemEval runner.

        Args:
            dataset_path: Path to LongMemEval JSON file
            compression_at_middle: If True, compress at conversation midpoint
            compression_turn: Fixed compression turn (overrides compression_at_middle)
            retain_recent: Number of recent turns to retain after compression
            rolling_interval: Interval for rolling compression (0 to disable)
            max_turns: Max turns to process per conversation (0 = all)
            dynamic_compression: Use dynamic compression triggers
            extraction_mode: "batch" for 40-turn batch extraction, "per_turn" for legacy
            load_cache: If True, load cached Canvas state when available
            save_cache: If True, save Canvas state to cache after extraction
            extract_only: If True, only extract and cache, skip QA phase
            qa_parallel: Number of parallel workers for QA phase (1 = sequential)
            score_model: LLM judge model name (uses SCORE_MODEL env var or gpt-4o-mini)
        """
        self.compression_at_middle = compression_at_middle
        self.fixed_compression_turn = compression_turn
        self.retain_recent = retain_recent
        self.conversations = self._load_dataset(dataset_path)
        self.rolling_interval = rolling_interval
        self.max_turns = max_turns
        self.dynamic_compression = dynamic_compression
        self.extraction_mode = extraction_mode
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.extract_only = extract_only
        self.qa_parallel = qa_parallel
        self._score_client = None  # Lazy init for LLM scoring
        self._score_model = score_model

    def _get_score_client(self):
        """Get or create OpenAI client for LLM Judge (uses SCORE_API_* config)."""
        if self._score_client is None:
            import os
            from openai import OpenAI
            # Use SCORE_API_* for evaluation, fall back to default API_*
            api_key = os.getenv("SCORE_API_KEY") or os.getenv("API_KEY")
            api_base = os.getenv("SCORE_API_BASE") or os.getenv("API_BASE")
            self._score_client = OpenAI(
                api_key=api_key,
                base_url=api_base,
            )
        return self._score_client

    def _get_score_model(self) -> str:
        """Get the LLM judge model name."""
        if self._score_model:
            return self._score_model
        import os
        return os.getenv("SCORE_MODEL", "gpt-4o-mini")

    def _score_answer(
        self,
        answer: str,
        ground_truth: str,
        question: str,
        question_type: str,
        is_abstention: bool,
    ) -> LongMemEvalScoreResult:
        """
        Score an answer using LLM Judge (official LongMemEval protocol).

        LongMemEval always uses LLM Judge evaluation -- there is no F1 fallback.

        Args:
            answer: Model's answer
            ground_truth: Expected answer
            question: Original question
            question_type: LongMemEval question type
            is_abstention: Whether this is an abstention question

        Returns:
            LongMemEvalScoreResult
        """
        client = self._get_score_client()
        return score_longmemeval_answer(
            answer=answer,
            ground_truth=ground_truth,
            question=question,
            question_type=question_type,
            is_abstention=is_abstention,
            client=client,
            model=self._get_score_model(),
        )

    def _load_dataset(self, path: str) -> List[LongMemEvalConversation]:
        """Load and convert LongMemEval dataset."""
        print(f"Loading LongMemEval dataset from {path}...")
        raw_data = load_longmemeval(path)
        conversations = convert_to_eval_format(raw_data)
        print(f"Loaded {len(conversations)} conversations")
        return conversations

    def run(
        self,
        agent: Agent,
        num_samples: Optional[int] = None,
        verbose: int = 1,
        max_workers: int = 1,
        agent_factory: Optional[callable] = None,
        max_questions_per_conv: Optional[int] = None,
        categories: Optional[List[int]] = None,
    ) -> LongMemEvalExperimentResult:
        """
        Run LongMemEval experiment.

        Args:
            agent: The agent to evaluate (used when max_workers=1)
            num_samples: Number of conversations to evaluate (None = all)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug)
            max_workers: Number of parallel workers for conversations
            agent_factory: Factory function for creating agent instances (required for parallel)
            max_questions_per_conv: Max questions per conversation (usually 1 for LongMemEval)
            categories: Filter by question categories (e.g., [1, 4])

        Returns:
            LongMemEvalExperimentResult with all results
        """
        conversations = self.conversations

        # Filter by categories if specified
        if categories:
            conversations = [
                c for c in conversations
                if any(qa.category in categories for qa in c.qa_pairs)
            ]
            if verbose >= 1:
                print(f"Filtered to {len(conversations)} conversations matching categories {categories}")

        if num_samples:
            conversations = conversations[:num_samples]

        # Compute config hash for cache identification
        agent_config = {}
        if hasattr(agent, 'extractor_model'):
            agent_config['extractor_model'] = agent.extractor_model
        if hasattr(agent, 'embedding_model'):
            agent_config['embedding_model'] = agent.embedding_model
        if hasattr(agent, 'enable_temporal_heuristic'):
            agent_config['enable_temporal_heuristic'] = agent.enable_temporal_heuristic
        if hasattr(agent, 'enable_gleaning'):
            agent_config['enable_gleaning'] = agent.enable_gleaning
        # Representation mode MUST be part of the cache identity: artifacts,
        # chunks, and union runs produce incompatible canvases. Omitting these
        # made all configs share one hash and silently load each other's stores.
        if hasattr(agent, 'chunks_mode'):
            agent_config['chunks_mode'] = agent.chunks_mode
        if hasattr(agent, 'chunks_chunk_size'):
            agent_config['chunks_chunk_size'] = agent.chunks_chunk_size
        if hasattr(agent, 'chunks_overlap'):
            agent_config['chunks_overlap'] = agent.chunks_overlap
        if hasattr(agent, 'union_mode'):
            agent_config['union_mode'] = agent.union_mode
        agent_config['rolling_interval'] = self.rolling_interval

        # Generate config_hash if either load or save cache is enabled
        use_cache = self.load_cache or self.save_cache
        config_hash = get_extraction_config_hash(agent_config, self.extraction_mode) if use_cache else None

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"LongMemEval Experiment: {agent.name}")
            print(f"Conversations: {len(conversations)}")
            if self.rolling_interval > 0:
                print(f"Strategy: Rolling Compression (interval={self.rolling_interval})")
            else:
                print(
                    f"Strategy: Single Compression ({'middle' if self.compression_at_middle else f'turn {self.fixed_compression_turn}'})"
                )
            print(f"Retain recent: {self.retain_recent} turns")
            print(f"Max workers: {max_workers}")
            print(f"Verbose level: {verbose}")
            print(f"Score model: {self._get_score_model()}")
            if max_questions_per_conv:
                print(f"Max questions per conversation: {max_questions_per_conv}")
            if categories:
                # Count filtered questions
                total_filtered = sum(
                    len([qa for qa in c.qa_pairs if qa.category in categories])
                    for c in conversations
                )
                print(f"Categories filter: {categories} ({total_filtered} questions)")
            if config_hash:
                cache_mode = []
                if self.load_cache:
                    cache_mode.append("load")
                if self.save_cache:
                    cache_mode.append("save")
                print(f"Cache: {'+'.join(cache_mode)} (hash={config_hash})")
            print(f"Extraction mode: {self.extraction_mode}")
            if self.qa_parallel > 1:
                print(f"QA parallel: {self.qa_parallel} workers per conversation")
            print(f"{'='*60}\n")

        results = []

        if max_workers > 1:
            if agent_factory is None:
                raise ValueError("agent_factory is required for parallel execution")
            results = self._run_parallel(
                conversations,
                agent_factory,
                max_workers,
                verbose,
                max_questions_per_conv,
                categories,
                config_hash=config_hash,
            )
        else:
            for i, conv in enumerate(conversations):
                if verbose >= 1:
                    print(f"[{i+1}/{len(conversations)}] Conversation {conv.id}")

                result = self._run_single_conversation(
                    agent, conv, verbose, max_questions_per_conv, categories,
                    config_hash=config_hash,
                )
                results.append(result)

                if verbose >= 1:
                    print(
                        f"    => Accuracy: {result.accuracy:.0%} | "
                        f"Exact: {result.exact_match_rate:.0%} | "
                        f"F1: {result.avg_f1_score:.0%}"
                    )

        experiment_result = LongMemEvalExperimentResult(
            agent_name=agent.name,
            conversation_results=results,
            config={
                "rolling_interval": self.rolling_interval,
                "compression_at_middle": self.compression_at_middle,
                "fixed_compression_turn": self.fixed_compression_turn,
                "retain_recent": self.retain_recent,
                "num_samples": num_samples or len(self.conversations),
                "max_questions_per_conv": max_questions_per_conv,
                "categories": categories,
                "benchmark_type": "longmemeval",
                "score_model": self._get_score_model(),
            },
            timestamp=datetime.now().isoformat(),
        )

        if verbose >= 1:
            print(f"\n{'='*60}")
            print("LONGMEMEVAL RESULTS SUMMARY")
            print(f"{'='*60}")
            for k, v in experiment_result.summary().items():
                print(f"  {k}: {v}")

        return experiment_result

    def _run_parallel(
        self,
        conversations: List[LongMemEvalConversation],
        agent_factory: callable,
        max_workers: int,
        verbose: int,
        max_questions_per_conv: Optional[int],
        categories: Optional[List[int]] = None,
        config_hash: str = None,
    ) -> List[LongMemEvalConversationResult]:
        """Run conversations in parallel."""
        results = [None] * len(conversations)
        completed = [0]
        lock = threading.Lock()

        def process_conv(
            idx: int, conv: LongMemEvalConversation
        ) -> Tuple[int, LongMemEvalConversationResult]:
            # In parallel mode, only use verbose >= 2 for per-question detail
            conv_verbose = verbose if verbose >= 2 else 0
            try:
                with lock:
                    print(f"  [Worker {idx}] Creating agent for {conv.id}...", flush=True)
                agent = agent_factory()
                with lock:
                    print(f"  [Worker {idx}] Agent created, starting conversation {conv.id}...", flush=True)
                if verbose >= 2:
                    with lock:
                        print(f"  [Starting] {conv.id} ({len(conv.turns)} turns, {len(conv.qa_pairs)} questions)")
                result = self._run_single_conversation(
                    agent, conv, verbose=conv_verbose, max_questions=max_questions_per_conv,
                    categories=categories, config_hash=config_hash,
                )
            except Exception as e:
                print(f"Error in conversation {conv.id}: {e}, skipping...")
                import traceback
                traceback.print_exc()
                # Return empty result instead of crashing
                result = LongMemEvalConversationResult(
                    conversation_id=conv.id,
                    num_turns=len(conv.turns),
                    compression_turn=0,
                    question_results=[],
                    total_time_ms=0,
                )

            with lock:
                completed[0] += 1
                if verbose >= 1:
                    # Show per-question breakdown at -vv
                    detail = ""
                    if verbose >= 2:
                        passed = sum(1 for q in result.question_results if q.score.passed)
                        total = len(result.question_results)
                        detail = f" | Passed: {passed}/{total}"
                    print(
                        f"[{completed[0]}/{len(conversations)}] {conv.id} => "
                        f"Accuracy: {result.accuracy:.0%}{detail}"
                    )

            return idx, result

        # Sort conversations by number of turns (descending) for better load balancing
        indexed_convs = list(enumerate(conversations))
        indexed_convs.sort(key=lambda x: len(x[1].turns), reverse=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_conv, i, conv): i
                for i, conv in indexed_convs
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _fix_temporal_resolution(
        self,
        canvas,
        turns: List[ConversationTurn],
        verbose: int = 0,
    ) -> int:
        """
        Fix temporal resolution for Canvas objects after batch extraction.

        In batch extraction, all relative times are resolved using the last session's datetime.
        This method re-resolves them using each object's turn_id to find the correct session_datetime.

        Args:
            canvas: The Canvas instance with extracted objects
            turns: List of conversation turns with session_datetime
            verbose: Verbosity level

        Returns:
            Number of objects with corrected event_time
        """
        from cogcanvas.temporal import resolve_relative_time

        # Build turn_id -> session_datetime mapping
        turn_to_datetime = {}
        for turn in turns:
            session_dt = getattr(turn, "session_datetime", None)
            if session_dt:
                turn_to_datetime[turn.turn_id] = session_dt

        if not turn_to_datetime:
            return 0  # No session datetimes available

        fixed_count = 0
        for obj in canvas._objects.values():
            # Skip objects without relative time expressions
            if not obj.event_time_raw:
                continue

            # Skip objects with absolute dates (no relative expressions)
            # Relative times: yesterday, last week, today, N days ago, etc.
            raw_lower = obj.event_time_raw.lower()
            is_relative = any(kw in raw_lower for kw in [
                'yesterday', 'today', 'tomorrow', 'last', 'next', 'ago',
                'this week', 'this month', 'this year'
            ])

            if not is_relative:
                continue

            # Get the correct session_datetime for this object's turn
            turn_id = getattr(obj, 'turn_id', None)
            if turn_id is None:
                continue

            correct_datetime = turn_to_datetime.get(turn_id)
            if not correct_datetime:
                # Try to find the closest earlier turn with datetime
                for tid in sorted(turn_to_datetime.keys(), reverse=True):
                    if tid <= turn_id:
                        correct_datetime = turn_to_datetime[tid]
                        break

            if not correct_datetime:
                continue

            # Re-resolve the relative time with the correct session_datetime
            new_event_time = resolve_relative_time(obj.event_time_raw, correct_datetime)

            if new_event_time and new_event_time != obj.event_time:
                if verbose >= 3:
                    print(f"      [TEMPORAL FIX] Turn {turn_id}: '{obj.event_time_raw}' "
                          f"({obj.event_time} -> {new_event_time}) using {correct_datetime}")
                obj.event_time = new_event_time
                fixed_count += 1

        return fixed_count

    def _run_single_conversation(
        self,
        agent: Agent,
        conv: LongMemEvalConversation,
        verbose: int = 0,
        max_questions: Optional[int] = None,
        categories: Optional[List[int]] = None,
        config_hash: str = None,
    ) -> LongMemEvalConversationResult:
        """Run experiment on single conversation."""
        agent.reset()
        start_time = time.time()

        # Set conversation ID for agents that support caching (e.g., GraphRAG)
        if hasattr(agent, 'set_conv_id'):
            agent.set_conv_id(conv.id)

        # === CACHE CHECK ===
        cache_path = None
        cache_loaded = False
        if config_hash and hasattr(agent, 'load_canvas_state'):
            cache_path = get_cache_path(conv.id, config_hash)
            # Only load from cache if load_cache is enabled
            if self.load_cache and cache_path.exists():
                cache_loaded = agent.load_canvas_state(str(cache_path))
                if cache_loaded and verbose >= 1:
                    print(f"    [CACHE HIT] Loaded Canvas state from {cache_path}")

        # Decide strategy: Dynamic vs Rolling vs Single
        is_dynamic = self.dynamic_compression and hasattr(agent, 'should_compress')
        is_rolling = self.rolling_interval > 0 and not is_dynamic

        # Apply max_turns limit if set
        turns_to_process = conv.turns
        if self.max_turns > 0:
            turns_to_process = conv.turns[:self.max_turns]
            if verbose >= 2:
                print(f"    [max-turns] Limited to {len(turns_to_process)}/{len(conv.turns)} turns")

        # --- PROCESSING LOOP ---
        # Skip extraction if cache was loaded successfully
        if cache_loaded:
            if verbose >= 2:
                print(f"    [CACHE] Skipping extraction, using cached Canvas state")
            compression_turn = len(turns_to_process)
        elif is_dynamic:
            # === DYNAMIC COMPRESSION STRATEGY (Letta-inspired) ===
            current_buffer = []
            compression_count = 0
            if verbose >= 2:
                print(f"    Running Dynamic Compression (topic shift / density / turn limit)...")

            for i, turn in enumerate(turns_to_process):
                agent.process_turn(turn, verbose=verbose)
                current_buffer.append(turn)

                # Check dynamic compression trigger
                should_trigger, reason = agent.should_compress(
                    current_turn_index=i,
                    total_turns=len(turns_to_process),
                    verbose=verbose
                )

                if should_trigger:
                    compression_count += 1
                    retained_turns = current_buffer[-self.retain_recent:]
                    agent.on_compression(retained_turns, verbose=verbose, reason=reason)
                    current_buffer = list(retained_turns)

                    if verbose >= 2:
                        print(f"      [Dynamic #{compression_count}] Compressed at turn {i+1}, reason={reason}")

            # Final compression
            retained_turns = current_buffer[-self.retain_recent:]
            agent.on_compression(retained_turns, verbose=verbose, reason="dynamic_final")
            compression_turn = len(turns_to_process)

            if verbose >= 2:
                print(f"    Dynamic compression triggered {compression_count} times during conversation")

        elif is_rolling:
            # === ROLLING COMPRESSION STRATEGY ===
            current_buffer = []
            use_batch = self.extraction_mode == "batch" and hasattr(agent, 'batch_extract')

            if verbose >= 2:
                mode_str = "Batch Extraction" if use_batch else "Per-Turn Extraction (legacy)"
                print(f"    Running Rolling Compression with {mode_str} (interval={self.rolling_interval})...")

            batch_buffer = []  # Accumulate turns for batch extraction

            for i, turn in enumerate(turns_to_process):
                batch_buffer.append(turn)
                current_buffer.append(turn)

                # Every N turns: extract + compress
                if (i + 1) % self.rolling_interval == 0:
                    # Step 1: Extract
                    if use_batch and hasattr(agent, 'batch_extract'):
                        # BATCH MODE: 1 LLM call for N turns (recommended, CogCanvas only)
                        agent.batch_extract(batch_buffer, verbose=verbose)
                    else:
                        # PER-TURN MODE: N LLM calls (for all agents)
                        for t in batch_buffer:
                            # Only CogCanvasAgent supports verbose in process_turn
                            if hasattr(agent, '_canvas'):
                                agent.process_turn(t, verbose=verbose)
                            else:
                                agent.process_turn(t)

                    # Step 2: Compress history (keep last 5 turns)
                    retained_turns = current_buffer[-self.retain_recent:]
                    if hasattr(agent, '_canvas'):  # CogCanvasAgent
                        agent.on_compression(retained_turns, verbose=verbose, reason=f"rolling_interval_{i+1}")
                    else:
                        agent.on_compression(retained_turns)

                    # Step 3: Reset buffers
                    batch_buffer = []  # Clear batch buffer for next interval
                    current_buffer = list(retained_turns)

                    if verbose >= 2:
                        print(f"      [Rolling] Extracted & compressed at turn {turn.turn_id}")

            # Handle remaining turns (< interval size)
            if batch_buffer:
                if use_batch and hasattr(agent, 'batch_extract'):
                    agent.batch_extract(batch_buffer, verbose=verbose)
                else:
                    for t in batch_buffer:
                        if hasattr(agent, '_canvas'):
                            agent.process_turn(t, verbose=verbose)
                        else:
                            agent.process_turn(t)

            # Final compression to ensure state is consistent before QA
            retained_turns = current_buffer[-self.retain_recent:]
            if hasattr(agent, '_canvas'):  # CogCanvasAgent
                agent.on_compression(retained_turns, verbose=verbose, reason="rolling_final")
            else:
                agent.on_compression(retained_turns)
            compression_turn = len(conv.turns)  # Logic compression point is the end

        else:
            # === SINGLE COMPRESSION STRATEGY (Legacy) ===

            # Determine compression point
            if self.fixed_compression_turn:
                compression_turn = self.fixed_compression_turn
            else:
                compression_turn = conv.get_compression_point()
            compression_turn = min(compression_turn, len(turns_to_process))

            if verbose >= 2:
                print(f"    Compression at turn {compression_turn}/{len(turns_to_process)}")

            # Phase 1: Pre-compression
            pre_turns = [t for t in turns_to_process if t.turn_id <= compression_turn]
            for i, turn in enumerate(pre_turns):
                agent.process_turn(turn, verbose=verbose)

            # Phase 2: Compression
            retained_turns = [
                t for t in turns_to_process
                if t.turn_id > compression_turn - self.retain_recent
                and t.turn_id <= compression_turn
            ]
            # Pass verbose/reason to CogCanvasAgent for detailed logging
            if hasattr(agent, '_canvas'):  # CogCanvasAgent
                agent.on_compression(retained_turns, verbose=verbose, reason=f"single_compression_turn_{compression_turn}")
            else:
                agent.on_compression(retained_turns)

            # Phase 3: Post-compression
            post_turns = [t for t in turns_to_process if t.turn_id > compression_turn]
            for i, turn in enumerate(post_turns):
                agent.process_turn(turn, verbose=verbose)

        # === POST-PROCESS: Fix temporal resolution ===
        # In batch extraction, all turns use the last session_datetime for relative time resolution
        # This post-processing step corrects event_time based on each object's turn_id
        if not cache_loaded and hasattr(agent, '_canvas'):
            fixed_count = self._fix_temporal_resolution(agent._canvas, turns_to_process, verbose)
            if verbose >= 2 and fixed_count > 0:
                print(f"    [TEMPORAL FIX] Corrected {fixed_count} event times")

        # === CACHE SAVE ===
        # Save Canvas state if extraction was performed (not loaded from cache) and save_cache is enabled
        if not cache_loaded and self.save_cache and cache_path and hasattr(agent, 'save_canvas_state'):
            agent.save_canvas_state(str(cache_path))
            if verbose >= 2:
                print(f"    [CACHE SAVE] Saved Canvas state to {cache_path}")

        # --- EXTRACT ONLY MODE: Skip QA ---
        if self.extract_only:
            elapsed = (time.time() - start_time) * 1000
            if verbose >= 1:
                num_objects = len(agent._canvas._objects) if hasattr(agent, '_canvas') else 0
                print(f"    [EXTRACT ONLY] Cached {num_objects} objects in {elapsed:.0f}ms")
            return LongMemEvalConversationResult(
                conversation_id=conv.id,
                num_turns=len(turns_to_process),
                compression_turn=compression_turn,
                question_results=[],
                total_time_ms=elapsed,
            )

        # --- QA PHASE ---
        return self._run_qa_phase(
            agent, conv, compression_turn, start_time, verbose, max_questions, categories
        )

    def _answer_single_question_parallel(
        self,
        qa: LongMemEvalQAPair,
        conv: LongMemEvalConversation,
        context: str,
        answer_model: str,
        prompt_style: str,
    ) -> LongMemEvalQuestionResult:
        """
        Answer a single question in parallel mode.

        Uses independent LLM client to avoid thread safety issues.
        Context is pre-built to avoid concurrent Canvas access.
        """
        import os
        from openai import OpenAI
        from experiments.llm_utils import call_llm_with_retry

        q_start = time.time()

        # Create independent client for this thread
        api_key = os.getenv("ANSWER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("ANSWER_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")
        client = OpenAI(api_key=api_key, base_url=api_base)

        # Build prompt with question_date context
        question = qa.question
        question_date = qa.question_date

        if prompt_style == "cot":
            prompt = f"""Based on the following memory context, answer the question.

## Memory Context
{context}

## Current Date
{question_date}

## Question
{question}

## Instructions
1. Identify relevant facts from the context
2. Connect facts if needed for multi-hop reasoning
3. Consider temporal information and dates when relevant
4. Synthesize a complete answer

## Answer
Provide a concise, direct answer."""
        else:
            prompt = f"""Based on the following context, answer the question.

Context: {context}

Current date: {question_date}

Question: {question}

Answer:"""

        try:
            answer = call_llm_with_retry(
                client=client,
                model=answer_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
                verbose=False,
                timeout=60,
                max_retries=5,
            )
            if answer is None:
                answer = "Error: LLM returned None"
        except Exception as e:
            import threading
            thread_id = threading.current_thread().name
            print(f"        [WARN] Q#{qa.question[:30]}... failed in {thread_id}: {type(e).__name__}", flush=True)
            answer = f"Error: {type(e).__name__}"

        latency = (time.time() - q_start) * 1000

        # Score using LLM Judge (always, for LongMemEval)
        score = score_longmemeval_answer(
            answer=answer,
            ground_truth=qa.answer,
            question=qa.question,
            question_type=qa.question_type,
            is_abstention=qa.is_abstention,
            client=client,
            model=self._get_score_model(),
        )

        return LongMemEvalQuestionResult(
            question_id=qa.question_id,
            question=qa.question,
            question_type=qa.question_type,
            category=qa.category,
            category_name=qa.category_name,
            is_abstention=qa.is_abstention,
            ground_truth=qa.answer,
            answer=answer,
            score=score,
            latency_ms=latency,
        )

    def _run_qa_phase(
        self,
        agent: Agent,
        conv: LongMemEvalConversation,
        compression_turn: int,
        start_time: float,
        verbose: int,
        max_questions: Optional[int],
        categories: Optional[List[int]],
    ) -> LongMemEvalConversationResult:
        """Run the Question-Answering phase (supports parallel execution)."""

        qa_pairs = conv.qa_pairs

        # Filter by categories
        if categories:
            qa_pairs = [qa for qa in qa_pairs if qa.category in categories]

        # Limit number of questions
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]

        # Parallel QA only works for CogCanvas agents (requires _canvas for retrieval)
        use_parallel = self.qa_parallel > 1 and len(qa_pairs) > 1 and hasattr(agent, '_canvas') and agent._canvas

        if verbose >= 2:
            mode = f"parallel ({self.qa_parallel} workers)" if use_parallel else "sequential"
            if self.qa_parallel > 1 and not use_parallel:
                mode += " (parallel not supported for this agent)"
            print(f"    Phase 4: Answering {len(qa_pairs)} questions ({mode})...")

        question_results = []

        if use_parallel:
            # === PARALLEL QA ===
            # Strategy: Pre-compute retrieval contexts (sequential), then parallel LLM calls
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Step 1: Batch embed all questions first
            if verbose >= 2:
                print(f"      [Parallel] Step 1a: Batch embedding {len(qa_pairs)} questions...", flush=True)

            embed_start = time.time()
            questions = [qa.question for qa in qa_pairs]

            if hasattr(agent, '_canvas') and agent._canvas:
                query_embeddings = agent._canvas.batch_embed_queries(questions)
            else:
                query_embeddings = [None] * len(questions)

            embed_ms = (time.time() - embed_start) * 1000
            if verbose >= 2:
                print(f"      [Parallel] Batch embedding done in {embed_ms:.0f}ms", flush=True)

            # Step 1b: Retrieval with pre-computed embeddings
            if verbose >= 2:
                print(f"      [Parallel] Step 1b: Retrieving contexts...", flush=True)

            qa_contexts = []
            retrieval_start = time.time()

            for qi, (qa, query_emb) in enumerate(zip(qa_pairs, query_embeddings)):
                if verbose >= 2 and (qi + 1) % 50 == 0:
                    print(f"        Retrieval progress: {qi + 1}/{len(qa_pairs)}", flush=True)
                if hasattr(agent, '_canvas') and agent._canvas:
                    retrieval_result = agent._canvas.retrieve(
                        query=qa.question,
                        top_k=getattr(agent, 'retrieval_top_k', 10),
                        method=getattr(agent, 'retrieval_method', 'hybrid'),
                        include_related=getattr(agent, 'enable_graph_expansion', True),
                        max_hops=getattr(agent, 'graph_hops', 1),
                        query_embedding=query_emb,
                    )
                    context_parts = []
                    for obj in retrieval_result.objects:
                        if obj.quote:
                            context_parts.append(f"- {obj.quote}")
                        if obj.content:
                            context_parts.append(f"  ({obj.content})")
                    context = "\n".join(context_parts) if context_parts else "No relevant context found."
                else:
                    context = "No canvas available."

                qa_contexts.append((qa, context))

            retrieval_ms = (time.time() - retrieval_start) * 1000
            if verbose >= 2:
                print(f"      [Parallel] Retrieval done in {retrieval_ms:.0f}ms (total embed+retrieve: {embed_ms + retrieval_ms:.0f}ms)", flush=True)

            # Step 2: Parallel LLM calls
            if verbose >= 2:
                print(f"      [Parallel] Step 2: Sending {len(qa_pairs)} LLM requests ({self.qa_parallel} workers)...")

            answer_model = getattr(agent, 'answer_model', 'gpt-4o-mini')
            prompt_style = getattr(agent, 'prompt_style', 'cot')

            with ThreadPoolExecutor(max_workers=self.qa_parallel) as executor:
                future_to_idx = {
                    executor.submit(
                        self._answer_single_question_parallel,
                        qa, conv, context, answer_model, prompt_style
                    ): i
                    for i, (qa, context) in enumerate(qa_contexts)
                }

                results_by_idx = {}
                completed = 0
                llm_start_time = time.time()

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    results_by_idx[idx] = result
                    completed += 1

                    if verbose >= 2 and (completed % 10 == 0 or completed == len(qa_pairs)):
                        passed_so_far = sum(1 for r in results_by_idx.values() if r.score.passed)
                        elapsed = time.time() - llm_start_time
                        avg_per_q = elapsed / completed if completed > 0 else 0
                        remaining = len(qa_pairs) - completed
                        eta = avg_per_q * remaining
                        print(
                            f"        LLM progress: {completed}/{len(qa_pairs)} done, {passed_so_far} passed "
                            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", flush=True
                        )

                        if verbose >= 3:
                            pending_indices = [future_to_idx[f] for f in future_to_idx if f not in results_by_idx or not f.done()]
                            if pending_indices:
                                print(f"          Pending question indices: {pending_indices[:10]}{'...' if len(pending_indices) > 10 else ''}", flush=True)

            # Restore original order
            question_results = [results_by_idx[i] for i in range(len(qa_contexts))]

            # Show sample results for debugging
            if verbose >= 2:
                print(f"      [Sample Results]", flush=True)
                passed_samples = [r for r in question_results if r.score.passed][:3]
                failed_samples = [r for r in question_results if not r.score.passed][:3]

                for r in passed_samples:
                    print(f"        PASS Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          Judge: {r.score.judge_response.strip()}", flush=True)

                for r in failed_samples:
                    print(f"        FAIL Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          Judge: {r.score.judge_response.strip()}", flush=True)

        else:
            # === SEQUENTIAL QA ===
            for qi, qa in enumerate(qa_pairs):
                q_start = time.time()

                # Build question prompt with question_date
                if hasattr(agent, '_canvas'):
                    response = agent.answer_question(qa.question, verbose=verbose)
                else:
                    response = agent.answer_question(qa.question)

                latency = (time.time() - q_start) * 1000

                # Score using LLM Judge
                score = self._score_answer(
                    answer=response.answer,
                    ground_truth=qa.answer,
                    question=qa.question,
                    question_type=qa.question_type,
                    is_abstention=qa.is_abstention,
                )

                if verbose >= 2:
                    status = "PASS" if score.passed else "FAIL"
                    print(
                        f"      {status} [{conv.id}] [{qa.category_name}] {qa.question[:35]:35s} -> "
                        f"Judge: {score.judge_response.strip()}"
                    )

                question_results.append(
                    LongMemEvalQuestionResult(
                        question_id=qa.question_id,
                        question=qa.question,
                        question_type=qa.question_type,
                        category=qa.category,
                        category_name=qa.category_name,
                        is_abstention=qa.is_abstention,
                        ground_truth=qa.answer,
                        answer=response.answer,
                        score=score,
                        latency_ms=latency,
                    )
                )

            # Show sample results for debugging (sequential mode)
            if verbose >= 2:
                print(f"      [Sample Results]", flush=True)
                passed_samples = [r for r in question_results if r.score.passed][:3]
                failed_samples = [r for r in question_results if not r.score.passed][:3]

                for r in passed_samples:
                    print(f"        PASS Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          Judge: {r.score.judge_response.strip()}", flush=True)

                for r in failed_samples:
                    print(f"        FAIL Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          Judge: {r.score.judge_response.strip()}", flush=True)

        total_time = (time.time() - start_time) * 1000

        return LongMemEvalConversationResult(
            conversation_id=conv.id,
            num_turns=len(conv.turns),
            compression_turn=compression_turn,
            question_results=question_results,
            total_time_ms=total_time,
        )


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse
    import os
    from dotenv import load_dotenv
    from pathlib import Path

    # Load .env file
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    # Configure OpenAI API
    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
    os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

    parser = argparse.ArgumentParser(description="Run LongMemEval evaluation experiments")
    parser.add_argument(
        "--dataset",
        "-d",
        default="experiments/data/longmemeval/data/longmemeval_s_cleaned.json",
        help="Path to LongMemEval dataset JSON file",
    )
    parser.add_argument(
        "--agent",
        "-a",
        choices=[
            "cogcanvas",
            "cogcanvas-nograph",
            "cogcanvas-filter",
            "cogcanvas-boost",
            "cogcanvas-baseline",
            "cogcanvas-temporal",
            "cogcanvas-hybrid",
            "cogcanvas-cot",
            "cogcanvas-3hop",
            "cogcanvas-3hop-rerank",
            "cogcanvas-qexp",
            "cogcanvas-enhanced",
            "cogcanvas-vage",
            "cogcanvas-vage-learned",
            "cogcanvas-vage-chain",
            "cogcanvas-cot-v2",
            "cogcanvas-cot-fusion",
            # Ablation variants
            "cogcanvas-no-cot",
            "cogcanvas-no-temporal",
            "cogcanvas-no-hybrid",
            "cogcanvas-no-rerank",
            "cogcanvas-no-graph",
            "cogcanvas-no-gleaning",
            "cogcanvas-chunks",          # P1-7: chunks-as-artifacts ablation (graph ON)
            "cogcanvas-chunks-nograph",  # P1-7: chunks + graph OFF
            "cogcanvas-minimal",
            # Multi-round retrieval variants
            "cogcanvas-multiround",
            "cogcanvas-multiround-routed",
            "cogcanvas-multiround-expand",
            "cogcanvas-expand-only",
            "cogcanvas-smart",
            "cogcanvas-recall-boost",
            "cogcanvas-balanced",
            "native",
            "summarization",
            "rag",
            "rag-rerank",
            "memgpt-lite",
            "graphrag-lite",
            "graphrag",
        ],
        default="cogcanvas",
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=None,
        help="Number of conversations to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--compression-turn",
        type=int,
        default=None,
        help="Fixed compression turn (default: middle of conversation)",
    )
    parser.add_argument(
        "--retain-recent",
        type=int,
        default=5,
        help="Number of recent turns to retain",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max questions per conversation (for quick testing)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Filter by question categories, e.g. '1,4' for info-extraction,temporal",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Verbose output (-v for progress, -vv for detailed question results)",
    )
    parser.add_argument(
        "--vage-mode",
        choices=["off", "standard", "chain"],
        default="off",
        help="VAGE selection mode: off (disabled), standard (original), chain (graph-aware)",
    )
    parser.add_argument(
        "--vage-verbose",
        action="store_true",
        help="Print detailed VAGE progress logs",
    )
    parser.add_argument(
        "--rolling-interval",
        type=int,
        default=40,
        help="Interval for rolling compression (e.g. 40 turns). 0 to disable.",
    )
    parser.add_argument(
        "--dynamic-compression",
        action="store_true",
        help="Enable dynamic compression (Letta-inspired): trigger based on topic shift, object density, or turn limit instead of fixed interval.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Max turns to process per conversation (0 = all turns). Useful for quick testing.",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["per_turn", "batch"],
        default="batch",
        help="Extraction strategy: 'per_turn' (legacy, slow), 'batch' (recommended, 40-turn batches)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't load from cache (force re-extraction), but still save to cache",
    )
    parser.add_argument(
        "--no-cache-save",
        action="store_true",
        help="Don't save to cache (completely disable caching)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract and cache Canvas state, skip QA phase (for cache warmup)",
    )
    parser.add_argument(
        "--qa-parallel",
        type=int,
        default=1,
        help="Number of parallel workers for QA phase per conversation (default: 1 = sequential)",
    )
    # No --llm-score flag: LongMemEval always uses LLM Judge

    # P1-7 chunks ablation overrides (only when --agent cogcanvas-chunks*)
    parser.add_argument(
        "--chunks-chunk-size",
        type=int,
        default=None,
        help="Override sliding-window chunk size in chars (chunks variant).",
    )
    parser.add_argument(
        "--chunks-overlap",
        type=int,
        default=None,
        help="Override sliding-window overlap in chars (chunks variant).",
    )

    args = parser.parse_args()

    # Create agent and agent factory
    agent_factory = None
    agent = None

    if args.agent.startswith("cogcanvas"):
        from experiments.agents.cogcanvas_agent import CogCanvasAgent

        # Default Full Config (SOTA) - v3.3: Recall-optimized
        config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": True,
            "enable_gleaning": True,
            "retrieval_method": "hybrid",
            "prompt_style": "cot",
            "retrieval_top_k": 15,
            "graph_hops": 3,
            "use_reranker": True,
            "reranker_candidate_k": 30,
        }

        # =============================================================
        # Ablation Variants
        # =============================================================

        if args.agent in ("cogcanvas-nograph", "cogcanvas-no-graph"):
            config["enable_graph_expansion"] = False

        elif args.agent == "cogcanvas-no-cot":
            config["prompt_style"] = "direct"

        elif args.agent == "cogcanvas-no-temporal":
            config["enable_temporal_heuristic"] = False

        elif args.agent == "cogcanvas-no-hybrid":
            config["retrieval_method"] = "semantic"

        elif args.agent == "cogcanvas-no-rerank":
            config["use_reranker"] = False

        elif args.agent == "cogcanvas-no-gleaning":
            config["enable_gleaning"] = False

        # P1-7: chunks-as-artifacts ablation (graph ON)
        elif args.agent == "cogcanvas-chunks":
            config["chunks_mode"] = True
            config["chunks_chunk_size"] = 512
            config["chunks_overlap"] = 100

        # P1-7: chunks + graph OFF (winning config on LoCoMo)
        elif args.agent == "cogcanvas-chunks-nograph":
            config["chunks_mode"] = True
            config["chunks_chunk_size"] = 512
            config["chunks_overlap"] = 100
            config["enable_graph_expansion"] = False

        # P1-7 CLI overrides for chunks variants
        if config.get("chunks_mode"):
            if getattr(args, "chunks_chunk_size", None) is not None:
                config["chunks_chunk_size"] = args.chunks_chunk_size
            if getattr(args, "chunks_overlap", None) is not None:
                config["chunks_overlap"] = args.chunks_overlap

        elif args.agent == "cogcanvas-minimal":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": False,
                "retrieval_method": "semantic",
                "prompt_style": "direct",
                "retrieval_top_k": 10,
                "graph_hops": 1,
                "use_reranker": False,
            }

        # Legacy aliases
        elif args.agent == "cogcanvas-baseline":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": False,
                "retrieval_method": "semantic",
                "prompt_style": "direct",
                "retrieval_top_k": 20,
            }
        elif args.agent == "cogcanvas-temporal":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "semantic",
                "prompt_style": "direct",
                "retrieval_top_k": 20,
            }
        elif args.agent == "cogcanvas-hybrid":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": False,
                "retrieval_method": "hybrid",
                "prompt_style": "direct",
                "retrieval_top_k": 20,
            }
        elif args.agent == "cogcanvas-cot":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": False,
                "retrieval_method": "semantic",
                "prompt_style": "cot",
                "retrieval_top_k": 20,
            }
        elif args.agent == "cogcanvas-filter":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "use_llm_filter": True,
                "filter_candidate_k": 20,
            }
        elif args.agent == "cogcanvas-3hop":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
            }
        elif args.agent == "cogcanvas-3hop-rerank":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
            }
        elif args.agent == "cogcanvas-qexp":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
                "reranker_candidate_k": 20,
                "use_query_expansion": True,
                "query_expansion_n": 3,
            }
        elif args.agent == "cogcanvas-enhanced":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 5,
                "graph_hops": 3,
                "use_reranker": True,
                "filter_candidate_k": 20,
            }
        elif args.agent == "cogcanvas-vage":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "enable_vage": True,
                "use_learned_vage": False,
                "vage_budget_k": 10,
            }
        elif args.agent == "cogcanvas-vage-learned":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "enable_vage": True,
                "use_learned_vage": True,
                "vage_budget_k": 10,
            }
        elif args.agent == "cogcanvas-cot-v2":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot_v2",
                "retrieval_top_k": 10,
                "graph_hops": 3,
            }
        elif args.agent == "cogcanvas-cot-fusion":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot_fusion",
                "retrieval_top_k": 10,
                "graph_hops": 3,
            }
        elif args.agent == "cogcanvas-vage-chain":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "vage_mode": "chain",
            }

        # =============================================================
        # Multi-Round Retrieval Variants
        # =============================================================
        elif args.agent == "cogcanvas-multiround":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "enable_gleaning": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
                "reranker_candidate_k": 20,
                "use_multi_round": True,
                "max_retrieval_rounds": 3,
                "confidence_threshold": 0.6,
            }
        elif args.agent == "cogcanvas-multiround-routed":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "enable_gleaning": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
                "reranker_candidate_k": 20,
                "use_multi_round": True,
                "max_retrieval_rounds": 3,
                "confidence_threshold": 0.7,
                "use_query_routing": True,
            }
        elif args.agent == "cogcanvas-multiround-expand":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "enable_gleaning": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
                "reranker_candidate_k": 20,
                "use_multi_round": True,
                "max_retrieval_rounds": 3,
                "confidence_threshold": 0.6,
                "use_query_expansion": True,
                "query_expansion_n": 3,
                "use_query_routing": False,
            }
        elif args.agent == "cogcanvas-expand-only":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "enable_gleaning": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
                "reranker_candidate_k": 20,
                "use_query_expansion": True,
                "query_expansion_n": 3,
            }
        elif args.agent == "cogcanvas-smart":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "enable_gleaning": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
                "reranker_candidate_k": 20,
                "use_smart_routing": True,
                "use_multi_round": True,
                "max_retrieval_rounds": 3,
                "confidence_threshold": 0.6,
                "use_query_expansion": True,
                "query_expansion_n": 3,
            }
        elif args.agent == "cogcanvas-recall-boost":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "enable_gleaning": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 20,
                "graph_hops": 4,
                "use_reranker": True,
                "reranker_candidate_k": 40,
            }
        elif args.agent == "cogcanvas-balanced":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "enable_gleaning": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 18,
                "graph_hops": 4,
                "use_reranker": True,
                "reranker_candidate_k": 35,
            }

        # Apply --vage-mode override if specified
        if args.vage_mode != "off":
            config["vage_mode"] = args.vage_mode

        # Apply --vage-verbose if specified
        if args.vage_verbose:
            config["vage_verbose"] = True

        agent_factory = lambda: CogCanvasAgent(**config)
        agent = agent_factory()

    elif args.agent == "rag":
        from experiments.agents.rag_agent import RagAgent

        agent = RagAgent(retain_recent=args.retain_recent)
        agent_factory = lambda: RagAgent(retain_recent=args.retain_recent)
    elif args.agent == "rag-rerank":
        from experiments.agents.rag_agent import RagAgent

        agent = RagAgent(retain_recent=args.retain_recent, use_reranker=True)
        agent_factory = lambda: RagAgent(
            retain_recent=args.retain_recent, use_reranker=True
        )
    elif args.agent == "native":
        from experiments.agents.native_agent import NativeAgent

        agent = NativeAgent(retain_recent=args.retain_recent)
        agent_factory = lambda: NativeAgent(retain_recent=args.retain_recent)
    elif args.agent == "summarization":
        from experiments.agents.summarization_agent import SummarizationAgent

        agent = SummarizationAgent(retain_recent=args.retain_recent)
        agent_factory = lambda: SummarizationAgent(retain_recent=args.retain_recent)
    elif args.agent == "memgpt-lite":
        from experiments.agents.memgpt_lite_agent import MemGPTLiteAgent

        agent = MemGPTLiteAgent(core_memory_size=args.retain_recent)
        agent_factory = lambda: MemGPTLiteAgent(core_memory_size=args.retain_recent)
    elif args.agent == "graphrag-lite":
        from experiments.agents.graphrag_lite_agent import GraphRAGLiteAgent

        agent = GraphRAGLiteAgent(retain_recent=args.retain_recent)
        agent_factory = lambda: GraphRAGLiteAgent(retain_recent=args.retain_recent)
    elif args.agent == "graphrag":
        from experiments.agents.graphrag_agent import create_graphrag_agent

        agent = create_graphrag_agent(search_method="local")
        agent_factory = lambda: create_graphrag_agent(search_method="local")
    else:
        raise NotImplementedError(f"Agent '{args.agent}' not implemented")

    # Determine score model
    score_model = os.getenv("SCORE_MODEL", "gpt-4o-mini")

    # Run experiment
    runner = LongMemEvalExperimentRunner(
        dataset_path=args.dataset,
        compression_at_middle=(args.compression_turn is None),
        compression_turn=args.compression_turn,
        retain_recent=args.retain_recent,
        rolling_interval=args.rolling_interval,
        max_turns=args.max_turns,
        dynamic_compression=args.dynamic_compression,
        extraction_mode=args.extraction_mode,
        load_cache=not args.no_cache,
        save_cache=not args.no_cache_save,
        extract_only=args.extract_only,
        qa_parallel=args.qa_parallel,
        score_model=score_model,
    )

    # Parse categories filter
    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]

    result = runner.run(
        agent,
        num_samples=args.samples,
        verbose=args.verbose + 1,  # default verbose=0 -> level 1, -v -> level 2, -vv -> level 3
        max_workers=args.workers,
        agent_factory=agent_factory,
        max_questions_per_conv=args.max_questions,
        categories=categories,
    )

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
