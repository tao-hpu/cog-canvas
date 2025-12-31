"""
LoCoMo Experiment Runner for CogCanvas Evaluation.

This runner evaluates agents on the LoCoMo (Long Context Multi-hop) benchmark.

LoCoMo Characteristics:
- Real-world multi-session conversations
- Single-hop, temporal, and multi-hop questions
- Evidence-based evaluation with dialogue ID references

Evaluation Strategy:
1. Process conversation turns up to compression point (middle of conversation)
2. Trigger compression
3. Process remaining turns
4. Ask questions from all categories
5. Score using keyword overlap and exact match

Question Categories:
- Single-hop (category 1): Direct fact retrieval
- Temporal (category 2): Time-based reasoning
- Multi-hop (category 3): Requires connecting multiple facts

Scoring:
- Keyword overlap: Fraction of answer keywords found in response
- Exact match: Full answer string appears in response
- Pass threshold: 60% keyword overlap or exact match
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
        "extraction_mode": extraction_mode,
        "rolling_interval": config.get("rolling_interval", 40),
    }
    config_str = json.dumps(extraction_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_cache_path(conv_id: str, config_hash: str) -> Path:
    """Get the cache file path for a conversation."""
    cache_dir = Path("experiments/cache/extraction") / config_hash
    return cache_dir / f"{conv_id}.json"


from experiments.locomo_adapter import (
    load_locomo,
    convert_to_eval_format,
    LoCoMoConversation,
    LoCoMoQAPair,
)


# =============================================================================
# Scoring
# =============================================================================


@dataclass
class LoCoMoScoreResult:
    """Result of scoring a LoCoMo answer (aligned with official LoCoMo evaluation)."""

    f1_score: float  # Token-level F1 score (official LoCoMo metric)
    precision: float  # Token precision
    recall: float  # Token recall
    exact_match: bool  # Whether exact answer appears in response
    prediction_tokens: List[str]  # Normalized + stemmed tokens from prediction
    ground_truth_tokens: List[str]  # Normalized + stemmed tokens from ground truth
    answer: str
    ground_truth: str

    @property
    def passed(self) -> bool:
        """Consider passed if F1 >= 0.5 (aligned with LoCoMo threshold)."""
        return self.f1_score >= 0.5

    # Backward compatibility
    @property
    def keyword_overlap(self) -> float:
        """Alias for recall (backward compatibility)."""
        return self.recall

    @property
    def found_keywords(self) -> List[str]:
        """Tokens found in both prediction and ground truth."""
        return list(set(self.prediction_tokens) & set(self.ground_truth_tokens))

    @property
    def missing_keywords(self) -> List[str]:
        """Tokens in ground truth but not in prediction."""
        return list(set(self.ground_truth_tokens) - set(self.prediction_tokens))


# Porter Stemmer for token normalization (aligned with LoCoMo official)
try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
except ImportError:
    _stemmer = None


def normalize_answer(text: str) -> str:
    """
    Normalize answer text (aligned with LoCoMo official implementation).

    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    """
    import string

    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def tokenize_and_stem(text: str) -> List[str]:
    """
    Tokenize and stem text (aligned with LoCoMo official implementation).

    Uses Porter Stemmer for word stemming.
    """
    normalized = normalize_answer(text)
    tokens = normalized.split()

    # Apply stemming if available
    if _stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]

    return tokens


def compute_f1_score(prediction: str, ground_truth: str) -> tuple:
    """
    Compute token-level F1 score (aligned with LoCoMo official implementation).

    Returns:
        (f1, precision, recall, pred_tokens, truth_tokens)
    """
    pred_tokens = tokenize_and_stem(prediction)
    truth_tokens = tokenize_and_stem(ground_truth)

    if not pred_tokens or not truth_tokens:
        return (0.0, 0.0, 0.0, pred_tokens, truth_tokens)

    # Count common tokens
    common = set(pred_tokens) & set(truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return (0.0, 0.0, 0.0, pred_tokens, truth_tokens)

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return (f1, precision, recall, pred_tokens, truth_tokens)


def score_locomo_answer(answer: str, ground_truth: str) -> LoCoMoScoreResult:
    """
    Score answer using token-level F1 (aligned with LoCoMo official evaluation).

    Scoring approach (from official LoCoMo):
    1. Normalize: lowercase, remove articles (a/an/the), remove punctuation
    2. Tokenize and stem using Porter Stemmer
    3. Compute token-level precision, recall, F1
    4. Exact match: Check if normalized ground truth in normalized answer

    Args:
        answer: The model's answer
        ground_truth: Expected answer from LoCoMo

    Returns:
        LoCoMoScoreResult with F1 scoring details
    """
    # Compute F1 score (official LoCoMo method)
    f1, precision, recall, pred_tokens, truth_tokens = compute_f1_score(answer, ground_truth)

    # Exact match check (on normalized text)
    answer_normalized = normalize_answer(answer)
    truth_normalized = normalize_answer(ground_truth)
    exact_match = truth_normalized in answer_normalized

    return LoCoMoScoreResult(
        f1_score=f1,
        precision=precision,
        recall=recall,
        exact_match=exact_match,
        prediction_tokens=pred_tokens,
        ground_truth_tokens=truth_tokens,
        answer=answer,
        ground_truth=ground_truth,
    )


def score_locomo_answer_llm(
    answer: str,
    ground_truth: str,
    question: str,
    client,
    model: str = "glm-4-flash",
) -> LoCoMoScoreResult:
    """
    Score answer using LLM-based semantic evaluation.

    This method is fairer for cases like:
    - "her mother" vs "her mom" (synonyms)
    - "a few years ago" vs "a few years before 2023" (equivalent expressions)
    - "7 May 2023" vs "May 7, 2023" (date format differences)

    Args:
        answer: The model's answer
        ground_truth: Expected answer from LoCoMo
        question: The original question (for context)
        client: OpenAI-compatible client
        model: Model to use for scoring

    Returns:
        LoCoMoScoreResult with LLM-based scoring
    """
    from experiments.llm_utils import call_llm_with_retry

    prompt = f"""You are an expert evaluator. Judge if the predicted answer is semantically correct compared to the ground truth.

## Question
{question}

## Ground Truth Answer
{ground_truth}

## Predicted Answer
{answer}

## Evaluation Criteria (5-point scale)
- CORRECT: Fully correct, conveys the same meaning (synonyms, paraphrases, equivalent date formats are acceptable)
- MOSTLY_CORRECT: Almost correct with minor omissions or imprecisions that don't change the core meaning
- PARTIAL: Partially correct, contains some correct info but missing key parts or has notable errors
- MOSTLY_WRONG: Contains a small bit of relevant info but largely incorrect or misleading
- INCORRECT: Completely wrong, irrelevant, or contradicts the ground truth

## Response Format
Reply with ONLY one word: CORRECT, MOSTLY_CORRECT, PARTIAL, MOSTLY_WRONG, or INCORRECT"""

    try:
        response = call_llm_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        judgment = response.strip().upper()

        # Map judgment to 5-point scores
        # IMPORTANT: Check longer strings first to avoid substring matching issues
        if "MOSTLY_CORRECT" in judgment:
            llm_score = 0.75
        elif "MOSTLY_WRONG" in judgment:
            llm_score = 0.25
        elif "INCORRECT" in judgment:
            llm_score = 0.0
        elif "PARTIAL" in judgment:
            llm_score = 0.5
        elif "CORRECT" in judgment:
            llm_score = 1.0
        else:
            llm_score = 0.0  # Unknown response treated as incorrect

    except Exception as e:
        print(f"LLM scoring failed: {e}, falling back to F1")
        llm_score = None

    # Also compute F1 for comparison
    f1, precision, recall, pred_tokens, truth_tokens = compute_f1_score(answer, ground_truth)
    answer_normalized = normalize_answer(answer)
    truth_normalized = normalize_answer(ground_truth)
    exact_match = truth_normalized in answer_normalized

    # Use LLM score if available, otherwise fall back to F1
    final_f1 = llm_score if llm_score is not None else f1

    return LoCoMoScoreResult(
        f1_score=final_f1,
        precision=precision,
        recall=recall,
        exact_match=exact_match or (llm_score == 1.0),
        prediction_tokens=pred_tokens,
        ground_truth_tokens=truth_tokens,
        answer=answer,
        ground_truth=ground_truth,
    )


# =============================================================================
# Results
# =============================================================================


@dataclass
class LoCoMoQuestionResult:
    """Result for a single LoCoMo question."""

    question: str
    category: int
    category_name: str
    evidence_turns: List[int]  # Turn IDs where evidence is located
    ground_truth: str
    answer: str
    score: LoCoMoScoreResult
    latency_ms: float


@dataclass
class LoCoMoConversationResult:
    """Result for a single LoCoMo conversation."""

    conversation_id: str
    num_turns: int
    compression_turn: int
    question_results: List[LoCoMoQuestionResult]
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
        """Average F1 score across all questions."""
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
class LoCoMoExperimentResult:
    """Result for complete LoCoMo experiment."""

    agent_name: str
    conversation_results: List[LoCoMoConversationResult]
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
        """Overall average F1 score."""
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

    def summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            "agent": self.agent_name,
            "num_conversations": len(self.conversation_results),
            "overall_accuracy": f"{self.overall_accuracy:.1%}",
            "exact_match_rate": f"{self.overall_exact_match_rate:.1%}",
            "avg_f1_score": f"{self.overall_f1_score:.1%}",
            "single_hop_accuracy": f"{self.accuracy_by_category(1):.1%}",
            "temporal_accuracy": f"{self.accuracy_by_category(2):.1%}",
            "multi_hop_accuracy": f"{self.accuracy_by_category(3):.1%}",
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
                    "single_hop_accuracy": c.accuracy_by_category(1),
                    "temporal_accuracy": c.accuracy_by_category(2),
                    "multi_hop_accuracy": c.accuracy_by_category(3),
                    "questions": [
                        {
                            "question": q.question,
                            "category": q.category,
                            "category_name": q.category_name,
                            "ground_truth": q.ground_truth,
                            "answer": q.answer,
                            "f1_score": q.score.f1_score,
                            "precision": q.score.precision,
                            "recall": q.score.recall,
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


class LoCoMoExperimentRunner:
    """
    Runs LoCoMo evaluation experiments.

    Flow:
    1. Load LoCoMo conversations
    2. For each conversation:
       a. Process turns up to compression point (middle)
       b. Trigger compression
       c. Process remaining turns
       d. Ask all QA questions
       e. Score based on keyword overlap and exact match
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
        llm_score: bool = False,  # Use LLM-based semantic scoring instead of F1
    ):
        """
        Initialize LoCoMo runner.

        Args:
            dataset_path: Path to LoCoMo JSON file
            compression_at_middle: If True, compress at conversation midpoint
            compression_turn: Fixed compression turn (overrides compression_at_middle)
            retain_recent: Number of recent turns to retain after compression
            rolling_interval: Interval for rolling compression (0 to disable)
            max_turns: Max turns to process per conversation (0 = all)
            dynamic_compression: Use dynamic compression triggers (topic shift, density, etc.)
            extraction_mode: "batch" for 40-turn batch extraction, "per_turn" for legacy per-turn
            load_cache: If True, load cached Canvas state when available
            save_cache: If True, save Canvas state to cache after extraction
            extract_only: If True, only extract and cache, skip QA phase
            qa_parallel: Number of parallel workers for QA phase (1 = sequential)
            llm_score: Use LLM-based semantic scoring (fairer for synonyms/paraphrases)
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
        self.llm_score = llm_score
        self._score_client = None  # Lazy init for LLM scoring

    def _get_score_client(self):
        """Get or create OpenAI client for LLM scoring (uses SCORE_API_* config)."""
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

    def _score_answer(self, answer: str, ground_truth: str, question: str = "") -> LoCoMoScoreResult:
        """
        Score an answer using either F1 or LLM-based evaluation.

        Args:
            answer: Model's answer
            ground_truth: Expected answer
            question: Original question (needed for LLM scoring)

        Returns:
            LoCoMoScoreResult
        """
        import os
        if self.llm_score:
            client = self._get_score_client()
            return score_locomo_answer_llm(
                answer, ground_truth, question, client,
                model=os.getenv("SCORE_MODEL", "gpt-4o-mini")
            )
        else:
            return score_locomo_answer(answer, ground_truth)

    def _load_dataset(self, path: str) -> List[LoCoMoConversation]:
        """Load and convert LoCoMo dataset."""
        print(f"Loading LoCoMo dataset from {path}...")
        raw_data = load_locomo(path)
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
    ) -> LoCoMoExperimentResult:
        """
        Run LoCoMo experiment.
        """
        conversations = self.conversations
        if num_samples:
            conversations = conversations[:num_samples]

        # Compute config hash for cache identification
        # Extract config from agent if possible (for CogCanvasAgent)
        agent_config = {}
        if hasattr(agent, 'extractor_model'):
            agent_config['extractor_model'] = agent.extractor_model
        if hasattr(agent, 'embedding_model'):
            agent_config['embedding_model'] = agent.embedding_model
        if hasattr(agent, 'enable_temporal_heuristic'):
            agent_config['enable_temporal_heuristic'] = agent.enable_temporal_heuristic
        agent_config['rolling_interval'] = self.rolling_interval

        # Generate config_hash if either load or save cache is enabled
        use_cache = self.load_cache or self.save_cache
        config_hash = get_extraction_config_hash(agent_config, self.extraction_mode) if use_cache else None

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"LoCoMo Experiment: {agent.name}")
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

        experiment_result = LoCoMoExperimentResult(
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
                "benchmark_type": "locomo",
            },
            timestamp=datetime.now().isoformat(),
        )

        if verbose >= 1:
            print(f"\n{'='*60}")
            print("LOCOMO RESULTS SUMMARY")
            print(f"{'='*60}")
            for k, v in experiment_result.summary().items():
                print(f"  {k}: {v}")

        return experiment_result

    def _run_parallel(
        self,
        conversations: List[LoCoMoConversation],
        agent_factory: callable,
        max_workers: int,
        verbose: int,
        max_questions_per_conv: Optional[int],
        categories: Optional[List[int]] = None,
        config_hash: str = None,  # For cache path generation
    ) -> List[LoCoMoConversationResult]:
        """Run conversations in parallel."""
        results = [None] * len(conversations)
        completed = [0]
        lock = threading.Lock()

        def process_conv(
            idx: int, conv: LoCoMoConversation
        ) -> Tuple[int, LoCoMoConversationResult]:
            # In parallel mode, only use verbose >= 2 for per-question detail
            conv_verbose = verbose if verbose >= 2 else 0
            try:
                agent = agent_factory()
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
                result = LoCoMoConversationResult(
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

        # Sort conversations by number of questions (descending) for better load balancing
        indexed_convs = list(enumerate(conversations))
        indexed_convs.sort(key=lambda x: len(x[1].qa_pairs), reverse=True)

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
        conv: LoCoMoConversation,
        verbose: int = 0,
        max_questions: Optional[int] = None,
        categories: Optional[List[int]] = None,
        config_hash: str = None,  # For cache path generation
    ) -> LoCoMoConversationResult:
        """Run experiment on single conversation."""
        agent.reset()
        start_time = time.time()

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
                    if use_batch:
                        # BATCH MODE: 1 LLM call for N turns (recommended)
                        agent.batch_extract(batch_buffer, verbose=verbose)
                    else:
                        # PER-TURN MODE: N LLM calls (legacy, slow)
                        for t in batch_buffer:
                            agent.process_turn(t, verbose=verbose)

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
                if use_batch:
                    agent.batch_extract(batch_buffer, verbose=verbose)
                else:
                    for t in batch_buffer:
                        agent.process_turn(t, verbose=verbose)

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
            return LoCoMoConversationResult(
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
        qa: LoCoMoQAPair,
        conv: LoCoMoConversation,
        context: str,  # Pre-built context string
        answer_model: str,
        prompt_style: str,
    ) -> LoCoMoQuestionResult:
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
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")
        client = OpenAI(api_key=api_key, base_url=api_base)

        # Build prompt (similar to _extract_answer_from_context)
        question = qa.question
        if prompt_style == "cot":
            prompt = f"""Based on the following memory context, answer the question.

## Memory Context
{context}

## Question
{question}

## Instructions
1. Identify relevant facts from the context
2. Connect facts if needed for multi-hop reasoning
3. Synthesize a complete answer

## Answer
Provide a concise, direct answer."""
        else:
            prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""

        try:
            answer = call_llm_with_retry(
                client=client,
                model=answer_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
                verbose=False,  # Suppress retry logs in parallel
            )
        except Exception as e:
            answer = f"Error: {e}"

        latency = (time.time() - q_start) * 1000

        # Score the answer (use LLM scoring if enabled)
        if self.llm_score:
            score = score_locomo_answer_llm(
                answer, qa.answer, qa.question, client,
                model=os.getenv("ANSWER_MODEL", "glm-4-flash")
            )
        else:
            score = score_locomo_answer(answer, qa.answer)

        # Map evidence IDs to turn numbers
        evidence_turns = [
            conv.dialogue_id_to_turn.get(eid, -1) for eid in qa.evidence
        ]

        return LoCoMoQuestionResult(
            question=qa.question,
            category=qa.category,
            category_name=qa.category_name,
            evidence_turns=evidence_turns,
            ground_truth=qa.answer,
            answer=answer,
            score=score,
            latency_ms=latency,
        )

    def _run_qa_phase(
        self,
        agent: Agent,
        conv: LoCoMoConversation,
        compression_turn: int,
        start_time: float,
        verbose: int,
        max_questions: Optional[int],
        categories: Optional[List[int]],
    ) -> LoCoMoConversationResult:
        """Run the Question-Answering phase (supports parallel execution)."""

        qa_pairs = conv.qa_pairs

        # Filter by categories
        if categories:
            qa_pairs = [qa for qa in qa_pairs if qa.category in categories]

        # Limit number of questions
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]

        if verbose >= 2:
            mode = f"parallel ({self.qa_parallel} workers)" if self.qa_parallel > 1 else "sequential"
            print(f"    Phase 4: Answering {len(qa_pairs)} questions ({mode})...")

        question_results = []

        if self.qa_parallel > 1 and len(qa_pairs) > 1:
            # === PARALLEL QA ===
            # Strategy: Pre-compute retrieval contexts (sequential), then parallel LLM calls
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Step 1: Batch embed all questions first (major optimization!)
            if verbose >= 2:
                print(f"      [Parallel] Step 1a: Batch embedding {len(qa_pairs)} questions...", flush=True)

            embed_start = time.time()
            questions = [qa.question for qa in qa_pairs]

            # Batch embed all questions at once
            if hasattr(agent, '_canvas') and agent._canvas:
                query_embeddings = agent._canvas.batch_embed_queries(questions)
            else:
                query_embeddings = [None] * len(questions)

            embed_ms = (time.time() - embed_start) * 1000
            if verbose >= 2:
                print(f"      [Parallel] Batch embedding done in {embed_ms:.0f}ms", flush=True)

            # Step 1b: Retrieval with pre-computed embeddings (now fast - no API calls)
            if verbose >= 2:
                print(f"      [Parallel] Step 1b: Retrieving contexts...", flush=True)

            qa_contexts = []  # List of (qa, context_str)
            retrieval_start = time.time()

            for qi, (qa, query_emb) in enumerate(zip(qa_pairs, query_embeddings)):
                if verbose >= 2 and (qi + 1) % 50 == 0:
                    print(f"        Retrieval progress: {qi + 1}/{len(qa_pairs)}", flush=True)
                # Get retrieval result using pre-computed embedding
                if hasattr(agent, '_canvas') and agent._canvas:
                    retrieval_result = agent._canvas.retrieve(
                        query=qa.question,
                        top_k=getattr(agent, 'retrieval_top_k', 10),
                        method=getattr(agent, 'retrieval_method', 'hybrid'),
                        include_related=getattr(agent, 'enable_graph_expansion', True),
                        max_hops=getattr(agent, 'graph_hops', 1),
                        query_embedding=query_emb,  # Use pre-computed embedding!
                    )
                    # Build context string from objects
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
                # Submit all questions with pre-computed contexts
                future_to_idx = {
                    executor.submit(
                        self._answer_single_question_parallel,
                        qa, conv, context, answer_model, prompt_style
                    ): i
                    for i, (qa, context) in enumerate(qa_contexts)
                }

                # Collect results as they complete
                results_by_idx = {}
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    results_by_idx[idx] = result
                    completed += 1

                    if verbose >= 2 and (completed % 10 == 0 or completed == len(qa_pairs)):
                        # Show progress every 10 questions
                        passed_so_far = sum(1 for r in results_by_idx.values() if r.score.passed)
                        print(
                            f"        LLM progress: {completed}/{len(qa_pairs)} done, {passed_so_far} passed", flush=True
                        )

            # Restore original order
            question_results = [results_by_idx[i] for i in range(len(qa_contexts))]

            # Show sample results for debugging
            if verbose >= 2:
                print(f"      [Sample Results]", flush=True)
                # Show first 3 passed and first 3 failed
                passed_samples = [r for r in question_results if r.score.passed][:3]
                failed_samples = [r for r in question_results if not r.score.passed][:3]

                for r in passed_samples:
                    print(f"        ✓ Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          F1={r.score.f1_score:.2f}, P={r.score.precision:.2f}, R={r.score.recall:.2f}, EM={r.score.exact_match}", flush=True)

                for r in failed_samples:
                    print(f"        ✗ Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          F1={r.score.f1_score:.2f}, P={r.score.precision:.2f}, R={r.score.recall:.2f}, EM={r.score.exact_match}", flush=True)

        else:
            # === SEQUENTIAL QA ===
            for qi, qa in enumerate(qa_pairs):
                q_start = time.time()
                response = agent.answer_question(qa.question, verbose=verbose)
                latency = (time.time() - q_start) * 1000

                score = self._score_answer(response.answer, qa.answer, qa.question)

                # Map evidence IDs to turn numbers
                evidence_turns = [
                    conv.dialogue_id_to_turn.get(eid, -1) for eid in qa.evidence
                ]

                if verbose >= 2:
                    status = "✓" if score.passed else "✗"
                    score_display = "EXACT" if score.exact_match else f"F1={score.f1_score:.0%}"
                    print(
                        f"      {status} [{conv.id}] [{qa.category_name}] {qa.question[:35]:35s} -> {score_display}"
                    )

                question_results.append(
                    LoCoMoQuestionResult(
                        question=qa.question,
                        category=qa.category,
                        category_name=qa.category_name,
                        evidence_turns=evidence_turns,
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
                    print(f"        ✓ Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          F1={r.score.f1_score:.2f}, P={r.score.precision:.2f}, R={r.score.recall:.2f}, EM={r.score.exact_match}", flush=True)

                for r in failed_samples:
                    print(f"        ✗ Q: {r.question[:50]}...", flush=True)
                    print(f"          GT: {r.ground_truth[:50]}...", flush=True)
                    print(f"          Ans: {r.answer[:50]}...", flush=True)
                    print(f"          F1={r.score.f1_score:.2f}, P={r.score.precision:.2f}, R={r.score.recall:.2f}, EM={r.score.exact_match}", flush=True)

        total_time = (time.time() - start_time) * 1000

        return LoCoMoConversationResult(
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

    parser = argparse.ArgumentParser(description="Run LoCoMo evaluation experiments")
    parser.add_argument(
        "--dataset",
        "-d",
        default="experiments/data/locomo10.json",
        help="Path to LoCoMo dataset JSON file",
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
            "cogcanvas-3hop-rerank",  # New enhanced variants
            "cogcanvas-qexp",  # Query Expansion + Reranking
            "cogcanvas-enhanced",  # CoT Extraction + Two-stage Retrieval
            "cogcanvas-vage",  # Rule-based VAGE
            "cogcanvas-vage-learned",  # Learned VAGE
            "cogcanvas-vage-chain",  # Chain-Level VAGE (graph-aware)
            "cogcanvas-cot-v2",  # CoT V2 prompt
            "cogcanvas-cot-fusion",  # CoT Fusion prompt
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
        help="Number of parallel workers (default: 1)",
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
        help="Filter by question categories, e.g. '1,2,3' for single-hop/temporal/multi-hop only",
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
        default=40,  # Default to Rolling Compression (standard strategy)
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
    parser.add_argument(
        "--llm-score",
        action="store_true",
        help="Use LLM-based semantic scoring instead of F1 (fairer for synonyms/paraphrases)",
    )

    args = parser.parse_args()

    # Create agent and agent factory
    agent_factory = None
    agent = None

    if args.agent.startswith("cogcanvas"):
        from experiments.agents.cogcanvas_agent import CogCanvasAgent

        # Default Full Config (SOTA) - v3.2: +Gleaning +AdaptiveTopK +Reranking
        config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": True,
            "retrieval_method": "hybrid",
            "prompt_style": "cot",
            "retrieval_top_k": 10,  # Base value, adaptive top-k adjusts per question
            "graph_hops": 3,
            "use_reranker": True,  # BGE reranking: +3.3pp
            "reranker_candidate_k": 20,  # Retrieve 20, rerank to top-k
        }

        if args.agent == "cogcanvas-nograph":
            config["enable_graph_expansion"] = False

        # Ablation Variants
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
            # Full config with LLM Filtering
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "use_llm_filter": True,
                "filter_candidate_k": 20,
            }
        elif args.agent == "cogcanvas-3hop":
            # 3-hop graph expansion (enhanced multi-hop reasoning)
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,  # 3-hop expansion
            }
        elif args.agent == "cogcanvas-3hop-rerank":
            # 3-hop + reranking (full enhanced config)
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,  # Enable BGE reranker
            }
        elif args.agent == "cogcanvas-qexp":
            # Query Expansion + Reranking (Perplexity-style multi-query)
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "use_reranker": True,
                "reranker_candidate_k": 20,
                "use_query_expansion": True,  # NEW: Enable query expansion
                "query_expansion_n": 3,  # Generate 3 queries (original + 2 variants)
            }
        elif args.agent == "cogcanvas-enhanced":
            # CoT Extraction + Two-stage Retrieval (Hybrid + Rerank)
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",  # BM25 + Semantic fusion
                "prompt_style": "cot",
                "retrieval_top_k": 5,  # Final top-k after reranking
                "graph_hops": 3,
                "use_reranker": True,  # Two-stage: retrieve 20 → rerank to 5
                "filter_candidate_k": 20,  # Coarse retrieval
            }
        elif args.agent == "cogcanvas-vage":
            # Rule-based VAGE (heuristic vulnerability model)
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
            # Learned VAGE (trained vulnerability model)
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
            # CoT V2 prompt style
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot_v2",
                "retrieval_top_k": 10,
                "graph_hops": 3,
            }
        elif args.agent == "cogcanvas-cot-fusion":
            # CoT Fusion prompt style (Multi-Artifact Fusion)
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot_fusion",
                "retrieval_top_k": 10,
                "graph_hops": 3,
            }
        elif args.agent == "cogcanvas-vage-chain":
            # Chain-Level VAGE (graph-aware selection)
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "hybrid",
                "prompt_style": "cot",
                "retrieval_top_k": 10,
                "graph_hops": 3,
                "vage_mode": "chain",
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

    # Run experiment
    runner = LoCoMoExperimentRunner(
        dataset_path=args.dataset,
        compression_at_middle=(args.compression_turn is None),
        compression_turn=args.compression_turn,
        retain_recent=args.retain_recent,
        rolling_interval=args.rolling_interval,
        max_turns=args.max_turns,
        dynamic_compression=args.dynamic_compression,
        extraction_mode=args.extraction_mode,
        load_cache=not args.no_cache,  # --no-cache disables loading
        save_cache=not args.no_cache_save,  # --no-cache-save disables saving
        extract_only=args.extract_only,
        qa_parallel=args.qa_parallel,
        llm_score=args.llm_score,  # Use LLM-based semantic scoring
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
