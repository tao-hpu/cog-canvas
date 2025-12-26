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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

from experiments.runner import Agent, AgentResponse
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
    """Result of scoring a LoCoMo answer."""

    keyword_overlap: float  # Fraction of answer keywords found (0-1)
    exact_match: bool  # Whether exact answer appears in response
    found_keywords: List[str]
    missing_keywords: List[str]
    answer: str
    ground_truth: str

    @property
    def passed(self) -> bool:
        """Consider passed if exact match OR keyword overlap >= 60%."""
        return self.exact_match or self.keyword_overlap >= 0.6


def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text for comparison.

    Strategy:
    - Split on whitespace and punctuation
    - Lowercase
    - Filter out very short tokens (< 2 chars)
    - Remove common stop words

    Args:
        text: Text to extract keywords from

    Returns:
        List of keywords
    """
    # Common stop words to ignore
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "the",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
    }

    # Tokenize: split on whitespace and punctuation
    tokens = re.findall(r"\b\w+\b", text.lower())

    # Filter
    keywords = [t for t in tokens if len(t) >= 2 and t not in stop_words]

    return keywords


def score_locomo_answer(answer: str, ground_truth: str) -> LoCoMoScoreResult:
    """
    Score answer based on keyword overlap and exact match.

    Scoring approach:
    1. Exact match: Check if ground truth appears in answer (case-insensitive)
    2. Keyword overlap: Extract keywords from both, compute overlap

    Args:
        answer: The model's answer
        ground_truth: Expected answer from LoCoMo

    Returns:
        LoCoMoScoreResult with scoring details
    """
    answer_lower = answer.lower().strip()
    truth_lower = ground_truth.lower().strip()

    # Exact match check
    exact_match = truth_lower in answer_lower

    # Keyword overlap
    truth_keywords = extract_keywords(truth_lower)
    answer_keywords = extract_keywords(answer_lower)

    found = []
    missing = []

    for kw in truth_keywords:
        if kw in answer_keywords:
            found.append(kw)
        else:
            missing.append(kw)

    overlap = len(found) / len(truth_keywords) if truth_keywords else 0.0

    return LoCoMoScoreResult(
        keyword_overlap=overlap,
        exact_match=exact_match,
        found_keywords=found,
        missing_keywords=missing,
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
    def avg_keyword_overlap(self) -> float:
        """Average keyword overlap across all questions."""
        if not self.question_results:
            return 0.0
        return sum(r.score.keyword_overlap for r in self.question_results) / len(
            self.question_results
        )

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
    def overall_keyword_overlap(self) -> float:
        """Overall keyword overlap."""
        if not self.conversation_results:
            return 0.0
        return sum(c.avg_keyword_overlap for c in self.conversation_results) / len(
            self.conversation_results
        )

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
            "keyword_overlap": f"{self.overall_keyword_overlap:.1%}",
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
                    "avg_keyword_overlap": c.avg_keyword_overlap,
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
                            "keyword_overlap": q.score.keyword_overlap,
                            "exact_match": q.score.exact_match,
                            "passed": q.score.passed,
                            "found_keywords": q.score.found_keywords,
                            "missing_keywords": q.score.missing_keywords,
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
    ):
        """
        Initialize LoCoMo runner.

        Args:
            dataset_path: Path to LoCoMo JSON file
            compression_at_middle: If True, compress at conversation midpoint
            compression_turn: Fixed compression turn (overrides compression_at_middle)
            retain_recent: Number of recent turns to retain after compression
            rolling_interval: Interval for rolling compression (0 to disable)
        """
        self.compression_at_middle = compression_at_middle
        self.fixed_compression_turn = compression_turn
        self.retain_recent = retain_recent
        self.conversations = self._load_dataset(dataset_path)
        self.rolling_interval = rolling_interval

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
            )
        else:
            for i, conv in enumerate(conversations):
                if verbose >= 1:
                    print(f"[{i+1}/{len(conversations)}] Conversation {conv.id}")

                result = self._run_single_conversation(
                    agent, conv, verbose, max_questions_per_conv, categories,
                )
                results.append(result)

                if verbose >= 1:
                    print(
                        f"    => Accuracy: {result.accuracy:.0%} | "
                        f"Exact: {result.exact_match_rate:.0%} | "
                        f"Overlap: {result.avg_keyword_overlap:.0%}"
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
                    categories=categories
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

    def _run_single_conversation(
        self,
        agent: Agent,
        conv: LoCoMoConversation,
        verbose: int = 0,
        max_questions: Optional[int] = None,
        categories: Optional[List[int]] = None,
    ) -> LoCoMoConversationResult:
        """Run experiment on single conversation."""
        agent.reset()
        start_time = time.time()

        # Decide strategy: Rolling vs Single
        is_rolling = self.rolling_interval > 0

        # --- PROCESSING LOOP ---
        
        current_buffer = [] # To track turns for rolling compression context
        
        if is_rolling:
            # === ROLLING COMPRESSION STRATEGY ===
            if verbose >= 2:
                print(f"    Running Rolling Compression (interval={self.rolling_interval})...")
            
            for i, turn in enumerate(conv.turns):
                agent.process_turn(turn)
                current_buffer.append(turn)
                
                # Check for compression trigger
                # Trigger at interval, but NOT at the very last turn (leave that for QA state if needed)
                # Actually, we SHOULD compress at the end too if we want to enforce strict retention
                if (i + 1) % self.rolling_interval == 0:
                    retained_turns = current_buffer[-self.retain_recent:]
                    agent.on_compression(retained_turns)
                    
                    # Update buffer to reflect that older turns are gone from immediate context
                    current_buffer = list(retained_turns)
                    
                    if verbose >= 3:
                        print(f"      [Rolling] Compressed at turn {turn.turn_id}, retained {len(retained_turns)} turns")
            
            # Final compression to ensure state is consistent before QA (only retain last N)
            # This simulates that we answered questions AFTER the conversation ended, with limited memory
            retained_turns = current_buffer[-self.retain_recent:]
            agent.on_compression(retained_turns)
            compression_turn = len(conv.turns) # Logic compression point is the end

        else:
            # === SINGLE COMPRESSION STRATEGY (Legacy) ===
            
            # Determine compression point
            if self.fixed_compression_turn:
                compression_turn = self.fixed_compression_turn
            else:
                compression_turn = conv.get_compression_point()
            compression_turn = min(compression_turn, len(conv.turns))

            if verbose >= 2:
                print(f"    Compression at turn {compression_turn}/{len(conv.turns)}")

            # Phase 1: Pre-compression
            pre_turns = [t for t in conv.turns if t.turn_id <= compression_turn]
            for i, turn in enumerate(pre_turns):
                agent.process_turn(turn)
            
            # Phase 2: Compression
            retained_turns = [
                t for t in conv.turns
                if t.turn_id > compression_turn - self.retain_recent
                and t.turn_id <= compression_turn
            ]
            agent.on_compression(retained_turns)

            # Phase 3: Post-compression
            post_turns = [t for t in conv.turns if t.turn_id > compression_turn]
            for i, turn in enumerate(post_turns):
                agent.process_turn(turn)
        
        # --- QA PHASE ---
        return self._run_qa_phase(
            agent, conv, compression_turn, start_time, verbose, max_questions, categories
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
        """Run the Question-Answering phase."""
        
        qa_pairs = conv.qa_pairs

        # Filter by categories
        if categories:
            qa_pairs = [qa for qa in qa_pairs if qa.category in categories]

        # Limit number of questions
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]

        if verbose >= 2:
            print(f"    Phase 4: Answering {len(qa_pairs)} questions...")

        question_results = []
        for qi, qa in enumerate(qa_pairs):
            q_start = time.time()
            response = agent.answer_question(qa.question)
            latency = (time.time() - q_start) * 1000

            score = score_locomo_answer(response.answer, qa.answer)

            # Map evidence IDs to turn numbers
            evidence_turns = [
                conv.dialogue_id_to_turn.get(eid, -1) for eid in qa.evidence
            ]

            if verbose >= 2:
                status = "✓" if score.passed else "✗"
                score_display = "EXACT" if score.exact_match else f"{score.keyword_overlap:.0%}"
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

    args = parser.parse_args()

    # Create agent and agent factory
    agent_factory = None
    agent = None

    if args.agent.startswith("cogcanvas"):
        from experiments.agents.cogcanvas_agent import CogCanvasAgent

        # Default Full Config (SOTA) - Updated with Top-K=10 and max_hops=3 (v3.1 optimization)
        config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": True,
            "retrieval_method": "hybrid",
            "prompt_style": "cot",
            "retrieval_top_k": 10, # Optimized: reduced from 20 to reduce noise
            "graph_hops": 3, # Optimized: reduced from 5 to reduce noise
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
