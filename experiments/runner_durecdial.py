"""
DuRecDial 2.0 Experiment Runner for CogCanvas Evaluation.

This runner evaluates agents on the DuRecDial 2.0 dataset (adapted to LoCoMo format).

DuRecDial Characteristics:
- Goal-driven recommendation dialogues
- Adapted to include Single-hop, Multi-hop, and Temporal questions

Evaluation Strategy:
1. Process conversation turns up to compression point (middle of conversation)
2. Trigger compression
3. Process remaining turns
4. Ask generated questions
5. Score using keyword overlap and exact match
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
# CHANGE: Import from durecdial_adapter instead of locomo_adapter
from experiments.durecdial_adapter import (
    load_durecdial_file,
    convert_durecdial_to_locomo,
    LoCoMoConversation, # We reuse these data structures
    LoCoMoQAPair,
)
# We can reuse the result classes from runner_locomo if we import them, 
# or just redefine them here to keep it self-contained. 
# For simplicity and to avoid circular deps if any, I'll redefine the scoring/result classes 
# or better yet, import the generic ones if they existed. 
# runner_locomo defined them inline. I will copy them.

from experiments.extraction_cache import (
    ExtractionCache,
    ExtractionConfig,
)

# =============================================================================
# Scoring (Copied from runner_locomo.py)
# =============================================================================

@dataclass
class LoCoMoScoreResult:
    """Result of scoring an answer."""
    keyword_overlap: float
    exact_match: bool
    found_keywords: List[str]
    missing_keywords: List[str]
    answer: str
    ground_truth: str

    @property
    def passed(self) -> bool:
        return self.exact_match or self.keyword_overlap >= 0.6

def extract_keywords(text: str) -> List[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "the", "this", "but", "they", "have", "had", "what", "when", "where", "who", "which", "why", "how",
    }
    tokens = re.findall(r"\b\w+\b", text.lower())
    keywords = [t for t in tokens if len(t) >= 2 and t not in stop_words]
    return keywords

def score_answer(answer: str, ground_truth: str) -> LoCoMoScoreResult:
    answer_lower = answer.lower().strip()
    truth_lower = ground_truth.lower().strip()
    exact_match = truth_lower in answer_lower
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
class DuRecDialQuestionResult:
    question: str
    category: int
    category_name: str
    evidence_turns: List[int]
    ground_truth: str
    answer: str
    score: LoCoMoScoreResult
    latency_ms: float

@dataclass
class DuRecDialConversationResult:
    conversation_id: str
    num_turns: int
    compression_turn: int
    question_results: List[DuRecDialQuestionResult]
    total_time_ms: float

    @property
    def accuracy(self) -> float:
        if not self.question_results: return 0.0
        return sum(1 for r in self.question_results if r.score.passed) / len(self.question_results)

    @property
    def exact_match_rate(self) -> float:
        if not self.question_results: return 0.0
        return sum(1 for r in self.question_results if r.score.exact_match) / len(self.question_results)

    @property
    def avg_keyword_overlap(self) -> float:
        if not self.question_results: return 0.0
        return sum(r.score.keyword_overlap for r in self.question_results) / len(self.question_results)

    def accuracy_by_category(self, category: int) -> float:
        category_results = [r for r in self.question_results if r.category == category]
        if not category_results: return 0.0
        return sum(1 for r in category_results if r.score.passed) / len(category_results)
@dataclass
class DuRecDialExperimentResult:
    agent_name: str
    conversation_results: List[DuRecDialConversationResult]
    config: Dict[str, Any]
    timestamp: str

    @property
    def overall_accuracy(self) -> float:
        if not self.conversation_results: return 0.0
        return sum(c.accuracy for c in self.conversation_results) / len(self.conversation_results)

    @property
    def overall_exact_match_rate(self) -> float:
        if not self.conversation_results: return 0.0
        return sum(c.exact_match_rate for c in self.conversation_results) / len(self.conversation_results)

    @property
    def overall_keyword_overlap(self) -> float:
        if not self.conversation_results: return 0.0
        return sum(c.avg_keyword_overlap for c in self.conversation_results) / len(self.conversation_results)

    def accuracy_by_category(self, category: int) -> float:
        category_results = []
        for conv in self.conversation_results:
            category_results.extend([r for r in conv.question_results if r.category == category])
        if not category_results: return 0.0
        return sum(1 for r in category_results if r.score.passed) / len(category_results)

    def summary(self) -> Dict[str, Any]:
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
                        } for q in c.question_results
                    ],
                } for c in self.conversation_results
            ],
        }

# =============================================================================
# Runner
# =============================================================================

class DuRecDialExperimentRunner:
    """Runs DuRecDial evaluation experiments."""

    def __init__(
        self,
        dataset_path: str,
        compression_at_middle: bool = True,
        compression_turn: Optional[int] = None,
        retain_recent: int = 5,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
        min_turns: int = 30,
    ):
        self.compression_at_middle = compression_at_middle
        self.fixed_compression_turn = compression_turn
        self.retain_recent = retain_recent
        self.min_turns = min_turns
        self.conversations = self._load_dataset(dataset_path)

        self.use_cache = use_cache
        self.cache_dir = cache_dir or "experiments/cache/extraction"
        self._cache = ExtractionCache(self.cache_dir) if use_cache else None

    def _load_dataset(self, path: str) -> List[LoCoMoConversation]:
        print(f"Loading DuRecDial dataset from {path}...")
        raw_data = load_durecdial_file(path)
        conversations = convert_durecdial_to_locomo(raw_data, min_turns=self.min_turns)
        print(f"Loaded {len(conversations)} conversations (min_turns={self.min_turns})")
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
    ) -> DuRecDialExperimentResult:
        conversations = self.conversations
        if num_samples:
            conversations = conversations[:num_samples]

        extraction_config = None
        if self.use_cache and hasattr(agent, 'get_extraction_config'):
            extraction_config = agent.get_extraction_config()

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"DuRecDial Experiment: {agent.name}")
            print(f"Conversations: {len(conversations)}")
            print(f"Compression: {'middle' if self.compression_at_middle else f'turn {self.fixed_compression_turn}'}")
            print(f"Retain recent: {self.retain_recent} turns")
            print(f"Max workers: {max_workers}")
            print(f"{'='*60}\n")

        results = []

        if max_workers > 1:
            if agent_factory is None:
                raise ValueError("agent_factory is required for parallel execution")
            results = self._run_parallel(
                conversations, agent_factory, max_workers, verbose, max_questions_per_conv, categories, extraction_config
            )
        else:
            for i, conv in enumerate(conversations):
                if verbose >= 1:
                    print(f"[{i+1}/{len(conversations)}] Conversation {conv.id}")
                result = self._run_single_conversation(
                    agent, conv, verbose, max_questions_per_conv, categories, extraction_config
                )
                results.append(result)
                if verbose >= 1:
                    print(f"    => Accuracy: {result.accuracy:.0%} | Exact: {result.exact_match_rate:.0%} | Overlap: {result.avg_keyword_overlap:.0%}")

        experiment_result = DuRecDialExperimentResult(
            agent_name=agent.name,
            conversation_results=results,
            config={
                "compression_at_middle": self.compression_at_middle,
                "fixed_compression_turn": self.fixed_compression_turn,
                "retain_recent": self.retain_recent,
                "num_samples": num_samples or len(self.conversations),
                "benchmark_type": "durecdial",
            },
            timestamp=datetime.now().isoformat(),
        )

        if verbose >= 1:
            print(f"\n{'='*60}")
            print("DURECDIAL RESULTS SUMMARY")
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
        categories: Optional[List[int]],
        extraction_config: Optional[ExtractionConfig],
    ) -> List[DuRecDialConversationResult]:
        results = [None] * len(conversations)
        completed = [0]
        lock = threading.Lock()

        def process_conv(idx: int, conv: LoCoMoConversation) -> Tuple[int, DuRecDialConversationResult]:
            conv_verbose = verbose if verbose >= 2 else 0
            try:
                agent = agent_factory()
                result = self._run_single_conversation(
                    agent, conv, verbose=conv_verbose, max_questions=max_questions_per_conv, categories=categories, extraction_config=extraction_config
                )
            except Exception as e:
                print(f"Error in conversation {conv.id}: {e}")
                import traceback
                traceback.print_exc()
                result = DuRecDialConversationResult(
                    conversation_id=conv.id, num_turns=len(conv.turns), compression_turn=0, question_results=[], total_time_ms=0
                )

            with lock:
                completed[0] += 1
                if verbose >= 1:
                    print(f"[{completed[0]}/{len(conversations)}] {conv.id} => Accuracy: {result.accuracy:.0%}")
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_conv, i, conv): i for i, conv in enumerate(conversations)}
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
        extraction_config: Optional[ExtractionConfig] = None,
    ) -> DuRecDialConversationResult:
        agent.reset()
        start_time = time.time()
        
        # Compression
        compression_turn = self.fixed_compression_turn or conv.get_compression_point()
        compression_turn = min(compression_turn, len(conv.turns))

        # Check cache
        cache_hit = False
        if self.use_cache and self._cache and extraction_config:
            if self._cache.has(conv.id, extraction_config):
                cached_state = self._cache.load(conv.id, extraction_config)
                if cached_state and hasattr(agent, 'restore_from_cache'):
                    agent.restore_from_cache(cached_state)
                    cache_hit = True

        if not cache_hit:
            pre_turns = [t for t in conv.turns if t.turn_id <= compression_turn]
            for turn in pre_turns:
                agent.process_turn(turn)
            
            retained = [t for t in conv.turns if t.turn_id > compression_turn - self.retain_recent and t.turn_id <= compression_turn]
            agent.on_compression(retained)
            
            post_turns = [t for t in conv.turns if t.turn_id > compression_turn]
            for turn in post_turns:
                agent.process_turn(turn)

            if self.use_cache and self._cache and extraction_config and hasattr(agent, 'get_canvas_state'):
                canvas_state = agent.get_canvas_state()
                if canvas_state:
                    self._cache.save(conv.id, extraction_config, canvas_state)

        # Questions
        qa_pairs = conv.qa_pairs
        if categories:
            qa_pairs = [qa for qa in qa_pairs if qa.category in categories]
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]

        question_results = []
        for qa in qa_pairs:
            q_start = time.time()
            response = agent.answer_question(qa.question)
            latency = (time.time() - q_start) * 1000
            score = score_answer(response.answer, qa.answer)
            
            if verbose >= 2:
                status = "✓" if score.passed else "✗"
                print(f"      {status} [{conv.id}] [{qa.category_name}] {qa.question[:35]:35s} -> {score.keyword_overlap:.0%}")

            question_results.append(DuRecDialQuestionResult(
                question=qa.question,
                category=qa.category,
                category_name=qa.category_name,
                evidence_turns=[conv.dialogue_id_to_turn.get(eid, -1) for eid in qa.evidence],
                ground_truth=qa.answer,
                answer=response.answer,
                score=score,
                latency_ms=latency
            ))

        return DuRecDialConversationResult(
            conversation_id=conv.id,
            num_turns=len(conv.turns),
            compression_turn=compression_turn,
            question_results=question_results,
            total_time_ms=(time.time() - start_time) * 1000,
        )

# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    import os
    from dotenv import load_dotenv

    # Load .env
    project_root = Path(__file__).parent.parent.parent # Adjusted for cog-canvas/experiments/runner_durecdial.py if necessary?
    # Wait, __file__ is experiments/runner_durecdial.py (inside cog-canvas root)
    # So parent is experiments, parent.parent is cog-canvas
    # The .env is in cog-canvas/
    load_dotenv(Path(__file__).parent.parent / ".env")

    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
    os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

    parser = argparse.ArgumentParser(description="Run DuRecDial evaluation")
    parser.add_argument("--dataset", default="experiments/data/durecdial_sample.jsonl")
    parser.add_argument("--agent", default="cogcanvas", help="Agent to evaluate (see runner_locomo.py for options)")
    parser.add_argument("--samples", "-n", type=int, default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--min-turns", type=int, default=30)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--workers", "-w", type=int, default=1)
    
    # ... (Add other args similar to runner_locomo if needed) 
    
    args = parser.parse_args()

    # Reuse agent factory logic from runner_locomo? 
    # For now, let's just support basic 'cogcanvas' to test.
    # In a real scenario, we'd copy the massive switch-case or import it.
    
    agent = None
    agent_factory = None
    
    # Quick dirty import of agent classes
    if args.agent == "cogcanvas":
        from experiments.agents.cogcanvas_agent import CogCanvasAgent
        config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": True,
            "retrieval_method": "hybrid",
            "prompt_style": "cot",
        }
        agent_factory = lambda: CogCanvasAgent(**config)
        agent = agent_factory()
    elif args.agent == "native":
         from experiments.agents.native_agent import NativeAgent
         agent_factory = lambda: NativeAgent(retain_recent=5)
         agent = agent_factory()
    else:
        print(f"Warning: Agent {args.agent} not explicitly configured in this minimal runner. Using NativeAgent fallback.")
        from experiments.agents.native_agent import NativeAgent
        agent_factory = lambda: NativeAgent(retain_recent=5)
        agent = agent_factory()

    runner = DuRecDialExperimentRunner(
        dataset_path=args.dataset,
        min_turns=args.min_turns
    )
    
    result = runner.run(
        agent,
        num_samples=args.samples,
        verbose=args.verbose + 1,
        max_workers=args.workers,
        agent_factory=agent_factory
    )
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
