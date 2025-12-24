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
    # Check if text contains Chinese characters
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    if has_chinese:
        # Simple character-level tokenization for Chinese
        # Remove punctuation and whitespace
        clean_text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        # Return individual characters as keywords (or bigrams if preferred, but chars are safer for recall)
        # Actually, for "剁椒鱼头", we want "剁","椒","鱼","头" to match? 
        # Or just use the whole string as one keyword if it's short?
        # Let's stick to simple character list for overlap calc, similar to ROUGE-1 char level
        return list(clean_text)
    
    # English/Latin processing
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
        min_turns: int = 10,
        id_prefix: str = "drd",
        rolling_interval: int = 40,  # Default to 40 for consistency with LoCoMo
    ):
        self.compression_at_middle = compression_at_middle
        self.fixed_compression_turn = compression_turn
        self.retain_recent = retain_recent
        self.min_turns = min_turns
        self.conversations = self._load_dataset(dataset_path, id_prefix)
        self.rolling_interval = rolling_interval

    def _load_dataset(self, path: str, id_prefix: str) -> List[LoCoMoConversation]:
        """Load and convert DuRecDial dataset."""
        print(f"Loading DuRecDial dataset from {path}...")
        raw_data = load_durecdial_file(path)
        conversations = convert_durecdial_to_locomo(raw_data, min_turns=self.min_turns, id_prefix=id_prefix)
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
        language: str = "en",
    ) -> DuRecDialExperimentResult:
        # ... (keep existing run logic until result creation)
        conversations = self.conversations
        if num_samples:
            conversations = conversations[:num_samples]

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"DuRecDial Experiment: {agent.name}")
            print(f"Language: {language}")
            print(f"Conversations: {len(conversations)}")
            if self.rolling_interval > 0:
                print(f"Strategy: Rolling Compression (interval={self.rolling_interval})")
            else:
                print(f"Strategy: Single Compression ({'middle' if self.compression_at_middle else f'turn {self.fixed_compression_turn}'})")
            print(f"Retain recent: {self.retain_recent} turns")
            print(f"Max workers: {max_workers}")
            print(f"{'='*60}\n")

        results = []

        if max_workers > 1:
            if agent_factory is None:
                raise ValueError("agent_factory is required for parallel execution")
            results = self._run_parallel(
                conversations, agent_factory, max_workers, verbose, max_questions_per_conv, categories, language
            )
        else:
            for i, conv in enumerate(conversations):
                if verbose >= 1:
                    print(f"[{i+1}/{len(conversations)}] Conversation {conv.id}")
                result = self._run_single_conversation(
                    agent, conv, verbose, max_questions_per_conv, categories, language
                )
                results.append(result)
                if verbose >= 1:
                    print(f"    => Accuracy: {result.accuracy:.0%} | Exact: {result.exact_match_rate:.0%} | Overlap: {result.avg_keyword_overlap:.0%}")

        experiment_result = DuRecDialExperimentResult(
            agent_name=agent.name,
            conversation_results=results,
            config={
                "rolling_interval": self.rolling_interval,
                "compression_at_middle": self.compression_at_middle,
                "fixed_compression_turn": self.fixed_compression_turn,
                "retain_recent": self.retain_recent,
                "num_samples": num_samples or len(self.conversations),
                "benchmark_type": "durecdial",
                "language": language,
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
        categories: Optional[List[int]] = None,
        language: str = "en",
    ) -> List[DuRecDialConversationResult]:
        """Run conversations in parallel."""
        results = [None] * len(conversations)
        completed = [0]
        lock = threading.Lock()

        def process_conv(
            idx: int, conv: LoCoMoConversation
        ) -> Tuple[int, DuRecDialConversationResult]:
            # In parallel mode, only use verbose >= 2 for per-question detail
            conv_verbose = verbose if verbose >= 2 else 0
            try:
                agent = agent_factory()
                if verbose >= 2:
                    with lock:
                        print(f"  [Starting] {conv.id} ({len(conv.turns)} turns, {len(conv.qa_pairs)} questions)")
                result = self._run_single_conversation(
                    agent, conv, verbose=conv_verbose, max_questions=max_questions_per_conv,
                    categories=categories, language=language
                )
            except Exception as e:
                print(f"Error in conversation {conv.id}: {e}, skipping...")
                import traceback
                traceback.print_exc()
                # Return empty result instead of crashing
                result = DuRecDialConversationResult(
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

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_conv, i, conv): i
                for i, conv in enumerate(conversations)
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
        language: str = "en",
    ) -> DuRecDialConversationResult:
        agent.reset()
        start_time = time.time()
        
        # Decide strategy
        is_rolling = self.rolling_interval > 0
        compression_turn = 0 # Placeholder for result

        # --- PROCESSING LOOP ---
        current_buffer = [] 
        
        if is_rolling:
            # === ROLLING COMPRESSION ===
            if verbose >= 2:
                print(f"    Running Rolling Compression (interval={self.rolling_interval})...")
            
            for i, turn in enumerate(conv.turns):
                agent.process_turn(turn)
                current_buffer.append(turn)
                
                if (i + 1) % self.rolling_interval == 0:
                    retained_turns = current_buffer[-self.retain_recent:]
                    agent.on_compression(retained_turns)
                    current_buffer = list(retained_turns)
                    if verbose >= 3:
                        print(f"      [Rolling] Compressed at turn {turn.turn_id}")
            
            # Final state setup
            retained_turns = current_buffer[-self.retain_recent:]
            agent.on_compression(retained_turns)
            compression_turn = len(conv.turns)

        else:
            # === SINGLE COMPRESSION (Legacy) ===
            compression_turn = self.fixed_compression_turn or conv.get_compression_point()
            compression_turn = min(compression_turn, len(conv.turns))

            if verbose >= 2:
                print(f"    Compression at turn {compression_turn}/{len(conv.turns)}")

            pre_turns = [t for t in conv.turns if t.turn_id <= compression_turn]
            for turn in pre_turns:
                agent.process_turn(turn)
            
            retained = [t for t in conv.turns if t.turn_id > compression_turn - self.retain_recent and t.turn_id <= compression_turn]
            agent.on_compression(retained)
            
            post_turns = [t for t in conv.turns if t.turn_id > compression_turn]
            for turn in post_turns:
                agent.process_turn(turn)

        # --- QA PHASE ---
        return self._run_qa_phase(
            agent, conv, compression_turn, start_time, verbose, max_questions, categories, language
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
        language: str,
    ) -> DuRecDialConversationResult:
        qa_pairs = conv.qa_pairs
        if categories:
            qa_pairs = [qa for qa in qa_pairs if qa.category in categories]
        if max_questions:
            qa_pairs = qa_pairs[:max_questions]

        question_results = []
        for qa in qa_pairs:
            q_start = time.time()
            
            question_text = qa.question
            if language == "zh":
                question_text += " (请用中文回答)"

            response = agent.answer_question(question_text)
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
    # ... (imports)
    import argparse
    import os
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
    os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

    parser = argparse.ArgumentParser(description="Run DuRecDial evaluation")
    # ... (existing args)
    parser.add_argument("--dataset", default="experiments/data/durecdial_sample.jsonl")
    parser.add_argument("--agent", default="cogcanvas", help="Agent to evaluate")
    parser.add_argument("--samples", "-n", type=int, default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--min-turns", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--language", default="en", choices=["en", "zh"], help="Language for instructions (en/zh)")
    parser.add_argument("--id-prefix", default="drd", help="Prefix for conversation IDs to match cache (default: drd)")
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--retain-recent", type=int, default=5)
    
    # NEW ARGUMENT
    parser.add_argument(
        "--rolling-interval",
        type=int,
        default=40,
        help="Interval for rolling compression (e.g. 40 turns). 0 to disable.",
    )
    
    args = parser.parse_args()

    # ... (Agent initialization logic)
    agent = None
    agent_factory = None
    
    if args.agent == "cogcanvas":
        from experiments.agents.cogcanvas_agent import CogCanvasAgent
        # UPDATE CONFIG TO MATCH LOCOMO BEST PRACTICES
        config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": True,
            "retrieval_method": "hybrid",
            "prompt_style": "cot",
            "retrieval_top_k": 20, # Updated
        }
        agent_factory = lambda: CogCanvasAgent(**config)
        agent = agent_factory()
    # ... (other agents same as before)
    elif args.agent == "native":
         from experiments.agents.native_agent import NativeAgent
         agent_factory = lambda: NativeAgent(retain_recent=args.retain_recent)
         agent = agent_factory()
    elif args.agent == "summarization":
        from experiments.agents.summarization_agent import SummarizationAgent
        agent_factory = lambda: SummarizationAgent(retain_recent=args.retain_recent)
        agent = agent_factory()
    elif args.agent == "rag":
        from experiments.agents.rag_agent import RagAgent
        agent_factory = lambda: RagAgent(retain_recent=args.retain_recent)
        agent = agent_factory()
    elif args.agent == "memgpt-lite":
        from experiments.agents.memgpt_lite_agent import MemGPTLiteAgent
        agent_factory = lambda: MemGPTLiteAgent(core_memory_size=args.retain_recent)
        agent = agent_factory()
    elif args.agent == "graphrag":
        from experiments.agents.graphrag_agent import create_graphrag_agent
        agent_factory = lambda: create_graphrag_agent(search_method="local")
        agent = agent_factory()
    else:
        print(f"Warning: Agent {args.agent} not explicitly configured. Using NativeAgent fallback.")
        from experiments.agents.native_agent import NativeAgent
        agent_factory = lambda: NativeAgent(retain_recent=args.retain_recent)
        agent = agent_factory()

    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]
    
    runner = DuRecDialExperimentRunner(
        dataset_path=args.dataset,
        min_turns=args.min_turns,
        id_prefix=args.id_prefix,
        rolling_interval=args.rolling_interval # Pass new arg
    )
    
    result = runner.run(
        agent,
        num_samples=args.samples,
        verbose=args.verbose + 1,
        max_workers=args.workers,
        agent_factory=agent_factory,
        categories=categories,
        language=args.language
    )
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
