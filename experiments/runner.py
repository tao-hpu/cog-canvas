"""
Experiment Runner for CogCanvas Evaluation.

Core components:
1. Agent Interface: Abstract base class for all systems being evaluated
2. Truncation Simulation: Mimics context compression at turn 40
3. Deterministic Scoring: Exact match + fuzzy match against ground truth

Usage:
    python -m experiments.runner --agent cogcanvas --samples 10
"""

import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from rapidfuzz import fuzz
import threading

from experiments.data_gen import (
    EvaluationDataset,
    SyntheticConversation,
    PlantedFact,
    ConversationTurn,
)


# =============================================================================
# Agent Interface
# =============================================================================

@dataclass
class AgentResponse:
    """Response from an agent for a single question."""
    answer: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """
    Abstract base class for all evaluation agents.

    Each agent must implement:
    - process_turn(): Handle a single conversation turn
    - answer_question(): Answer a recall question
    - on_compression(): Handle the compression event (truncation)
    - reset(): Reset state between conversations
    """

    @abstractmethod
    def process_turn(self, turn: ConversationTurn) -> None:
        """
        Process a single conversation turn.

        Called for each turn in the conversation. The agent should
        update its internal state (e.g., extract objects, store history).

        Args:
            turn: The conversation turn to process
        """
        pass

    @abstractmethod
    def answer_question(self, question: str) -> AgentResponse:
        """
        Answer a recall question about the conversation.

        Args:
            question: The question to answer

        Returns:
            AgentResponse with the answer and metadata
        """
        pass

    @abstractmethod
    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle the compression event.

        Called at the compression point (turn 40). The agent receives
        only the retained turns (last N turns) and must update its state.

        For baseline agents: This truncates their history.
        For CogCanvas: History is truncated but canvas objects are preserved.

        Args:
            retained_turns: The turns that survive compression
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's state between conversations."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the agent."""
        pass


# =============================================================================
# Scoring
# =============================================================================

@dataclass
class ScoreResult:
    """Result of scoring a single answer."""
    exact_match: bool
    fuzzy_score: float  # 0-100
    answer: str
    ground_truth: str

    @property
    def passed(self) -> bool:
        """Consider passed if exact match OR fuzzy score >= 80."""
        return self.exact_match or self.fuzzy_score >= 80


def score_answer(answer: str, ground_truth: str) -> ScoreResult:
    """
    Score an answer against ground truth using exact and fuzzy matching.

    Scoring criteria:
    1. Exact Match: Ground truth appears exactly in answer (case-insensitive)
    2. Fuzzy Score: RapidFuzz partial_ratio (0-100)

    Args:
        answer: The model's answer
        ground_truth: Expected answer

    Returns:
        ScoreResult with match details
    """
    answer_lower = answer.lower().strip()
    truth_lower = ground_truth.lower().strip()

    # Exact match: ground truth is substring of answer
    exact_match = truth_lower in answer_lower

    # Fuzzy match: partial ratio handles substring matching well
    fuzzy_score = fuzz.partial_ratio(truth_lower, answer_lower)

    return ScoreResult(
        exact_match=exact_match,
        fuzzy_score=fuzzy_score,
        answer=answer,
        ground_truth=ground_truth,
    )


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class FactResult:
    """Result for a single planted fact."""
    fact_id: str
    fact_type: str
    turn_planted: int
    question: str
    ground_truth: str
    answer: str
    score: ScoreResult
    latency_ms: float


@dataclass
class ConversationResult:
    """Result for a single conversation."""
    conversation_id: str
    fact_results: List[FactResult]
    total_extraction_time_ms: float
    compression_turn: int

    @property
    def recall_rate(self) -> float:
        """Fraction of facts correctly recalled."""
        if not self.fact_results:
            return 0.0
        passed = sum(1 for r in self.fact_results if r.score.passed)
        return passed / len(self.fact_results)

    @property
    def exact_match_rate(self) -> float:
        """Fraction of facts with exact match."""
        if not self.fact_results:
            return 0.0
        exact = sum(1 for r in self.fact_results if r.score.exact_match)
        return exact / len(self.fact_results)

    @property
    def avg_fuzzy_score(self) -> float:
        """Average fuzzy score across all facts."""
        if not self.fact_results:
            return 0.0
        return sum(r.score.fuzzy_score for r in self.fact_results) / len(self.fact_results)


@dataclass
class ExperimentResult:
    """Result for a complete experiment run."""
    agent_name: str
    conversation_results: List[ConversationResult]
    config: Dict[str, Any]
    timestamp: str

    @property
    def overall_recall_rate(self) -> float:
        """Overall recall rate across all conversations."""
        if not self.conversation_results:
            return 0.0
        return sum(c.recall_rate for c in self.conversation_results) / len(self.conversation_results)

    @property
    def overall_exact_match_rate(self) -> float:
        """Overall exact match rate."""
        if not self.conversation_results:
            return 0.0
        return sum(c.exact_match_rate for c in self.conversation_results) / len(self.conversation_results)

    @property
    def overall_avg_fuzzy_score(self) -> float:
        """Overall average fuzzy score."""
        if not self.conversation_results:
            return 0.0
        return sum(c.avg_fuzzy_score for c in self.conversation_results) / len(self.conversation_results)

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment results."""
        return {
            "agent": self.agent_name,
            "num_conversations": len(self.conversation_results),
            "recall_rate": f"{self.overall_recall_rate:.2%}",
            "exact_match_rate": f"{self.overall_exact_match_rate:.2%}",
            "avg_fuzzy_score": f"{self.overall_avg_fuzzy_score:.1f}",
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
                    "recall_rate": c.recall_rate,
                    "exact_match_rate": c.exact_match_rate,
                    "avg_fuzzy_score": c.avg_fuzzy_score,
                    "facts": [
                        {
                            "id": f.fact_id,
                            "type": f.fact_type,
                            "turn_planted": f.turn_planted,
                            "question": f.question,
                            "ground_truth": f.ground_truth,
                            "answer": f.answer,
                            "exact_match": f.score.exact_match,
                            "fuzzy_score": f.score.fuzzy_score,
                            "passed": f.score.passed,
                            "latency_ms": f.latency_ms,
                        }
                        for f in c.fact_results
                    ],
                }
                for c in self.conversation_results
            ],
        }


class ExperimentRunner:
    """
    Runs evaluation experiments with truncation simulation.

    Experiment flow:
    1. Load dataset
    2. For each conversation:
       a. Process turns 1 to compression_turn
       b. Trigger compression (truncate to last N turns)
       c. Process remaining turns
       d. Ask recall questions about planted facts
       e. Score answers
    """

    def __init__(
        self,
        dataset: EvaluationDataset,
        compression_turn: int = 40,
        retain_recent: int = 5,
    ):
        """
        Initialize the experiment runner.

        Args:
            dataset: The evaluation dataset to use
            compression_turn: Turn at which to simulate compression
            retain_recent: Number of recent turns to retain after compression
        """
        self.dataset = dataset
        self.compression_turn = compression_turn
        self.retain_recent = retain_recent

    def run(
        self,
        agent: Agent,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        max_workers: int = 1,
        agent_factory: Optional[callable] = None,
    ) -> ExperimentResult:
        """
        Run the experiment with the given agent.

        Args:
            agent: The agent to evaluate (used when max_workers=1)
            num_samples: Number of conversations to run (None = all)
            verbose: Whether to print progress
            max_workers: Number of parallel workers (default 1 = sequential)
            agent_factory: Factory function to create agent instances for parallel execution.
                           Required when max_workers > 1.

        Returns:
            ExperimentResult with all results
        """
        conversations = self.dataset.conversations
        if num_samples:
            conversations = conversations[:num_samples]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment with {agent.name}")
            print(f"Conversations: {len(conversations)}")
            print(f"Compression at turn {self.compression_turn}, retain {self.retain_recent} turns")
            print(f"Max workers: {max_workers}")
            print(f"{'='*60}\n")

        results = []

        if max_workers > 1:
            # Parallel execution
            if agent_factory is None:
                raise ValueError("agent_factory is required for parallel execution (max_workers > 1)")

            results = self._run_parallel(conversations, agent_factory, max_workers, verbose)
        else:
            # Sequential execution (original behavior)
            for i, conv in enumerate(conversations):
                if verbose:
                    print(f"[{i+1}/{len(conversations)}] Conversation {conv.id}")

                result = self._run_single_conversation(agent, conv, verbose=verbose)
                results.append(result)

                if verbose:
                    print(f"    => Recall: {result.recall_rate:.0%} | Exact: {result.exact_match_rate:.0%} | Time: {result.total_extraction_time_ms:.0f}ms")

        from datetime import datetime
        experiment_result = ExperimentResult(
            agent_name=agent.name,
            conversation_results=results,
            config={
                "compression_turn": self.compression_turn,
                "retain_recent": self.retain_recent,
                "num_samples": num_samples or len(self.dataset.conversations),
            },
            timestamp=datetime.now().isoformat(),
        )

        if verbose:
            print(f"\n{'='*60}")
            print("RESULTS SUMMARY")
            print(f"{'='*60}")
            for k, v in experiment_result.summary().items():
                print(f"  {k}: {v}")

        return experiment_result

    def _run_parallel(
        self,
        conversations: List[SyntheticConversation],
        agent_factory: callable,
        max_workers: int,
        verbose: bool,
    ) -> List[ConversationResult]:
        """
        Run conversations in parallel using ThreadPoolExecutor.

        Each worker gets its own agent instance to avoid state conflicts.
        """
        results = [None] * len(conversations)
        completed = [0]  # Use list for mutable counter in closure
        lock = threading.Lock()

        def process_conv(idx: int, conv: SyntheticConversation) -> Tuple[int, ConversationResult]:
            # Create a fresh agent instance for this worker
            agent = agent_factory()
            result = self._run_single_conversation(agent, conv, verbose=False)

            with lock:
                completed[0] += 1
                if verbose:
                    print(f"[{completed[0]}/{len(conversations)}] {conv.id} => Recall: {result.recall_rate:.0%} | Exact: {result.exact_match_rate:.0%}")

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
        conv: SyntheticConversation,
        verbose: bool = False,
    ) -> ConversationResult:
        """Run experiment on a single conversation."""
        agent.reset()

        total_extraction_time = 0.0
        num_turns_to_process = min(len(conv.turns), self.compression_turn)

        # Phase 1: Process turns up to compression point
        if verbose:
            print(f"\n    Phase 1: Processing {num_turns_to_process} turns...", end="", flush=True)

        for i, turn in enumerate(conv.turns):
            if turn.turn_id <= self.compression_turn:
                start = time.time()
                agent.process_turn(turn)
                elapsed = (time.time() - start) * 1000
                total_extraction_time += elapsed

                # Progress indicator every 10 turns
                if verbose and (i + 1) % 10 == 0:
                    print(f" {i+1}", end="", flush=True)

        if verbose:
            print(f" done ({total_extraction_time:.0f}ms)")

        # Phase 2: Simulate compression
        if verbose:
            print(f"    Phase 2: Compression (keeping last {self.retain_recent} turns)...", end="", flush=True)

        retained_turns = [
            t for t in conv.turns
            if t.turn_id > self.compression_turn - self.retain_recent
            and t.turn_id <= self.compression_turn
        ]
        agent.on_compression(retained_turns)

        if verbose:
            print(" done")

        # Phase 3: Process remaining turns (if any beyond compression)
        remaining_turns = [t for t in conv.turns if t.turn_id > self.compression_turn]
        if remaining_turns and verbose:
            print(f"    Phase 3: Processing {len(remaining_turns)} post-compression turns...", end="", flush=True)

        for turn in remaining_turns:
            start = time.time()
            agent.process_turn(turn)
            total_extraction_time += (time.time() - start) * 1000

        if remaining_turns and verbose:
            print(" done")

        # Phase 4: Ask recall questions
        if verbose:
            print(f"    Phase 4: Testing {len(conv.planted_facts)} facts...")

        fact_results = []
        for fact in conv.planted_facts:
            start = time.time()
            response = agent.answer_question(fact.test_question)
            latency = (time.time() - start) * 1000

            score = score_answer(response.answer, fact.ground_truth)

            if verbose:
                status = "✓" if score.passed else "✗"
                print(f"      {status} [T{fact.turn_id:2d}] {fact.ground_truth[:30]:30s} -> {score.fuzzy_score:.0f}%")

            fact_results.append(FactResult(
                fact_id=fact.id,
                fact_type=fact.fact_type,
                turn_planted=fact.turn_id,
                question=fact.test_question,
                ground_truth=fact.ground_truth,
                answer=response.answer,
                score=score,
                latency_ms=latency,
            ))

        return ConversationResult(
            conversation_id=conv.id,
            fact_results=fact_results,
            total_extraction_time_ms=total_extraction_time,
            compression_turn=self.compression_turn,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run CogCanvas evaluation experiments")
    parser.add_argument(
        "--dataset", "-d",
        default="experiments/data/eval_set.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--agent", "-a",
        choices=["cogcanvas", "native", "summarization", "rag"],
        default="cogcanvas",
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to run (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--compression-turn",
        type=int,
        default=40,
        help="Turn at which to simulate compression",
    )
    parser.add_argument(
        "--retain-recent",
        type=int,
        default=5,
        help="Number of recent turns to retain after compression",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for running conversations (default: 1 = sequential)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = EvaluationDataset.load(args.dataset)
    print(f"Loaded {len(dataset.conversations)} conversations")

    # Create agent and agent factory
    agent_factory = None

    if args.agent == "cogcanvas":
        from experiments.agents.cogcanvas_agent import CogCanvasAgent
        agent = CogCanvasAgent()
        agent_factory = lambda: CogCanvasAgent()
    elif args.agent == "native":
        from experiments.agents.native_agent import NativeAgent
        agent = NativeAgent(retain_recent=args.retain_recent)
        agent_factory = lambda: NativeAgent(retain_recent=args.retain_recent)
    elif args.agent == "summarization":
        from experiments.agents.summarization_agent import SummarizationAgent
        agent = SummarizationAgent(retain_recent=args.retain_recent)
        agent_factory = lambda: SummarizationAgent(retain_recent=args.retain_recent)
    else:
        raise NotImplementedError(f"Agent '{args.agent}' not implemented yet")

    # Run experiment
    runner = ExperimentRunner(
        dataset=dataset,
        compression_turn=args.compression_turn,
        retain_recent=args.retain_recent,
    )

    result = runner.run(
        agent,
        num_samples=args.samples,
        max_workers=args.max_workers,
        agent_factory=agent_factory,
    )

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
