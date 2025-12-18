"""
Multi-hop Experiment Runner for CogCanvas Evaluation.

This runner evaluates agents on multi-hop reasoning questions that require
understanding relationships between facts (constraints -> decisions).

Question types:
1. Causal: "Why did we choose X?" - requires linking decision to constraint
2. Impact: "What was affected by X?" - requires following constraint to decisions

Scoring:
- Keyword coverage: What fraction of required keywords appear in answer
- All-keywords: Binary - did answer include ALL required keywords
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn


# =============================================================================
# Multi-hop Data Structures
# =============================================================================

@dataclass
class MultiHopFact:
    """A fact with relationship information."""
    id: str
    turn_id: int
    fact_type: str  # constraint or decision
    content: str
    quote: str
    caused_by: List[str]
    leads_to: List[str]


@dataclass
class MultiHopQuestion:
    """A multi-hop reasoning question."""
    id: str
    question: str
    question_type: str  # causal or impact
    required_fact_ids: List[str]
    required_hops: int
    ground_truth_keywords: List[str]
    explanation: str


@dataclass
class MultiHopConversation:
    """A conversation with multi-hop metadata."""
    id: str
    turns: List[ConversationTurn]
    facts: List[MultiHopFact]
    relations: List[Dict]
    questions: List[MultiHopQuestion]


# =============================================================================
# Scoring
# =============================================================================

@dataclass
class MultiHopScoreResult:
    """Result of scoring a multi-hop question answer."""
    keyword_coverage: float  # Fraction of keywords found (0-1)
    all_keywords_found: bool  # Whether ALL keywords were found
    found_keywords: List[str]
    missing_keywords: List[str]
    answer: str

    @property
    def passed(self) -> bool:
        """Consider passed if at least 80% keywords found."""
        return self.keyword_coverage >= 0.8


def score_multihop_answer(answer: str, ground_truth_keywords: List[str]) -> MultiHopScoreResult:
    """
    Score answer based on keyword coverage.

    Args:
        answer: The model's answer
        ground_truth_keywords: Keywords that should appear in answer

    Returns:
        MultiHopScoreResult with coverage details
    """
    answer_lower = answer.lower()
    found = []
    missing = []

    for kw in ground_truth_keywords:
        kw_lower = kw.lower()
        if kw_lower in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)

    coverage = len(found) / len(ground_truth_keywords) if ground_truth_keywords else 0.0

    return MultiHopScoreResult(
        keyword_coverage=coverage,
        all_keywords_found=len(missing) == 0,
        found_keywords=found,
        missing_keywords=missing,
        answer=answer,
    )


# =============================================================================
# Results
# =============================================================================

@dataclass
class MultiHopQuestionResult:
    """Result for a single multi-hop question."""
    question_id: str
    question_type: str
    question: str
    required_hops: int
    ground_truth_keywords: List[str]
    answer: str
    score: MultiHopScoreResult
    latency_ms: float


@dataclass
class MultiHopConversationResult:
    """Result for a single conversation."""
    conversation_id: str
    question_results: List[MultiHopQuestionResult]
    total_time_ms: float

    @property
    def avg_keyword_coverage(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(r.score.keyword_coverage for r in self.question_results) / len(self.question_results)

    @property
    def all_keywords_rate(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(1 for r in self.question_results if r.score.all_keywords_found) / len(self.question_results)

    @property
    def pass_rate(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(1 for r in self.question_results if r.score.passed) / len(self.question_results)


@dataclass
class MultiHopExperimentResult:
    """Result for complete multi-hop experiment."""
    agent_name: str
    conversation_results: List[MultiHopConversationResult]
    config: Dict[str, Any]
    timestamp: str

    @property
    def overall_keyword_coverage(self) -> float:
        if not self.conversation_results:
            return 0.0
        return sum(c.avg_keyword_coverage for c in self.conversation_results) / len(self.conversation_results)

    @property
    def overall_all_keywords_rate(self) -> float:
        if not self.conversation_results:
            return 0.0
        return sum(c.all_keywords_rate for c in self.conversation_results) / len(self.conversation_results)

    @property
    def overall_pass_rate(self) -> float:
        if not self.conversation_results:
            return 0.0
        return sum(c.pass_rate for c in self.conversation_results) / len(self.conversation_results)

    def summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        # Break down by question type
        causal_results = []
        impact_results = []

        for conv in self.conversation_results:
            for qr in conv.question_results:
                if qr.question_type == 'causal':
                    causal_results.append(qr)
                elif qr.question_type == 'impact':
                    impact_results.append(qr)

        causal_coverage = sum(r.score.keyword_coverage for r in causal_results) / len(causal_results) if causal_results else 0
        impact_coverage = sum(r.score.keyword_coverage for r in impact_results) / len(impact_results) if impact_results else 0

        return {
            "agent": self.agent_name,
            "num_conversations": len(self.conversation_results),
            "overall_keyword_coverage": f"{self.overall_keyword_coverage:.1%}",
            "overall_all_keywords_rate": f"{self.overall_all_keywords_rate:.1%}",
            "overall_pass_rate": f"{self.overall_pass_rate:.1%}",
            "causal_coverage": f"{causal_coverage:.1%}",
            "impact_coverage": f"{impact_coverage:.1%}",
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
                    "avg_keyword_coverage": c.avg_keyword_coverage,
                    "all_keywords_rate": c.all_keywords_rate,
                    "pass_rate": c.pass_rate,
                    "questions": [
                        {
                            "id": q.question_id,
                            "type": q.question_type,
                            "question": q.question,
                            "required_hops": q.required_hops,
                            "ground_truth_keywords": q.ground_truth_keywords,
                            "answer": q.answer,
                            "keyword_coverage": q.score.keyword_coverage,
                            "all_keywords_found": q.score.all_keywords_found,
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

class MultiHopExperimentRunner:
    """
    Runs multi-hop evaluation experiments.

    Flow:
    1. Load multi-hop dataset
    2. For each conversation:
       a. Process turns up to compression point
       b. Trigger compression
       c. Ask multi-hop questions
       d. Score based on keyword coverage
    """

    def __init__(
        self,
        dataset_path: str,
        compression_turn: int = 40,
        retain_recent: int = 5,
    ):
        self.compression_turn = compression_turn
        self.retain_recent = retain_recent
        self.conversations = self._load_dataset(dataset_path)

    def _load_dataset(self, path: str) -> List[MultiHopConversation]:
        """Load multi-hop dataset from JSON."""
        with open(path) as f:
            data = json.load(f)

        conversations = []
        for conv_data in data['conversations']:
            turns = [
                ConversationTurn(
                    turn_id=t['turn_id'],
                    user=t['user'],
                    assistant=t['assistant'],
                    contains_fact=t.get('contains_fact_id'),
                    is_distractor=False,
                )
                for t in conv_data['turns']
            ]

            facts = [
                MultiHopFact(
                    id=f['id'],
                    turn_id=f['turn_id'],
                    fact_type=f['fact_type'],
                    content=f['content'],
                    quote=f['quote'],
                    caused_by=f['caused_by'],
                    leads_to=f['leads_to'],
                )
                for f in conv_data['facts']
            ]

            questions = [
                MultiHopQuestion(
                    id=q['id'],
                    question=q['question'],
                    question_type=q['question_type'],
                    required_fact_ids=q['required_fact_ids'],
                    required_hops=q['required_hops'],
                    ground_truth_keywords=q['ground_truth_keywords'],
                    explanation=q['explanation'],
                )
                for q in conv_data['questions']
            ]

            conversations.append(MultiHopConversation(
                id=conv_data['id'],
                turns=turns,
                facts=facts,
                relations=conv_data['relations'],
                questions=questions,
            ))

        return conversations

    def run(
        self,
        agent: Agent,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        max_workers: int = 1,
        agent_factory: Optional[callable] = None,
    ) -> MultiHopExperimentResult:
        """Run multi-hop experiment."""
        conversations = self.conversations
        if num_samples:
            conversations = conversations[:num_samples]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Multi-hop Experiment: {agent.name}")
            print(f"Conversations: {len(conversations)}")
            print(f"Compression at turn {self.compression_turn}, retain {self.retain_recent}")
            print(f"Max workers: {max_workers}")
            print(f"{'='*60}\n")

        results = []

        if max_workers > 1:
            # Parallel execution
            if agent_factory is None:
                raise ValueError("agent_factory is required for parallel execution (max_workers > 1)")

            results = self._run_parallel(conversations, agent_factory, max_workers, verbose)
        else:
            # Sequential execution
            for i, conv in enumerate(conversations):
                if verbose:
                    print(f"[{i+1}/{len(conversations)}] Conversation {conv.id}")

                result = self._run_single_conversation(agent, conv, verbose)
                results.append(result)

                if verbose:
                    print(f"    => Coverage: {result.avg_keyword_coverage:.0%} | All: {result.all_keywords_rate:.0%} | Pass: {result.pass_rate:.0%}")

        experiment_result = MultiHopExperimentResult(
            agent_name=agent.name,
            conversation_results=results,
            config={
                "compression_turn": self.compression_turn,
                "retain_recent": self.retain_recent,
                "num_samples": num_samples or len(self.conversations),
                "benchmark_type": "multi-hop",
            },
            timestamp=datetime.now().isoformat(),
        )

        if verbose:
            print(f"\n{'='*60}")
            print("MULTI-HOP RESULTS SUMMARY")
            print(f"{'='*60}")
            for k, v in experiment_result.summary().items():
                print(f"  {k}: {v}")

        return experiment_result

    def _run_parallel(
        self,
        conversations: List[MultiHopConversation],
        agent_factory: callable,
        max_workers: int,
        verbose: bool,
    ) -> List[MultiHopConversationResult]:
        """
        Run conversations in parallel using ThreadPoolExecutor.

        Each worker gets its own agent instance to avoid state conflicts.
        """
        results = [None] * len(conversations)
        completed = [0]  # Use list for mutable counter in closure
        lock = threading.Lock()

        def process_conv(idx: int, conv: MultiHopConversation) -> Tuple[int, MultiHopConversationResult]:
            # Create a fresh agent instance for this worker
            try:
                agent = agent_factory()
                result = self._run_single_conversation(agent, conv, verbose=False) # Verbose false for parallel to avoid mixing output
            except Exception as e:
                print(f"Error in conversation {conv.id}: {e}")
                import traceback
                traceback.print_exc()
                # Return a dummy failed result or re-raise.
                # For now let's re-raise but the future will catch it.
                raise e

            with lock:
                completed[0] += 1
                if verbose:
                    print(f"[{completed[0]}/{len(conversations)}] {conv.id} => Coverage: {result.avg_keyword_coverage:.0%} | Pass: {result.pass_rate:.0%}")

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
        conv: MultiHopConversation,
        verbose: bool = False,
    ) -> MultiHopConversationResult:
        """Run experiment on single conversation."""
        agent.reset()
        start_time = time.time()

        # Phase 1: Process turns up to compression
        for turn in conv.turns:
            if turn.turn_id <= self.compression_turn:
                agent.process_turn(turn)

        # Phase 2: Compression
        retained_turns = [
            t for t in conv.turns
            if t.turn_id > self.compression_turn - self.retain_recent
            and t.turn_id <= self.compression_turn
        ]
        agent.on_compression(retained_turns)

        # Phase 3: Process remaining turns
        for turn in conv.turns:
            if turn.turn_id > self.compression_turn:
                agent.process_turn(turn)

        # Phase 4: Ask multi-hop questions
        question_results = []
        for q in conv.questions:
            q_start = time.time()
            response = agent.answer_question(q.question)
            latency = (time.time() - q_start) * 1000

            score = score_multihop_answer(response.answer, q.ground_truth_keywords)

            if verbose:
                status = "✓" if score.passed else "✗"
                print(f"      {status} [{q.question_type}] {q.question[:40]:40s} -> {score.keyword_coverage:.0%}")

            question_results.append(MultiHopQuestionResult(
                question_id=q.id,
                question_type=q.question_type,
                question=q.question,
                required_hops=q.required_hops,
                ground_truth_keywords=q.ground_truth_keywords,
                answer=response.answer,
                score=score,
                latency_ms=latency,
            ))

        total_time = (time.time() - start_time) * 1000

        return MultiHopConversationResult(
            conversation_id=conv.id,
            question_results=question_results,
            total_time_ms=total_time,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    import os # Import os for environment variables
    from dotenv import load_dotenv # Import load_dotenv
    from pathlib import Path # Ensure Path is imported
    
    # Load .env file from the project root (cog-canvas directory)
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
    
    # Configure OpenAI API from environment variables
    os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY', '')
    os.environ['OPENAI_API_BASE'] = os.getenv('API_BASE', '')

    parser = argparse.ArgumentParser(description="Run multi-hop evaluation experiments")
    parser.add_argument(
        "--dataset", "-d",
        default="experiments/data/multihop_eval.json",
        help="Path to multi-hop dataset",
    )
    parser.add_argument(
        "--agent", "-a",
        choices=[
            "cogcanvas", "cogcanvas-nograph", 
            "cogcanvas-baseline", "cogcanvas-temporal", "cogcanvas-hybrid", "cogcanvas-cot",
            "native", "summarization", "rag", "memgpt-lite", "graphrag-lite"
        ],
        default="cogcanvas",
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples (default: all)",
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
        help="Number of recent turns to retain",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading multi-hop dataset from {args.dataset}...")

    # Create agent and agent factory
    agent_factory = None
    agent = None # Initial agent for single threaded or just for name

    if args.agent.startswith("cogcanvas"):
        from experiments.agents.cogcanvas_agent import CogCanvasAgent
        
        # Default Full Config (SOTA)
        config = {
            "enable_graph_expansion": True,
            "enable_temporal_heuristic": True,
            "retrieval_method": "hybrid",
            "prompt_style": "cot"
        }
        
        if args.agent == "cogcanvas-nograph":
            config["enable_graph_expansion"] = False
        
        # Ablation Variants
        elif args.agent == "cogcanvas-baseline":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": False,
                "retrieval_method": "semantic",
                "prompt_style": "direct"
            }
        elif args.agent == "cogcanvas-temporal":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": True,
                "retrieval_method": "semantic",
                "prompt_style": "direct"
            }
        elif args.agent == "cogcanvas-hybrid":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": False,
                "retrieval_method": "hybrid",
                "prompt_style": "direct"
            }
        elif args.agent == "cogcanvas-cot":
            config = {
                "enable_graph_expansion": True,
                "enable_temporal_heuristic": False,
                "retrieval_method": "semantic",
                "prompt_style": "cot"
            }
            
        agent_factory = lambda: CogCanvasAgent(**config)
        agent = agent_factory() # Create a dummy for name prop
        
    elif args.agent == "rag":
        from experiments.agents.rag_agent import RagAgent
        agent = RagAgent(retain_recent=args.retain_recent)
        agent_factory = lambda: RagAgent(retain_recent=args.retain_recent)
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
    else:
        raise NotImplementedError(f"Agent '{args.agent}' not implemented")

    # Run experiment
    runner = MultiHopExperimentRunner(
        dataset_path=args.dataset,
        compression_turn=args.compression_turn,
        retain_recent=args.retain_recent,
    )

    result = runner.run(
        agent,
        num_samples=args.samples,
        max_workers=args.workers,
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
