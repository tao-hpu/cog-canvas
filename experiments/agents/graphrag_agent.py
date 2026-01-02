"""
GraphRAG Agent: Microsoft's original GraphRAG implementation for session-level memory.

This agent wraps Microsoft's graphrag library to provide:
1. Entity extraction and graph building from conversation turns
2. Community detection and summarization
3. Local/Global search for question answering

Key differences from GraphRAG-lite:
- Uses the full graphrag pipeline (entity extraction, community detection, summarization)
- Requires Python 3.10+ with graphrag package installed
- Much more computationally expensive (many LLM calls for indexing)
- Better quality but slower

Usage:
    This agent requires pre-computed graphrag indexes due to indexing cost.
    Run `python -m experiments.graphrag_indexer` to pre-index conversations.

Reference: Microsoft GraphRAG (2024)
- Paper: https://arxiv.org/abs/2404.16130
- Implementation: https://github.com/microsoft/graphrag
"""

from typing import List, Dict, Any, Optional
import time
import os
import tempfile
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG agent."""
    model: str = "gpt-4o-mini"
    embedding_model: str = "BAAI/bge-m3"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_api_base: Optional[str] = None
    search_method: str = "local"  # "local" or "global"
    community_level: int = 2
    response_type: str = "Single Paragraph"


class GraphRAGAgent(Agent):
    """
    GraphRAG baseline: Microsoft's full GraphRAG implementation.

    Workflow:
    1. On process_turn: Accumulate conversation turns
    2. On compression: Build graphrag index from old turns (expensive!)
    3. On answer_question: Query using graphrag local/global search

    Note: Due to the high cost of graphrag indexing, this agent supports
    pre-computed indexes for batch experiments.
    """

    SETTINGS_TEMPLATE = """### GraphRAG Configuration for CogCanvas experiments

models:
  default_chat_model:
    type: chat
    model_provider: openai
    auth_type: api_key
    api_key: {api_key}
    api_base: {api_base}
    model: {model}
    model_supports_json: true
    concurrent_requests: 5
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 5
  default_embedding_model:
    type: embedding
    model_provider: openai
    auth_type: api_key
    api_key: {embedding_api_key}
    api_base: {embedding_api_base}
    model: {embedding_model}
    concurrent_requests: 5
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 5

input:
  storage:
    type: file
    base_dir: "input"
  file_type: text

chunks:
  size: 800
  overlap: 100
  group_by_columns: [id]

output:
  type: file
  base_dir: "output"

cache:
  type: file
  base_dir: "cache"

reporting:
  type: file
  base_dir: "logs"

vector_store:
  default_vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default

embed_text:
  model_id: default_embedding_model
  vector_store_id: default_vector_store

extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [person, technology, decision, fact, organization, concept]
  max_gleanings: 1

summarize_descriptions:
  model_id: default_chat_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english
  async_mode: threaded

cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false

community_reports:
  model_id: default_chat_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  embeddings: false

local_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  chat_model_id: default_chat_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

basic_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
"""

    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        python_path: Optional[str] = None,
        retain_recent: int = 5,
        precomputed_index_path: Optional[str] = None,
    ):
        """
        Initialize GraphRAG agent.

        Args:
            config: GraphRAG configuration
            python_path: Path to Python with graphrag installed
            retain_recent: Number of recent turns to keep in context
            precomputed_index_path: Path to pre-computed graphrag index
        """
        from dotenv import load_dotenv
        load_dotenv()

        self.config = config or GraphRAGConfig()
        self.retain_recent = retain_recent
        self.precomputed_index_path = precomputed_index_path

        # Fill in API keys from env if not provided
        if not self.config.api_key:
            self.config.api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.config.api_base:
            self.config.api_base = os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")
        if not self.config.embedding_api_key:
            self.config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or self.config.api_key
        if not self.config.embedding_api_base:
            self.config.embedding_api_base = os.getenv("EMBEDDING_API_BASE") or self.config.api_base

        # Find Python with graphrag
        self.python_path = python_path
        if not self.python_path:
            # Try common locations
            candidates = [
                os.path.expanduser("~/opt/anaconda3/envs/graphrag/bin/python"),
                os.path.expanduser("~/anaconda3/envs/graphrag/bin/python"),
                "/opt/anaconda3/envs/graphrag/bin/python",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    self.python_path = candidate
                    break

        if not self.python_path:
            raise RuntimeError("Could not find Python with graphrag installed. "
                             "Create conda env: conda create -n graphrag python=3.11 && pip install graphrag")

        # Working directory for graphrag
        self._work_dir: Optional[str] = None
        self._index_built = False

        # History
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []

    @property
    def name(self) -> str:
        return f"GraphRAG({self.config.search_method})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self._cleanup_work_dir()
        self._history = []
        self._retained_history = []
        self._index_built = False

    def _cleanup_work_dir(self):
        """Clean up temporary working directory."""
        if self._work_dir and os.path.exists(self._work_dir):
            try:
                shutil.rmtree(self._work_dir)
            except Exception:
                pass
        self._work_dir = None

    def _setup_work_dir(self) -> str:
        """Set up working directory with graphrag configuration."""
        if self._work_dir and os.path.exists(self._work_dir):
            return self._work_dir

        # Create temp directory
        self._work_dir = tempfile.mkdtemp(prefix="graphrag_")
        work_path = Path(self._work_dir)

        # Create input directory
        (work_path / "input").mkdir(exist_ok=True)

        # Initialize graphrag to get prompts
        result = subprocess.run(
            [self.python_path, "-m", "graphrag", "init"],
            cwd=self._work_dir,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Write custom settings.yaml
        settings_content = self.SETTINGS_TEMPLATE.format(
            api_key=self.config.api_key,
            api_base=self.config.api_base,
            model=self.config.model,
            embedding_api_key=self.config.embedding_api_key,
            embedding_api_base=self.config.embedding_api_base,
            embedding_model=self.config.embedding_model,
        )
        (work_path / "settings.yaml").write_text(settings_content)

        return self._work_dir

    def process_turn(self, turn: ConversationTurn) -> None:
        """Store turn in history (indexing happens at compression time)."""
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle compression: Build graphrag index from old turns.

        WARNING: This is expensive! Each call makes many LLM API calls.
        """
        # Identify turns to process into graph
        turns_to_process = [
            t for t in self._history
            if t not in retained_turns
        ]

        if turns_to_process and not self._index_built:
            self._build_index(turns_to_process)
            self._index_built = True

        # Update retained history
        self._retained_history = list(retained_turns)
        self._history = list(retained_turns)

    def _build_index(self, turns: List[ConversationTurn]) -> None:
        """Build graphrag index from conversation turns."""
        work_dir = self._setup_work_dir()
        work_path = Path(work_dir)

        # Write conversation turns as input documents
        input_path = work_path / "input" / "conversation.txt"
        lines = []
        for turn in turns:
            lines.append(f"Turn {turn.turn_id}:")
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")
            lines.append("")

        input_path.write_text("\n".join(lines))

        # Run graphrag indexing
        try:
            result = subprocess.run(
                [self.python_path, "-m", "graphrag", "index"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )
            if result.returncode != 0:
                print(f"GraphRAG indexing failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("GraphRAG indexing timed out")
        except Exception as e:
            print(f"GraphRAG indexing error: {e}")

    def answer_question(self, question: str) -> AgentResponse:
        """Answer question using graphrag search."""
        start_time = time.time()

        # If we have a precomputed index, use it
        work_dir = self.precomputed_index_path or self._work_dir

        if not work_dir or not os.path.exists(work_dir):
            # No index available, fall back to recent history only
            return self._answer_from_history_only(question, start_time)

        # Query graphrag
        answer = self._query_graphrag(work_dir, question)

        # If graphrag failed, fall back to history
        if not answer:
            return self._answer_from_history_only(question, start_time)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "method": self.config.search_method,
                "index_path": work_dir,
            }
        )

    def _query_graphrag(self, work_dir: str, question: str, max_retries: int = 3) -> Optional[str]:
        """Query graphrag using CLI with retry mechanism."""
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    [
                        self.python_path, "-m", "graphrag", "query",
                        "--method", self.config.search_method,
                        "--query", question,
                    ],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    print(f"GraphRAG query failed: {result.stderr}")
                    if attempt < max_retries - 1:
                        print(f"  Retrying ({attempt + 1}/{max_retries})...")
                        continue
                    return None

            except subprocess.TimeoutExpired:
                if attempt < max_retries - 1:
                    print(f"GraphRAG query timed out, retrying ({attempt + 1}/{max_retries})...")
                    continue
                print("GraphRAG query timed out after 3 attempts")
                return None
            except Exception as e:
                print(f"GraphRAG query error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying ({attempt + 1}/{max_retries})...")
                    continue
                return None

        return None

    def _answer_from_history_only(self, question: str, start_time: float) -> AgentResponse:
        """Fallback: Answer from recent history only."""
        context_parts = []

        if self._retained_history:
            context_parts.append("## Recent Conversation")
            for turn in self._retained_history:
                context_parts.append(f"User: {turn.user}")
                context_parts.append(f"Assistant: {turn.assistant}")
                context_parts.append("")

        context = "\n".join(context_parts) if context_parts else "[No context available]"

        # Use simple LLM call for answer
        answer = self._generate_answer(context, question)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={"method": "fallback"}
        )

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer via LLM with infinite retry."""
        from experiments.llm_utils import call_llm_with_retry

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )

            prompt = f"""You are an expert reasoning agent. Your goal is to answer the user's question by connecting discrete facts from the retrieved information.

## Retrieved Context
{context}

## Instructions
1. Analyze the retrieved information carefully
2. Even if pieces of information are not explicitly linked, use your reasoning to infer relationships
3. Synthesize a complete answer that explains the reasoning process

## Question
{question}

## Answer
"""

            return call_llm_with_retry(
                client=client,
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )

        except Exception as e:
            return f"Error: {e}"

    def __del__(self):
        """Clean up on destruction."""
        self._cleanup_work_dir()


# Convenience function for experiments
def create_graphrag_agent(
    search_method: str = "local",
    precomputed_index_path: Optional[str] = None,
) -> GraphRAGAgent:
    """Create a GraphRAG agent with default configuration."""
    config = GraphRAGConfig(search_method=search_method)
    return GraphRAGAgent(
        config=config,
        precomputed_index_path=precomputed_index_path,
    )
