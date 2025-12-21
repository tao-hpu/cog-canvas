"""
MemGPT-lite Agent: Simplified hierarchical memory inspired by MemGPT/Letta.

This agent implements a simplified version of MemGPT's memory architecture:
1. Main Context (Core Memory): Recent turns kept in full
2. Archival Memory: Older turns stored with embeddings for semantic retrieval
3. Recall Memory: Keyword-based search capability (simplified)

Key differences from RAG baseline:
- Hierarchical memory structure (recent vs archival)
- Combines semantic search with keyword search
- More structured context injection

Expected performance:
- Should beat simple summarization (preserves details)
- Similar to RAG but with hybrid retrieval
- Likely below CogCanvas (no structured extraction or graph)
"""

from typing import List, Dict, Any, Optional
import time
import os
import numpy as np
from dataclasses import dataclass

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from experiments.llm_utils import call_llm_with_retry
from cogcanvas.embeddings import (
    APIEmbeddingBackend,
    MockEmbeddingBackend,
    batch_cosine_similarity,
)


@dataclass
class ArchivalEntry:
    """An entry in archival memory with embedding."""

    turn_id: int
    user: str
    assistant: str
    embedding: List[float]
    keywords: List[str]  # Simple keyword extraction for recall memory


class MemGPTLiteAgent(Agent):
    """
    MemGPT-lite agent - hierarchical memory with hybrid retrieval.

    Memory architecture:
    - Core Memory: Recent N turns (full context)
    - Archival Memory: Older turns with embeddings
    - Recall Memory: Keyword-based index (simplified)

    On compression:
    - Recent turns stay in core memory
    - Older turns move to archival memory with embeddings

    On answer:
    - Semantic search in archival memory
    - Keyword search in archival memory
    - Combine with core memory for context
    """

    def __init__(
        self,
        model: str = None,
        embedding_model: str = None,
        core_memory_size: int = 5,  # Recent turns to keep in full
        archival_top_k: int = 10,  # How many archival entries to retrieve
        use_keyword_search: bool = True,  # Enable recall memory
    ):
        """
        Initialize MemGPT-lite agent.

        Args:
            model: LLM for answering
            embedding_model: Embedding model name
            core_memory_size: Number of recent turns to keep in core memory
            archival_top_k: Number of archival entries to retrieve
            use_keyword_search: Whether to use keyword-based recall memory
        """
        from dotenv import load_dotenv

        load_dotenv()

        self.core_memory_size = core_memory_size
        self.archival_top_k = archival_top_k
        self.use_keyword_search = use_keyword_search

        # Models
        self.model = model or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
        embed_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "bge-large-zh-v1.5"
        )

        # Initialize LLM client
        self._client = None
        self._init_client()

        # Initialize Embedding backend
        try:
            embed_api_key = (
                os.getenv("EMBEDDING_API_KEY")
                or os.getenv("API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            embed_api_base = (
                os.getenv("EMBEDDING_API_BASE")
                or os.getenv("API_BASE")
                or os.getenv("OPENAI_API_BASE")
            )
            if embed_api_key:
                self.embedder = APIEmbeddingBackend(
                    model=embed_model_name,
                    api_key=embed_api_key,
                    api_base=embed_api_base,
                )
            else:
                print("Warning: EMBEDDING_API_KEY not set, using mock embeddings")
                self.embedder = MockEmbeddingBackend()
        except Exception as e:
            print(f"Failed to init embedding backend: {e}. Using mock.")
            self.embedder = MockEmbeddingBackend()

        # State
        self._history: List[ConversationTurn] = []
        self._core_memory: List[ConversationTurn] = []  # Recent turns (full)
        self._archival_memory: List[ArchivalEntry] = []  # Older turns (embedded)
        self._keyword_index: Dict[str, List[int]] = (
            {}
        )  # keyword -> list of archival indices

    def _init_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI

            api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")

            if api_key:
                self._client = OpenAI(api_key=api_key, base_url=api_base)
        except ImportError:
            pass

    @property
    def name(self) -> str:
        kw_status = "+KW" if self.use_keyword_search else ""
        return f"MemGPT-lite(core={self.core_memory_size}, k={self.archival_top_k}{kw_status})"

    def reset(self) -> None:
        """Reset state between conversations."""
        self._history = []
        self._core_memory = []
        self._archival_memory = []
        self._keyword_index = {}

    def process_turn(self, turn: ConversationTurn) -> None:
        """Store turn in history."""
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle compression: Move old turns to archival memory.
        """
        # Identify turns to archive
        turns_to_archive = [t for t in self._history if t not in retained_turns]

        if turns_to_archive:
            self._archive_turns(turns_to_archive)

        # Update core memory
        self._core_memory = list(retained_turns)
        # Update full history
        self._history = list(retained_turns)

    def _archive_turns(self, turns: List[ConversationTurn]) -> None:
        """Archive turns with embeddings and keywords."""
        if not turns:
            return

        # Prepare texts for embedding
        texts = []
        for turn in turns:
            # Combine user and assistant for embedding
            text = f"User: {turn.user}\nAssistant: {turn.assistant}"
            texts.append(text)

        # Batch embed
        embeddings = self.embedder.embed_batch(texts)

        # Create archival entries
        for turn, text, embedding in zip(turns, texts, embeddings):
            # Extract keywords (simple approach)
            keywords = self._extract_keywords(text)

            entry = ArchivalEntry(
                turn_id=turn.turn_id,
                user=turn.user,
                assistant=turn.assistant,
                embedding=embedding,
                keywords=keywords,
            )

            # Add to archival memory
            archival_idx = len(self._archival_memory)
            self._archival_memory.append(entry)

            # Update keyword index
            for kw in keywords:
                if kw not in self._keyword_index:
                    self._keyword_index[kw] = []
                self._keyword_index[kw].append(archival_idx)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text (simplified version).
        Enhanced to include more patterns for common entities.
        """
        import re

        text_lower = text.lower()
        keywords = set()

        # Existing patterns
        # 1. Numbers with units (dates, amounts, sizes)
        number_patterns = re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:gb|tb|mb|kb|%|requests?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|dollars?|\$|people|engineers?|developers?)\b",
            text_lower,
        )
        keywords.update(number_patterns)

        # 2. Specific dates (month name + day)
        date_patterns = re.findall(
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b",
            text_lower,
        )
        keywords.update(date_patterns)

        # 3. Technology names (common patterns)
        tech_words = [
            "aws",
            "azure",
            "gcp",
            "google cloud",
            "digitalocean",
            "postgresql",
            "mysql",
            "mongodb",
            "sqlite",
            "redis",
            "memcached",
            "fastapi",
            "flask",
            "django",
            "express",
            "node.js",
            "react",
            "angular",
            "oauth",
            "jwt",
            "api key",
            "api",
            "sdk",
            "cli",
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "jenkins",
            "gitlab ci",
            "github actions",
        ]
        for tech in tech_words:
            if tech in text_lower:
                keywords.add(tech)

        # 4. Dollar amounts
        dollar_patterns = re.findall(r"\$[\d,]+(?:\.\d{2})?", text)
        keywords.update([p.lower() for p in dollar_patterns])

        # 5. Quoted strings (often important)
        quoted = re.findall(r'"([^"]+)"', text)
        keywords.update(
            [q.lower() for q in quoted if len(q) < 50]
        )  # Limit length to avoid noise

        # New patterns
        # 6. Email addresses
        email_patterns = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text_lower
        )
        keywords.update(email_patterns)

        # 7. URLs (basic detection)
        url_patterns = re.findall(r"https?://\S+|www\.\S+", text_lower)
        keywords.update(url_patterns)

        # 8. Phone numbers (simple format)
        phone_patterns = re.findall(
            r"\b(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b", text_lower
        )
        keywords.update(phone_patterns)

        # 9. Proper nouns / capitalized words (more robust than just start of sentence)
        # Look for sequences of capitalized words, or capitalized words not at sentence start
        # This is a heuristic and might pick up some false positives.
        capitalized_words = re.findall(
            r"\b[A-Z][a-z0-9]+\b|\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+\b", text
        )
        for word_or_phrase in capitalized_words:
            # Avoid adding single common words that are capitalized by sentence start
            if word_or_phrase.lower() not in [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "for",
                "nor",
                "so",
                "yet",
                "it",
                "is",
                "in",
                "on",
                "at",
            ]:
                keywords.add(word_or_phrase.lower())

        return list(keywords)

    def answer_question(self, question: str) -> AgentResponse:
        """Answer question using hierarchical memory."""
        start_time = time.time()

        # 1. Retrieve from archival memory (semantic search)
        semantic_results = []  # List of (index, similarity_score)
        keyword_results_rrf = []  # Initialize here to avoid NameError

        if self._archival_memory:
            # Semantic search
            query_embedding = self.embedder.embed(question)
            archival_embeddings = [e.embedding for e in self._archival_memory]
            similarities = batch_cosine_similarity(query_embedding, archival_embeddings)

            # BUG FIX: Actually populate semantic_results from similarities!
            # Create (index, score) pairs and sort by score descending
            semantic_results = [
                (i, float(sim)) for i, sim in enumerate(similarities)
            ]
            semantic_results.sort(key=lambda x: x[1], reverse=True)
            semantic_results = semantic_results[: self.archival_top_k]  # Top-k

            if self.use_keyword_search:
                question_keywords = self._extract_keywords(question)
                keyword_matches = set()
                for kw in question_keywords:
                    if kw in self._keyword_index:
                        keyword_matches.update(self._keyword_index[kw])

                # Convert set to list for ordered indexing and limit to top-k
                keyword_matches_list = list(keyword_matches)

                # For RRF, we'll assign a dummy high score and rank based on order
                for i, idx in enumerate(keyword_matches_list[: self.archival_top_k]):
                    keyword_results_rrf.append(
                        (idx, 1.0, i + 1)
                    )  # (idx, dummy_score, rank)

        # 2. Combine results using Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[int, float] = {}
        RRF_K = 60  # Typically between 50-100

        # Process semantic results
        for i, (idx, score) in enumerate(semantic_results):
            rank = i + 1
            fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank))

        # Process keyword results
        for i, (idx, dummy_score, rank) in enumerate(keyword_results_rrf):
            # If an item also appeared in semantic search, its rank might be different.
            # We use its rank from the keyword search here.
            fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank))

        # Sort by fused RRF score
        reranked_archival_indices = sorted(
            fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True
        )[
            : self.archival_top_k
        ]  # Limit to top-k after RRF

        retrieved_entries = [
            self._archival_memory[idx] for idx in reranked_archival_indices
        ]

        # Sort by turn_id for chronological presentation (within the RRF-selected top-k)
        retrieved_entries.sort(key=lambda e: e.turn_id)

        # 3. Build context
        context_parts = []

        if retrieved_entries:
            context_parts.append(
                "## Archival Memory (retrieved from earlier conversation)"
            )
            for entry in retrieved_entries:
                context_parts.append(f"[Turn {entry.turn_id}]")
                context_parts.append(f"User: {entry.user}")
                context_parts.append(f"Assistant: {entry.assistant}")
                context_parts.append("")

        # Use _history (includes post-compression turns) instead of _core_memory
        if self._history:
            context_parts.append("## Core Memory (recent conversation)")
            for turn in self._history:
                context_parts.append(f"User: {turn.user}")
                context_parts.append(f"Assistant: {turn.assistant}")
                context_parts.append("")

        context = (
            "\n".join(context_parts) if context_parts else "[No context available]"
        )

        # 4. Generate answer
        answer = self._generate_answer(context, question)

        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "semantic_retrieved": len(semantic_results),
                "keyword_retrieved": len(keyword_results_rrf),
                "total_retrieved": len(retrieved_entries),
                "archival_size": len(self._archival_memory),
                "core_memory_size": len(self._core_memory),
            },
        )

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer via LLM."""
        prompt = f"""Answer the question based on the provided memory context.
The context includes:
- Archival Memory: Important information retrieved from earlier in the conversation
- Core Memory: Recent conversation turns

If the information is not available, say "I don't have enough information."

{context}

## Question
{question}

## Answer
Provide a concise, direct answer."""

        if self._client is None:
            return "I don't have enough information."

        try:
            return call_llm_with_retry(
                client=self._client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
        except Exception as e:
            return f"Error: {e}"
