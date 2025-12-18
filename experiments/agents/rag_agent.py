"""
RAG Baseline Agent: Naive Chunking + Vector Retrieval.

This agent represents the standard Retrieval-Augmented Generation approach:
1. Divide long conversation history into text chunks
2. Store chunks in a vector database (simple list in this case)
3. Retrieve relevant chunks based on semantic similarity to the question

Expected performance:
- Better than Summarization for specific details (if chunked correctly)
- Worse than CogCanvas for complex reasoning or scattered context (no graph structure)
"""

from typing import List, Tuple, Dict, Any
import time
import os
import numpy as np
from dataclasses import dataclass

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from cogcanvas.embeddings import APIEmbeddingBackend, MockEmbeddingBackend, batch_cosine_similarity

@dataclass
class TextChunk:
    """A chunk of text with its embedding."""
    content: str
    embedding: List[float]
    source_turns: List[int]  # Which turns this chunk covers

class RagAgent(Agent):
    """
    RAG baseline agent - uses vector search over text chunks.

    On compression:
    - Old history is converted to text
    - Text is split into overlapping chunks
    - Chunks are embedded and stored

    On answer:
    - Query is embedded
    - Top-k similar chunks are retrieved
    - Retrieved chunks + recent history are used as context
    """

    def __init__(
        self,
        model: str = None,
        embedding_model: str = None,
        retain_recent: int = 5,
        chunk_size: int = 1000,  # Characters
        chunk_overlap: int = 200,
        top_k: int = 3,
    ):
        """
        Initialize RagAgent.

        Args:
            model: LLM for answering (default: MODEL_WEAK_2)
            embedding_model: Embedding model name (default: EMBEDDING_MODEL)
            retain_recent: Number of recent turns to keep in context
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            top_k: Number of chunks to retrieve
        """
        from dotenv import load_dotenv
        load_dotenv()

        self.retain_recent = retain_recent
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Models
        self.model = model or os.getenv("MODEL_WEAK_2", "gpt-4o-mini")
        embed_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-large-zh-v1.5")
        
        # Initialize LLM client
        self._client = None
        self._init_client()

        # Initialize Embedding backend
        # Prefer dedicated EMBEDDING_API_* vars, fallback to general API_*
        try:
            embed_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            embed_api_base = os.getenv("EMBEDDING_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")
            if embed_api_key:
                self.embedder = APIEmbeddingBackend(
                    model=embed_model_name,
                    api_key=embed_api_key,
                    api_base=embed_api_base
                )
            else:
                print("Warning: EMBEDDING_API_KEY/API_KEY not set, using mock embeddings")
                self.embedder = MockEmbeddingBackend()
        except Exception as e:
            print(f"Failed to init embedding backend: {e}. Using mock.")
            self.embedder = MockEmbeddingBackend()

        # State
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []
        self._vector_store: List[TextChunk] = []

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
        return f"RAG(k={self.top_k}, chunk={self.chunk_size})"

    def reset(self) -> None:
        """Reset state."""
        self._history = []
        self._retained_history = []
        self._vector_store = []

    def process_turn(self, turn: ConversationTurn) -> None:
        """Store turn in history."""
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """
        Handle compression: Chunk and embed old history.
        """
        # Identify turns to move to long-term memory (vector store)
        turns_to_process = [
            t for t in self._history
            if t not in retained_turns
        ]

        if turns_to_process:
            self._process_into_chunks(turns_to_process)

        # Update retained history
        self._retained_history = retained_turns
        # Clear full history (simulate truncation)
        self._history = list(retained_turns)

    def _process_into_chunks(self, turns: List[ConversationTurn]) -> None:
        """Convert turns to text, chunk it, and embed."""
        if not turns:
            return

        # 1. Convert turns to a single text string with turn markers
        full_text = ""
        turn_mapping = []  # [(start_char, end_char, turn_id), ...]
        
        for turn in turns:
            start_idx = len(full_text)
            turn_text = f"[Turn {turn.turn_id}] User: {turn.user}\nAssistant: {turn.assistant}\n\n"
            full_text += turn_text
            turn_mapping.append((start_idx, len(full_text), turn.turn_id))

        # 2. Split into chunks
        chunks_text = []
        chunk_sources = []

        start = 0
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))
            
            # If not at end, try to break at newline to avoid cutting words/lines
            if end < len(full_text):
                # Look for last newline in the last 20% of the chunk
                search_start = max(start, int(end - self.chunk_size * 0.2))
                last_newline = full_text.rfind('\n', search_start, end)
                if last_newline != -1:
                    end = last_newline + 1
            
            chunk_content = full_text[start:end]
            chunks_text.append(chunk_content)
            
            # Identify source turns for this chunk
            sources = []
            for t_start, t_end, t_id in turn_mapping:
                # Check intersection
                if not (t_end <= start or t_start >= end):
                    sources.append(t_id)
            chunk_sources.append(sources)

            # Move start pointer (overlap)
            # If we reached the end, break out
            if end >= len(full_text):
                break

            start = end - self.chunk_overlap
            # Ensure we always advance (prevent infinite loop)
            if start <= 0 or start >= end:
                start = end

        # 3. Embed chunks
        if chunks_text:
            embeddings = self.embedder.embed_batch(chunks_text)
            
            for content, embedding, sources in zip(chunks_text, embeddings, chunk_sources):
                self._vector_store.append(TextChunk(
                    content=content,
                    embedding=embedding,
                    source_turns=sources
                ))

    def answer_question(self, question: str) -> AgentResponse:
        """Answer question using retrieved chunks + recent history."""
        start_time = time.time()

        # 1. Retrieve relevant chunks
        retrieved_chunks = []
        scores = []
        
        if self._vector_store:
            query_embedding = self.embedder.embed(question)
            chunk_embeddings = [c.embedding for c in self._vector_store]
            
            similarities = batch_cosine_similarity(query_embedding, chunk_embeddings)
            
            # Sort by score
            scored_chunks = sorted(
                zip(self._vector_store, similarities),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_results = scored_chunks[:self.top_k]
            retrieved_chunks = [c for c, s in top_results]
            scores = [s for c, s in top_results]

        # 2. Build context
        context_parts = []
        
        if retrieved_chunks:
            context_parts.append("## Retrieved Context (from earlier conversation)")
            # Re-sort chunks by turn ID (chronological) if possible, 
            # but usually RAG just concats by relevance. Let's stick to relevance or maybe reverse relevance?
            # Actually, chronological makes more sense for reading. 
            # But naive RAG often just dumps them. Let's dump them but clearly separated.
            for i, chunk in enumerate(retrieved_chunks):
                context_parts.append(f"--- Chunk {i+1} (Relevance: {scores[i]:.2f}) ---")
                context_parts.append(chunk.content)
            context_parts.append("")

        if self._retained_history:
            context_parts.append("## Recent Conversation")
            for turn in self._retained_history:
                context_parts.append(f"User: {turn.user}")
                context_parts.append(f"Assistant: {turn.assistant}")
                context_parts.append("")

        context = "\n".join(context_parts) if context_parts else "[No context available]"

        # 3. Generate Answer
        answer = self._generate_answer(context, question)
        
        latency = (time.time() - start_time) * 1000

        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "retrieved_chunks": len(retrieved_chunks),
                "vector_store_size": len(self._vector_store),
                "top_score": scores[0] if scores else 0.0,
            }
        )

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer via LLM."""
        prompt = f"""Answer the question based on the provided conversation context (including retrieved fragments from history).
If the information is not available, say "I don't have enough information."

{context}

## Question
{question}

## Answer
Provide a concise, direct answer."""

        if self._client is None:
            return "I don't have enough information."

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
