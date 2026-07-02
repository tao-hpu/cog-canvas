"""
SeCom Anchor Agent: Topical Segmentation + Extractive Compression.

Reproduces the *storage representation* of SeCom (Pan et al., ICLR 2025,
arXiv:2502.05589) inside our fixed retrieve -> rerank -> reason pipeline.
SeCom's defining contribution over naive RAG chunking is twofold:

1. **Conversation segmentation** -- instead of fixed-size character chunks,
   the dialogue is cut into topically coherent *segments* by an LLM (topic-shift
   detection). Retrieval granularity therefore tracks topic, not character count.
2. **Extractive denoising compression** -- each segment is compressed with
   LLMLingua-2 (token-level extractive dropping at a retention rate), removing
   low-information tokens while keeping near-verbatim wording.

This places SeCom one step *lossier* than verbatim chunks but well above
abstractive summarization on the fidelity spectrum, which is exactly the point
of the multi-schema fidelity curve.

FAIRNESS (controlled ablation, see paper Section 4.1):
Unlike SeCom's native stack (MPNet + FAISS + GPT-4 segmenter), we hold the
*pipeline* identical to every other anchor -- same embedder (EMBEDDING_MODEL,
bge-m3), same reranker, same answerer (ANSWER_MODEL, gpt-4o). The ONLY variable
swapped is the storage representation. The segmentation model is the shared
representation-builder model (EXTRACTOR_MODEL, gpt-4o-mini), matching the
artifact / Mem0 / A-Mem builders so that *structure* -- not builder strength --
is the isolated variable. Reproduced numbers will therefore differ from SeCom's
headline scores; that gap is the experiment, not a bug.

Set ``compress=False`` for the no-compression ablation (segmentation only).
"""

from typing import List, Optional, Dict, Any
import json
import time
import os

import threading

import numpy as np

from experiments.runner import Agent, AgentResponse
from experiments.data_gen import ConversationTurn
from experiments.llm_utils import call_llm_with_retry
from cogcanvas.embeddings import (
    APIEmbeddingBackend,
    MockEmbeddingBackend,
    batch_cosine_similarity,
)
from cogcanvas.reranker import Reranker
from experiments.agents._budget import (
    context_budget_chars, effective_k, fill_to_budget,
)


# Shared across all SecomAgent instances/workers so the LLMLingua-2 model is
# loaded once, not once per worker.
_SHARED_COMPRESSOR = None
_COMPRESSOR_LOCK = threading.Lock()


class Segment:
    """A topically coherent segment of conversation, optionally compressed."""

    __slots__ = ("content", "embedding", "source_turns", "raw_len", "kept_len")

    def __init__(self, content, embedding, source_turns, raw_len, kept_len):
        self.content = content
        self.embedding = embedding
        self.source_turns = source_turns
        self.raw_len = raw_len      # chars before compression
        self.kept_len = kept_len    # chars after compression


class SecomAgent(Agent):
    """
    SeCom anchor: topical segmentation + LLMLingua-2 extractive compression,
    retrieved/reranked/answered through the shared pipeline.
    """

    def __init__(
        self,
        model: str = None,
        embedding_model: str = None,
        segmenter_model: str = None,
        retain_recent: int = 5,
        top_k: int = 10,
        use_reranker: bool = True,
        compress: bool = True,
        compress_rate: float = 0.75,
        segment_window: int = 12,      # turns per segmentation LLM call
        max_segment_chars: int = 1500,  # hard cap so a runaway topic still splits
    ):
        from dotenv import load_dotenv

        load_dotenv()

        self.retain_recent = retain_recent
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.compress = compress
        self.compress_rate = compress_rate
        self.segment_window = segment_window
        self.max_segment_chars = max_segment_chars

        # Answerer = pipeline reasoner (gpt-4o), identical to RAG/Summarization.
        self.model = model or os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
        # Segmenter = shared representation-BUILDER model. Headline uses the
        # strongest builder (BUILDER_MODEL, defaulting to the gpt-4o answerer) as
        # a steelman; the builder-invariance check overrides BUILDER_MODEL to
        # gpt-4o-mini. Same knob is shared by every lossy anchor so that builder
        # strength is a controlled factor, not a confound.
        self.segmenter_model = (
            segmenter_model
            or os.getenv("BUILDER_MODEL")
            or os.getenv("ANSWER_MODEL")
            or "gpt-4o-mini"
        )

        self._client = None
        self._init_client()

        # Embedding backend -- same vars as RagAgent (pipeline-constant).
        embed_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
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
                print("Warning: EMBEDDING_API_KEY/API_KEY not set, using mock embeddings")
                self.embedder = MockEmbeddingBackend()
        except Exception as e:
            print(f"Failed to init embedding backend: {e}. Using mock.")
            self.embedder = MockEmbeddingBackend()

        # Reranker -- same as RagAgent.
        self.reranker = None
        if self.use_reranker:
            try:
                rk_key = os.getenv("RERANKER_API_KEY") or os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY")
                rk_base = os.getenv("RERANKER_API_BASE") or os.getenv("EMBEDDING_API_BASE") or os.getenv("API_BASE")
                if rk_key:
                    self.reranker = Reranker(
                        model=os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3"),
                        api_key=rk_key,
                        api_base=rk_base,
                        use_mock=False,
                    )
                else:
                    print("Warning: RERANKER_API_KEY not set, using mock reranker")
                    self.reranker = Reranker(use_mock=True)
            except Exception as e:
                print(f"Failed to init reranker: {e}. Using mock.")
                self.reranker = Reranker(use_mock=True)

        # Lazy LLMLingua-2 compressor.
        self._compressor = None

        # State
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []
        self._store: List[Segment] = []

    def _init_client(self):
        try:
            from openai import OpenAI

            api_key = os.getenv("ANSWER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("ANSWER_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")
            if api_key:
                self._client = OpenAI(api_key=api_key, base_url=api_base)
        except ImportError:
            pass

    @property
    def name(self) -> str:
        comp = f"+llmlingua2@{self.compress_rate}" if self.compress else "+nocompress"
        rr = "+rerank" if self.use_reranker else ""
        return f"SeCom(seg{comp}{rr}, k={self.top_k})"

    # ---- P1-8 backbone-integrated anchor: builder interface --------------
    # Used when this SeCom instance is passed as CogCanvasAgent(items_builder=...)
    # so SeCom's defining mechanism (segmentation + LLMLingua-2 compression)
    # produces the stored items while retrieval/rerank/answer/date-grounding
    # come from the shared backbone (the pipeline that scores 43.9 / 28.0).

    @property
    def anchor_label(self) -> str:
        return f"SeComSeg{'+llmlingua2@' + str(self.compress_rate) if self.compress else '+nocompress'}"

    def _format_turns_dated(self, turns: List[ConversationTurn]) -> str:
        """Like _format_turns but with the (Session: ...) marker, matching the
        verbatim-chunks anchor's input exactly so segmentation/compression is
        the ONLY swapped variable (and a dropped date is SeCom's own fidelity
        loss, not a harness asymmetry)."""
        out = []
        for t in turns:
            sdt = getattr(t, "session_datetime", None)
            dt = f" (Session: {sdt})" if sdt else ""
            out.append(f"[Turn {t.turn_id}{dt}] User: {t.user}\nAssistant: {t.assistant}\n\n")
        return "".join(out)

    def build_items(self, turns: List[ConversationTurn], verbose: int = 0):
        """Return stored items for CogCanvasAgent._items_ingest: SeCom segments
        (LLM topical segmentation + optional LLMLingua-2 compression), each as
        {content, source_turns, session_datetime}."""
        segments = self._segment_turns(turns)
        items = []
        for seg in segments:
            if not seg:
                continue
            text = self._format_turns_dated(seg)
            if self.compress:
                text = self._compress(text)
            sdt = next(
                (getattr(t, "session_datetime", None) for t in seg
                 if getattr(t, "session_datetime", None)),
                None,
            )
            items.append({
                "content": text,
                "source_turns": [t.turn_id for t in seg],
                "session_datetime": sdt,
            })
        return items

    def reset(self) -> None:
        self._history = []
        self._retained_history = []
        self._store = []

    def process_turn(self, turn: ConversationTurn) -> None:
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        turns_to_process = [t for t in self._history if t not in retained_turns]
        if turns_to_process:
            self._segment_and_store(turns_to_process)
        self._retained_history = retained_turns
        self._history = list(retained_turns)

    # ---- SeCom mechanism 1: topical segmentation -------------------------

    def _segment_and_store(self, turns: List[ConversationTurn]) -> None:
        segments = self._segment_turns(turns)
        texts, sources = [], []
        for seg_turns in segments:
            text = self._format_turns(seg_turns)
            if self.compress:
                text = self._compress(text)
            texts.append(text)
            sources.append([t.turn_id for t in seg_turns])

        if not texts:
            return
        embeddings = self.embedder.embed_batch(texts)
        for content, emb, src, seg_turns in zip(texts, embeddings, sources, segments):
            raw = len(self._format_turns(seg_turns))
            self._store.append(
                Segment(content=content, embedding=emb, source_turns=src,
                        raw_len=raw, kept_len=len(content))
            )

    def _segment_turns(self, turns: List[ConversationTurn]) -> List[List[ConversationTurn]]:
        """LLM topic-shift segmentation over sliding windows of turns.

        Returns a list of segments, each a list of consecutive ConversationTurns.
        Falls back to one-segment-per-window if the LLM call/parse fails.
        """
        segments: List[List[ConversationTurn]] = []
        W = self.segment_window
        for w_start in range(0, len(turns), W):
            window = turns[w_start:w_start + W]
            boundaries = self._llm_boundaries(window)
            # boundaries = set of local indices (1..len-1) where a NEW topic starts
            cur: List[ConversationTurn] = []
            for i, t in enumerate(window):
                if i in boundaries and cur:
                    segments.append(cur)
                    cur = []
                cur.append(t)
                # hard cap on segment size
                if sum(len(self._format_turns([x])) for x in cur) >= self.max_segment_chars:
                    segments.append(cur)
                    cur = []
            if cur:
                segments.append(cur)
        return segments

    def _llm_boundaries(self, window: List[ConversationTurn]) -> set:
        if self._client is None or len(window) <= 1:
            return set()
        listing = "\n".join(
            f"[{i}] User: {t.user}\n    Assistant: {t.assistant}"
            for i, t in enumerate(window)
        )
        prompt = (
            "Below are consecutive turns of a conversation, indexed [0], [1], ...\n"
            "Identify the indices where a NEW topic begins (a topic shift from the "
            "previous turn). Group turns about the same topic together.\n"
            "Return ONLY a JSON object: {\"boundaries\": [<indices where a new topic starts>]}\n"
            "Index 0 is never a boundary. If the whole window is one topic, return "
            "{\"boundaries\": []}.\n\n"
            f"{listing}\n\nJSON:"
        )
        try:
            raw = call_llm_with_retry(
                client=self._client,
                model=self.segmenter_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0,
                call_type="extract",
            )
            raw = raw.strip()
            if "{" in raw:
                raw = raw[raw.index("{"): raw.rindex("}") + 1]
            data = json.loads(raw)
            return {int(i) for i in data.get("boundaries", []) if 0 < int(i) < len(window)}
        except Exception:
            return set()

    # ---- SeCom mechanism 2: extractive compression (LLMLingua-2) ---------

    def _get_compressor(self):
        # Module-level singleton: the ~560MB LLMLingua-2 model is shared across
        # all agent instances/workers instead of one copy per worker.
        global _SHARED_COMPRESSOR
        if _SHARED_COMPRESSOR is None:
            try:
                from llmlingua import PromptCompressor

                with _COMPRESSOR_LOCK:
                    if _SHARED_COMPRESSOR is None:
                        _SHARED_COMPRESSOR = PromptCompressor(
                            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                            use_llmlingua2=True,
                            device_map=os.getenv("LLMLINGUA_DEVICE", "cpu"),
                        )
            except Exception as e:
                raise RuntimeError(
                    "LLMLingua-2 is required for SeCom compression (compress=True). "
                    "Install with `pip install llmlingua`. Original error: " + str(e)
                )
        return _SHARED_COMPRESSOR

    def _compress(self, text: str) -> str:
        if not text.strip():
            return text
        compressor = self._get_compressor()
        try:
            out = compressor.compress_prompt(
                text, rate=self.compress_rate, force_tokens=["\n", ":", "?", "."]
            )
            return out.get("compressed_prompt", text)
        except Exception:
            return text

    # ---- retrieval / answer (pipeline-constant, same as RagAgent) --------

    def _format_turns(self, turns: List[ConversationTurn]) -> str:
        return "".join(
            f"[Turn {t.turn_id}] User: {t.user}\nAssistant: {t.assistant}\n\n"
            for t in turns
        )

    def answer_question(self, question: str) -> AgentResponse:
        start_time = time.time()
        retrieved, scores = [], []
        budget = context_budget_chars()
        eff_k = effective_k(self.top_k, budget)
        if self._store:
            q_emb = self.embedder.embed(question)
            seg_embs = [s.embedding for s in self._store]
            sims = batch_cosine_similarity(q_emb, seg_embs)
            ranked = sorted(zip(self._store, sims), key=lambda x: x[1], reverse=True)
            if self.use_reranker and self.reranker is not None:
                cand_k = min(eff_k * 2, len(ranked))
                cand = [s for s, _ in ranked[:cand_k]]
                rr = self.reranker.rerank(question, [s.content for s in cand], top_k=eff_k)
                retrieved = [cand[idx] for idx, _ in rr]
                scores = [sc for _, sc in rr]
            else:
                top = ranked[:eff_k]
                retrieved = [s for s, _ in top]
                scores = [sc for _, sc in top]
            keep = len(fill_to_budget(retrieved, lambda s: s.content, budget))
            retrieved, scores = retrieved[:keep], scores[:keep]

        parts = []
        if retrieved:
            parts.append("## Retrieved Context (from earlier conversation)")
            for i, seg in enumerate(retrieved):
                parts.append(f"--- Segment {i+1} (Relevance: {scores[i]:.2f}) ---")
                parts.append(seg.content)
            parts.append("")
        if self._history:
            parts.append("## Recent Conversation")
            for t in self._history:
                parts.append(f"User: {t.user}")
                parts.append(f"Assistant: {t.assistant}")
                parts.append("")
        context = "\n".join(parts) if parts else "[No context available]"

        answer = self._generate_answer(context, question)
        latency = (time.time() - start_time) * 1000
        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "retrieved_segments": len(retrieved),
                "store_size": len(self._store),
                "top_score": scores[0] if scores else 0.0,
            },
        )

    def _generate_answer(self, context: str, question: str) -> str:
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
        if self._client is None:
            return "I don't have enough information."
        try:
            return call_llm_with_retry(
                client=self._client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
                call_type="gen",
            )
        except Exception as e:
            return f"Error: {e}"
