"""
Mem0 Anchor Agent: Incremental Facts + ADD/UPDATE/DELETE/NOOP Update Loop.

Reproduces the *storage representation* of Mem0 (Chhikara et al., arXiv:2504.19413)
inside our fixed retrieve -> rerank -> reason pipeline. Mem0's defining mechanism
is an online memory-update loop: as the dialogue flows, salient facts are
extracted from each turn, and for every candidate fact the system retrieves the
most similar existing memories and an LLM decides one of four operations --

    ADD     : the fact is new -> insert it
    UPDATE  : the fact refines/extends an existing memory -> rewrite it
    DELETE  : the fact contradicts an existing memory -> remove the stale one
    NOOP    : the fact is already captured -> do nothing

This yields a compact, de-duplicated, self-correcting set of natural-language
facts -- lossier than verbatim chunks (wording is paraphrased and detail is
dropped) but it carries an explicit consistency mechanism that summarization and
static artifact extraction lack. That places Mem0 in the middle of the fidelity
curve.

FAIRNESS (controlled ablation, see paper Section 4.1): like every other anchor,
the pipeline is held constant -- same embedder (EMBEDDING_MODEL, bge-m3), same
reranker, same answerer (ANSWER_MODEL, gpt-4o). We deliberately diverge from
Mem0's native text-embedding-3-small; only the storage representation is the
swapped variable. The fact-extraction and update-decision model is the shared
representation-BUILDER model (BUILDER_MODEL, defaulting to the gpt-4o answerer as
a steelman; overridden to gpt-4o-mini for the builder-invariance check), so that
*structure* -- not builder strength -- is the isolated variable.
"""

from typing import List, Dict, Any, Optional
import json
import time
import os

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


class Fact:
    __slots__ = ("id", "text", "embedding")

    def __init__(self, id, text, embedding):
        self.id = id
        self.text = text
        self.embedding = embedding


class Mem0Agent(Agent):
    """
    Mem0 anchor: incremental fact memory with an ADD/UPDATE/DELETE/NOOP loop,
    retrieved/reranked/answered through the shared pipeline.
    """

    def __init__(
        self,
        model: str = None,
        embedding_model: str = None,
        builder_model: str = None,
        retain_recent: int = 5,
        top_k: int = 10,
        use_reranker: bool = True,
        update_sim_threshold: float = 0.6,  # above this -> invoke UPDATE/DELETE/NOOP reasoning
        update_neighbors: int = 4,           # existing facts shown to the update decider
        extract_window: int = 1,             # turns per extraction call (1 = per-turn, faithful)
    ):
        from dotenv import load_dotenv

        load_dotenv()

        self.retain_recent = retain_recent
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.update_sim_threshold = update_sim_threshold
        self.update_neighbors = update_neighbors
        self.extract_window = extract_window

        self.model = model or os.getenv("ANSWER_MODEL") or os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
        self.builder_model = (
            builder_model
            or os.getenv("BUILDER_MODEL")
            or os.getenv("ANSWER_MODEL")
            or "gpt-4o-mini"
        )

        self._client = None
        self._init_client()

        embed_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        try:
            ek = os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            eb = os.getenv("EMBEDDING_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")
            self.embedder = (
                APIEmbeddingBackend(model=embed_model_name, api_key=ek, api_base=eb)
                if ek else MockEmbeddingBackend()
            )
            if not ek:
                print("Warning: EMBEDDING_API_KEY/API_KEY not set, using mock embeddings")
        except Exception as e:
            print(f"Failed to init embedding backend: {e}. Using mock.")
            self.embedder = MockEmbeddingBackend()

        self.reranker = None
        if self.use_reranker:
            try:
                rk = os.getenv("RERANKER_API_KEY") or os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY")
                rb = os.getenv("RERANKER_API_BASE") or os.getenv("EMBEDDING_API_BASE") or os.getenv("API_BASE")
                self.reranker = (
                    Reranker(model=os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3"),
                             api_key=rk, api_base=rb, use_mock=False)
                    if rk else Reranker(use_mock=True)
                )
            except Exception as e:
                print(f"Failed to init reranker: {e}. Using mock.")
                self.reranker = Reranker(use_mock=True)

        # State
        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []
        self._facts: List[Fact] = []
        self._next_id = 0
        # operation counters (for fairness / mechanism auditing)
        self.ops = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}

    def _init_client(self):
        try:
            from openai import OpenAI

            ak = os.getenv("ANSWER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            ab = os.getenv("ANSWER_API_BASE") or os.getenv("API_BASE") or os.getenv("OPENAI_API_BASE")
            if ak:
                self._client = OpenAI(api_key=ak, base_url=ab)
        except ImportError:
            pass

    @property
    def name(self) -> str:
        rr = "+rerank" if self.use_reranker else ""
        return f"Mem0(facts+update-loop{rr}, k={self.top_k})"

    # ---- P1-8 backbone-integrated anchor: builder interface --------------
    # The ADD/UPDATE/DELETE loop mutates the fact store, so the canvas must be
    # rebuilt from the full current set each batch (replaces_store=True).
    anchor_label = "Mem0Facts"
    replaces_store = True

    def build_items(self, turns: List[ConversationTurn], verbose: int = 0):
        """Run Mem0's online update loop over this batch (mutating self._facts),
        then return the FULL current fact set as items for the backbone store.
        Facts carry no per-fact timestamp (Mem0's representation has none); any
        temporal grounding lives in the fact text iff the extractor kept it."""
        W = self.extract_window
        for i in range(0, len(turns), W):
            group = turns[i:i + W]
            for f_text in self._extract_facts(group):
                self._update_memory(f_text)
        return [
            {"content": f.text, "source_turns": [], "session_datetime": None}
            for f in self._facts
        ]

    def reset(self) -> None:
        self._history = []
        self._retained_history = []
        self._facts = []
        self._next_id = 0
        self.ops = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}

    def process_turn(self, turn: ConversationTurn) -> None:
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        turns_to_process = [t for t in self._history if t not in retained_turns]
        # Process sequentially so UPDATE/DELETE can fire as later turns revise
        # facts established by earlier turns -- the online Mem0 loop, batched.
        W = self.extract_window
        for i in range(0, len(turns_to_process), W):
            group = turns_to_process[i:i + W]
            facts = self._extract_facts(group)
            for f_text in facts:
                self._update_memory(f_text)
        self._retained_history = retained_turns
        self._history = list(retained_turns)

    # ---- Mem0 mechanism: extraction + update loop ------------------------

    @staticmethod
    def _dated_text(turns: List[ConversationTurn]) -> str:
        """Turn text with absolute-date markers, so the fact extractor CAN keep
        temporal grounding (matching the verbatim anchor's input); a dropped
        date then reflects Mem0's own lossiness, not a starved harness."""
        out = []
        for t in turns:
            sdt = getattr(t, "session_datetime", None)
            dt = f" (Session: {sdt})" if sdt else ""
            out.append(f"[Turn {t.turn_id}{dt}] User: {t.user}\nAssistant: {t.assistant}\n")
        return "".join(out)

    def _extract_facts(self, turns: List[ConversationTurn]) -> List[str]:
        if self._client is None or not turns:
            return []
        text = self._dated_text(turns)
        prompt = (
            "Extract the salient, standalone facts stated in the following "
            "conversation excerpt. Each fact should be a concise declarative "
            "sentence that is meaningful on its own (resolve pronouns to names). "
            "Skip pleasantries and filler. Return ONLY a JSON object "
            '{"facts": ["...", "..."]}. If nothing salient, return {"facts": []}.\n\n'
            f"{text}\nJSON:"
        )
        try:
            raw = call_llm_with_retry(
                client=self._client, model=self.builder_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300, temperature=0, call_type="extract",
            ).strip()
            if "{" in raw:
                raw = raw[raw.index("{"): raw.rindex("}") + 1]
            return [str(x) for x in json.loads(raw).get("facts", []) if str(x).strip()]
        except Exception:
            return []

    def _update_memory(self, fact_text: str) -> None:
        """Run the ADD/UPDATE/DELETE/NOOP decision for one candidate fact."""
        emb = self.embedder.embed(fact_text)
        if not self._facts:
            self._add_fact(fact_text, emb)
            return

        sims = batch_cosine_similarity(emb, [f.embedding for f in self._facts])
        order = sorted(range(len(self._facts)), key=lambda i: sims[i], reverse=True)
        neighbors = [self._facts[i] for i in order[: self.update_neighbors]]
        top_sim = sims[order[0]] if order else 0.0

        # No similar memory -> straight ADD (Mem0 skips the LLM here).
        if top_sim < self.update_sim_threshold:
            self._add_fact(fact_text, emb)
            return

        op, target_id, new_text = self._decide_op(fact_text, neighbors)
        if op == "ADD":
            self._add_fact(fact_text, emb)
        elif op == "UPDATE" and target_id is not None:
            self._apply_update(target_id, new_text or fact_text)
        elif op == "DELETE" and target_id is not None:
            self._apply_delete(target_id)
            self._add_fact(fact_text, emb)  # add the corrected fact
        else:  # NOOP
            self.ops["NOOP"] += 1

    def _decide_op(self, fact_text: str, neighbors: List[Fact]):
        if self._client is None:
            return "ADD", None, None
        listing = "\n".join(f'- (id={f.id}) "{f.text}"' for f in neighbors)
        prompt = (
            "You maintain a memory of facts. A NEW candidate fact has arrived. "
            "Compare it to the most similar EXISTING facts and choose ONE operation:\n"
            "- ADD: the candidate is genuinely new information.\n"
            "- UPDATE: the candidate refines/extends one existing fact (give the merged text).\n"
            "- DELETE: the candidate contradicts/supersedes one existing fact (the old one is now wrong).\n"
            "- NOOP: the candidate is already fully captured by an existing fact.\n\n"
            f'NEW candidate: "{fact_text}"\n\n'
            f"EXISTING similar facts:\n{listing}\n\n"
            'Return ONLY JSON: {"op": "ADD|UPDATE|DELETE|NOOP", "id": <existing id or null>, '
            '"text": "<merged/updated text, only for UPDATE>"}'
        )
        try:
            raw = call_llm_with_retry(
                client=self._client, model=self.builder_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0, call_type="extract",
            ).strip()
            if "{" in raw:
                raw = raw[raw.index("{"): raw.rindex("}") + 1]
            d = json.loads(raw)
            op = str(d.get("op", "ADD")).upper()
            tid = d.get("id")
            tid = int(tid) if tid is not None and str(tid).strip() != "" else None
            return op, tid, d.get("text")
        except Exception:
            return "ADD", None, None

    def _add_fact(self, text: str, emb) -> None:
        self._facts.append(Fact(id=self._next_id, text=text, embedding=emb))
        self._next_id += 1
        self.ops["ADD"] += 1

    def _apply_update(self, target_id: int, new_text: str) -> None:
        for f in self._facts:
            if f.id == target_id:
                f.text = new_text
                f.embedding = self.embedder.embed(new_text)
                self.ops["UPDATE"] += 1
                return
        # target vanished -> treat as ADD
        self._add_fact(new_text, self.embedder.embed(new_text))

    def _apply_delete(self, target_id: int) -> None:
        before = len(self._facts)
        self._facts = [f for f in self._facts if f.id != target_id]
        if len(self._facts) < before:
            self.ops["DELETE"] += 1

    # ---- retrieval / answer (pipeline-constant) --------------------------

    def answer_question(self, question: str) -> AgentResponse:
        start = time.time()
        retrieved, scores = [], []
        budget = context_budget_chars()
        eff_k = effective_k(self.top_k, budget)
        if self._facts:
            q_emb = self.embedder.embed(question)
            sims = batch_cosine_similarity(q_emb, [f.embedding for f in self._facts])
            ranked = sorted(zip(self._facts, sims), key=lambda x: x[1], reverse=True)
            if self.use_reranker and self.reranker is not None:
                cand_k = min(eff_k * 2, len(ranked))
                cand = [f for f, _ in ranked[:cand_k]]
                rr = self.reranker.rerank(question, [f.text for f in cand], top_k=eff_k)
                retrieved = [cand[idx] for idx, _ in rr]
                scores = [sc for _, sc in rr]
            else:
                top = ranked[:eff_k]
                retrieved = [f for f, _ in top]
                scores = [sc for _, sc in top]
            keep = len(fill_to_budget(retrieved, lambda f: f.text, budget))
            retrieved, scores = retrieved[:keep], scores[:keep]

        parts = []
        if retrieved:
            parts.append("## Retrieved Memory (facts from earlier conversation)")
            for i, f in enumerate(retrieved):
                parts.append(f"- {f.text}")
            parts.append("")
        if self._history:
            parts.append("## Recent Conversation")
            for t in self._history:
                parts.append(f"User: {t.user}")
                parts.append(f"Assistant: {t.assistant}")
                parts.append("")
        context = "\n".join(parts) if parts else "[No context available]"

        answer = self._generate_answer(context, question)
        latency = (time.time() - start) * 1000
        return AgentResponse(
            answer=answer,
            latency_ms=latency,
            metadata={
                "retrieved_facts": len(retrieved),
                "store_size": len(self._facts),
                "ops": dict(self.ops),
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
                client=self._client, model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0, call_type="gen",
            )
        except Exception as e:
            return f"Error: {e}"
