"""
Typed-Artifacts Anchor (flat pipeline): the lossy end of the fidelity curve.

This is the most-distilled storage representation in the multi-schema fidelity
curve. The dialogue is distilled into typed structured objects -- using the same
taxonomy as CogCanvas's CanvasObject (decision / todo / key_fact / reminder /
insight / person_attribute / event / relationship) -- but, crucially, the
representation keeps only the *abstracted* fields (type + structured content +
context) and DROPS the verbatim quote. Departing furthest from the source text,
it is the lossy endpoint against which verbatim chunks are compared.

It runs in the SAME flat retrieve -> rerank -> reason pipeline as every other
anchor (verbatim chunks, SeCom, summary, Mem0, A-Mem), so the only variable that
moves across the whole curve is the storage representation. This is the
flat-pipeline counterpart of CogCanvas's graph-based typed-artifacts, added so
the artifacts endpoint is apples-to-apples with the rest of the curve.

FAIRNESS: same embedder (bge-m3), reranker, answerer (gpt-4o); extraction uses
the shared representation-BUILDER model (BUILDER_MODEL, default = gpt-4o
answerer as a steelman; gpt-4o-mini for the builder-invariance check).
"""

from typing import List, Dict, Any, Optional
import json
import time
import os

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

# Same taxonomy as cogcanvas.models.ObjectType.
ARTIFACT_TYPES = [
    "decision", "todo", "key_fact", "reminder", "insight",
    "person_attribute", "event", "relationship",
]


class Artifact:
    __slots__ = ("type", "content", "context", "embedding")

    def __init__(self, type, content, context, embedding):
        self.type = type
        self.content = content
        self.context = context
        self.embedding = embedding

    def render(self) -> str:
        out = f"[{self.type}] {self.content}"
        if self.context:
            out += f" ({self.context})"
        return out


class ArtifactsFlatAgent(Agent):
    """Typed-artifacts anchor in the flat pipeline (lossy curve endpoint)."""

    def __init__(
        self,
        model: str = None,
        embedding_model: str = None,
        builder_model: str = None,
        retain_recent: int = 5,
        top_k: int = 10,
        use_reranker: bool = True,
        extract_window: int = 4,  # turns per extraction call
    ):
        from dotenv import load_dotenv

        load_dotenv()

        self.retain_recent = retain_recent
        self.top_k = top_k
        self.use_reranker = use_reranker
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

        self._history: List[ConversationTurn] = []
        self._retained_history: List[ConversationTurn] = []
        self._artifacts: List[Artifact] = []

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
        return f"TypedArtifacts(flat{rr}, k={self.top_k})"

    def reset(self) -> None:
        self._history = []
        self._retained_history = []
        self._artifacts = []

    def process_turn(self, turn: ConversationTurn) -> None:
        self._history.append(turn)

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        turns_to_process = [t for t in self._history if t not in retained_turns]
        W = self.extract_window
        new_texts, new_objs = [], []
        for i in range(0, len(turns_to_process), W):
            group = turns_to_process[i:i + W]
            for obj in self._extract_artifacts(group):
                new_texts.append(obj.render())
                new_objs.append(obj)
        if new_texts:
            embs = self.embedder.embed_batch(new_texts)
            for obj, emb in zip(new_objs, embs):
                obj.embedding = emb
                self._artifacts.append(obj)
        self._retained_history = retained_turns
        self._history = list(retained_turns)

    def _extract_artifacts(self, turns: List[ConversationTurn]) -> List[Artifact]:
        if self._client is None or not turns:
            return []
        text = "".join(f"User: {t.user}\nAssistant: {t.assistant}\n" for t in turns)
        types = ", ".join(ARTIFACT_TYPES)
        prompt = (
            "Distill the following conversation excerpt into typed memory "
            "artifacts. Each artifact has a type (one of: " + types + "), a "
            "content field (a concise structured statement of the information, "
            "pronouns resolved to names), and a context field (why it matters). "
            "Abstract the information -- do NOT copy sentences verbatim.\n"
            'Return ONLY JSON: {"artifacts": [{"type": "...", "content": "...", '
            '"context": "..."}]}. If nothing salient, {"artifacts": []}.\n\n'
            f"{text}\nJSON:"
        )
        try:
            raw = call_llm_with_retry(
                client=self._client, model=self.builder_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500, temperature=0, call_type="extract",
            ).strip()
            if "{" in raw:
                raw = raw[raw.index("{"): raw.rindex("}") + 1]
            objs = json.loads(raw).get("artifacts", [])
            out = []
            for o in objs:
                if not isinstance(o, dict):
                    continue
                content = str(o.get("content", "")).strip()
                if not content:
                    continue
                t = str(o.get("type", "key_fact")).strip().lower()
                if t not in ARTIFACT_TYPES:
                    t = "key_fact"
                out.append(Artifact(type=t, content=content,
                                    context=str(o.get("context", "")), embedding=None))
            return out
        except Exception:
            return []

    def answer_question(self, question: str) -> AgentResponse:
        start = time.time()
        retrieved, scores = [], []
        budget = context_budget_chars()
        eff_k = effective_k(self.top_k, budget)
        if self._artifacts:
            q_emb = self.embedder.embed(question)
            sims = batch_cosine_similarity(q_emb, [a.embedding for a in self._artifacts])
            ranked = sorted(zip(self._artifacts, sims), key=lambda x: x[1], reverse=True)
            if self.use_reranker and self.reranker is not None:
                cand_k = min(eff_k * 2, len(ranked))
                cand = [a for a, _ in ranked[:cand_k]]
                rr = self.reranker.rerank(question, [a.render() for a in cand], top_k=eff_k)
                retrieved = [cand[idx] for idx, _ in rr]
                scores = [sc for _, sc in rr]
            else:
                top = ranked[:eff_k]
                retrieved = [a for a, _ in top]
                scores = [sc for _, sc in top]
            keep = len(fill_to_budget(retrieved, lambda a: a.render(), budget))
            retrieved, scores = retrieved[:keep], scores[:keep]

        parts = []
        if retrieved:
            parts.append("## Retrieved Memory (typed artifacts from earlier conversation)")
            for a in retrieved:
                parts.append(f"- {a.render()}")
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
                "retrieved_artifacts": len(retrieved),
                "store_size": len(self._artifacts),
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
