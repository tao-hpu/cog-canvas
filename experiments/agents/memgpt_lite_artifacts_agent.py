"""
MemGPT-lite ARTIFACTS variant: identical hierarchical-memory agent, but the
ARCHIVAL tier stores EXTRACTED ARTIFACTS instead of verbatim turn text.

Controlled ablation for the paper's "verbatim chunks beat lossy extracted
artifacts" claim, moved INSIDE a stateful memory agent. The ONLY difference
from `MemGPTLiteAgent` is the representation written to archival memory:

    base  : archival entry text = "User: {turn.user}\nAssistant: {turn.assistant}"
    this  : one archival entry PER extracted artifact, text = content + quote

Everything else -- core_memory_size, archival_top_k, compression cadence,
hybrid (semantic + keyword) retrieval, RRF fusion, answer prompt, answerer
model, embedder -- is inherited byte-for-byte from the base class. We only
override `__init__` (to spin up a CogCanvas extractor), `reset` (to also reset
the extractor), and `_archive_turns` (to extract + store artifacts).

If a turn yields no artifacts, NOTHING is stored for it. That information loss
is the phenomenon under test; there is deliberately NO verbatim fallback.
"""

from typing import List
import os
import sys

from experiments.agents.memgpt_lite_agent import MemGPTLiteAgent, ArchivalEntry
from experiments.data_gen import ConversationTurn

from cogcanvas import Canvas


class MemGPTLiteArtifactsAgent(MemGPTLiteAgent):
    """MemGPT-lite with archival = extracted artifacts (not verbatim turns)."""

    def __init__(self, *args, **kwargs):
        # Build the base agent unchanged (embedder, answerer, core/archival config).
        super().__init__(*args, **kwargs)

        # Extractor config from env, mirroring CogCanvasAgent.
        self._extractor_model = os.getenv("EXTRACTOR_MODEL", "gpt-4o-mini")
        self._embedding_model = os.getenv("EMBEDDING_MODEL", "bge-large-zh-v1.5")

        # Counter for sanity logging (how many artifacts got archived this conv).
        self._archived_artifact_count = 0

        # Build the extractor Canvas (LLM extraction only; graph/VAGE off so the
        # store is just typed artifacts, matching the stateless artifact store).
        # NOTE: attribute is deliberately NOT named `_canvas` -- the runner
        # duck-types on `hasattr(agent, '_canvas')` to treat an agent as a
        # CogCanvasAgent (calling process_turn/on_compression with verbose=/reason=
        # kwargs our inherited methods do not accept). Naming it `_extractor_canvas`
        # keeps this agent on the SAME runner code path as plain memgpt-lite.
        self._extractor_canvas = None
        self._init_canvas()

    def _init_canvas(self) -> None:
        self._extractor_canvas = Canvas(
            extractor_model=self._extractor_model,
            embedding_model=self._embedding_model,
        )

    @property
    def name(self) -> str:
        kw_status = "+KW" if self.use_keyword_search else ""
        return (
            f"MemGPT-lite-Artifacts(core={self.core_memory_size}, "
            f"k={self.archival_top_k}{kw_status})"
        )

    def reset(self) -> None:
        """Reset base state AND the extractor between conversations."""
        super().reset()
        self._archived_artifact_count = 0
        # _init_canvas may run before base __init__ finished on first construction;
        # guard so reset() called from super().__init__ does not crash.
        self._init_canvas()

    def _archive_turns(self, turns: List[ConversationTurn]) -> None:
        """
        Archive turns as EXTRACTED ARTIFACTS.

        For each turn: run the CogCanvas extractor, then create one ArchivalEntry
        PER extracted artifact. The searchable/embedded text is the artifact's
        `content` + grounding `quote` (mirroring the stateless CogCanvas store,
        which embeds f"{content} {quote}"). Embedding uses the SAME embedder the
        base agent uses. Keyword index uses the base `_extract_keywords`.
        """
        if not turns:
            return

        artifact_texts: List[str] = []
        artifact_records = []  # (turn_id, content, quote)

        for turn in turns:
            session_datetime = getattr(turn, "session_datetime", None)
            try:
                result = self._extractor_canvas.extract(
                    user=turn.user,
                    assistant=turn.assistant,
                    metadata={"turn_id": turn.turn_id},
                    session_datetime=session_datetime,
                )
            except Exception as e:
                print(f"      [Artifacts] extract failed on turn {turn.turn_id}: {e}",
                      file=sys.stderr)
                continue

            for obj in result.objects:
                content = (getattr(obj, "content", "") or "").strip()
                quote = (getattr(obj, "quote", "") or "").strip()
                if not content and not quote:
                    continue
                # Searchable/embedded text = content + grounding quote.
                text = f"{content} {quote}".strip() if quote else content
                artifact_texts.append(text)
                artifact_records.append((turn.turn_id, content, quote))

        # If this batch of turns yielded no artifacts, store nothing (no fallback).
        if not artifact_texts:
            return

        # Same embedder as the base agent (keeps embedding identical).
        embeddings = self.embedder.embed_batch(artifact_texts)

        for (turn_id, content, quote), text, embedding in zip(
            artifact_records, artifact_texts, embeddings
        ):
            keywords = self._extract_keywords(text)

            # Reuse ArchivalEntry: `user` carries the artifact content, `assistant`
            # carries the grounding quote (these fields become the retrieved
            # context text at answer time; base prints "User:"/"Assistant:").
            entry = ArchivalEntry(
                turn_id=turn_id,
                user=content,
                assistant=(f"(grounding quote: {quote})" if quote else ""),
                embedding=embedding,
                keywords=keywords,
            )

            archival_idx = len(self._archival_memory)
            self._archival_memory.append(entry)
            self._archived_artifact_count += 1

            for kw in keywords:
                if kw not in self._keyword_index:
                    self._keyword_index[kw] = []
                self._keyword_index[kw].append(archival_idx)

        print(
            f"      [Artifacts] archived {len(artifact_texts)} artifacts from "
            f"{len(turns)} turns (conv total: {self._archived_artifact_count})",
            file=sys.stderr,
        )
