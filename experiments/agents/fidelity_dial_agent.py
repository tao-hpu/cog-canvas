"""
Fidelity-Dial Builder: a controlled, single-variable fidelity knob.

The multi-schema anchors (SeCom / Mem0 / A-Mem) differ on many axes at once
(segmentation, compression, update loops, linking), so their accuracies are a
confounded scatter. This builder isolates fidelity as ONE continuous variable:
it chunks the transcript exactly like the verbatim-chunks anchor, then keeps a
fixed random fraction `retain` of each chunk's tokens, dropping the rest. Stored
text therefore carries a measured fraction `retain` of the source wording and
nothing else changes -- same backbone (date-grounding + hybrid + rerank + CoT),
same embedder/reranker/answerer.

Sweeping retain in {1.0, 0.8, 0.6, 0.4, 0.2} traces accuracy vs. fidelity as a
clean causal curve: retain=1.0 reproduces the verbatim-chunks anchor (a built-in
sanity check); lower values degrade only fidelity. The named-system scatter then
overlays this curve as real-world evidence that accuracy tracks distance from the
source.

Routed through CogCanvasAgent(items_builder=...), so it exposes build_items.
retain is read from DIAL_RETAIN (env) or passed explicitly.
"""

import os
import random
from typing import List

from experiments.data_gen import ConversationTurn


class FidelityDialBuilder:
    """Verbatim chunks with a fixed fraction of tokens randomly dropped.

    Append-only (no mutation), so replaces_store stays False -- the canvas
    accumulates degraded chunks across batches exactly like the chunks anchor.
    """

    replaces_store = False

    def __init__(self, retain: float = None, chunk_size: int = 512,
                 overlap: int = 100, seed: int = 7):
        if retain is None:
            try:
                retain = float(os.getenv("DIAL_RETAIN", "1.0"))
            except ValueError:
                retain = 1.0
        self.retain = max(0.0, min(1.0, retain))
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.seed = seed

    @property
    def anchor_label(self) -> str:
        return f"Dial{int(round(self.retain * 100)):03d}"

    @staticmethod
    def _dated_full_text(turns: List[ConversationTurn]):
        """Same dated transcript the verbatim-chunks anchor builds, with a
        (start, end, turn_id, session_datetime) map for source attribution."""
        full_text = ""
        turn_map = []
        for t in turns:
            s = len(full_text)
            sdt = getattr(t, "session_datetime", None)
            dt = f" (Session: {sdt})" if sdt else ""
            full_text += f"[Turn {t.turn_id}{dt}] User: {t.user}\nAssistant: {t.assistant}\n\n"
            turn_map.append((s, len(full_text), t.turn_id, sdt))
        return full_text, turn_map

    def _degrade(self, text: str, rng: random.Random) -> str:
        """Keep each whitespace token with probability `retain`; order preserved.
        retain=1.0 returns the text unchanged (verbatim sanity anchor)."""
        if self.retain >= 1.0:
            return text
        toks = text.split()
        if not toks:
            return text
        kept = [w for w in toks if rng.random() < self.retain]
        if not kept:  # never emit an empty chunk
            kept = [toks[rng.randrange(len(toks))]]
        return " ".join(kept)

    def build_items(self, turns: List[ConversationTurn], verbose: int = 0):
        full_text, turn_map = self._dated_full_text(turns)
        rng = random.Random(self.seed)
        items = []
        cursor = 0
        cs, ov = self.chunk_size, self.overlap
        idx = 0
        while cursor < len(full_text):
            end = min(cursor + cs, len(full_text))
            if end < len(full_text):
                search_start = max(cursor, int(end - cs * 0.2))
                nl = full_text.rfind("\n", search_start, end)
                if nl != -1:
                    end = nl + 1
            raw = full_text[cursor:end]
            sources, sdt = [], None
            for ts, te, tid, tdt in turn_map:
                if not (te <= cursor or ts >= end):
                    sources.append(tid)
                    if sdt is None and tdt:
                        sdt = tdt
            # deterministic per-chunk rng so repeated runs are identical
            chunk_rng = random.Random((self.seed, idx))
            content = self._degrade(raw, chunk_rng)
            items.append({
                "content": content,
                "source_turns": sources,
                "session_datetime": sdt,
            })
            idx += 1
            if end >= len(full_text):
                break
            cursor = end - ov
            if cursor <= 0 or cursor >= end:
                cursor = end
        return items
