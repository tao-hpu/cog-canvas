"""
Sentence-Verbatim Builder: a fine-grained but *lossless* store.

The verbatim-chunks anchor (~512-char windows) and the typed-artifact store
differ on TWO axes at once: granularity (long window vs. short item) AND fidelity
(raw source vs. lossy distillation). The chunks-vs-artifacts gap could therefore
be a granularity effect rather than a fidelity one. This builder isolates that:
it stores the transcript at *artifact-scale granularity* (one or more whole
sentences packed to <= max_item_chars) while keeping the wording fully verbatim
(no extraction, no compression, no paraphrase, sentences never split).

It is the orthogonalizing control for the budget/granularity confound: if the
sentence-verbatim store stays near the verbatim-chunks anchor and well above the
typed artifacts, the gap is fidelity, not granularity. Routed through
CogCanvasAgent(items_builder=...) so only the stored text changes -- same dated
transcript, same backbone (date-grounding + hybrid + rerank + CoT), same
embedder/reranker/answerer.

Punctuation handles both English (".!?") and Chinese (CJK terminators) so the
same builder serves the LoCoMo control and the PerLTQA cross-lingual run.
``max_item_chars`` is read from SENT_MAX_CHARS (env) or passed explicitly.
"""

import os
import re
from typing import List

from experiments.data_gen import ConversationTurn


# Split AFTER a sentence terminator, keeping the terminator with its sentence.
# CJK enders (。！？；…) split unconditionally -- Chinese has no inter-sentence
# spaces -- except when a closing quote/bracket follows (kept attached). ASCII
# enders (. ! ?) split only at a word boundary (space / end / closer) so that
# "Dr.", "U.S.", and "3.14" are not broken.
_SENT_SPLIT = re.compile(
    r'(?<=[。！？；…])(?![”」』）)])'
    r'|(?<=[.!?])(?=\s|$|["”」』）)])'
)


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


class SentenceChunkBuilder:
    """Verbatim transcript stored at sentence granularity, packed to a cap.

    Append-only (no mutation), so replaces_store stays False -- the canvas
    accumulates sentence units across batches exactly like the chunks anchor.
    """

    replaces_store = False

    def __init__(self, max_item_chars: int = None):
        if max_item_chars is None:
            try:
                max_item_chars = int(os.getenv("SENT_MAX_CHARS", "160"))
            except ValueError:
                max_item_chars = 160
        self.max_item_chars = max(40, max_item_chars)

    @property
    def anchor_label(self) -> str:
        return f"SentVerbatim{self.max_item_chars:03d}"

    def _emit_role(self, turn_id, dt_marker, sdt, role, text, items):
        """Pack whole sentences of one utterance into <=max_item_chars items,
        each self-contained with the same dated turn marker the chunks anchor
        prints, so date grounding is identical and only granularity changes."""
        prefix = f"[Turn {turn_id}{dt_marker}] {role}: "
        budget = self.max_item_chars
        cur = ""
        for sent in _split_sentences(text):
            # A single sentence longer than the cap stands alone (never split).
            if cur and len(cur) + 1 + len(sent) > budget:
                items.append({
                    "content": prefix + cur,
                    "source_turns": [turn_id],
                    "session_datetime": sdt,
                })
                cur = sent
            else:
                cur = sent if not cur else f"{cur} {sent}"
        if cur:
            items.append({
                "content": prefix + cur,
                "source_turns": [turn_id],
                "session_datetime": sdt,
            })

    def build_items(self, turns: List[ConversationTurn], verbose: int = 0):
        items = []
        for t in turns:
            sdt = getattr(t, "session_datetime", None)
            dt_marker = f" (Session: {sdt})" if sdt else ""
            if (t.user or "").strip():
                self._emit_role(t.turn_id, dt_marker, sdt, "User", t.user, items)
            if (t.assistant or "").strip():
                self._emit_role(t.turn_id, dt_marker, sdt, "Assistant", t.assistant, items)
        return items
