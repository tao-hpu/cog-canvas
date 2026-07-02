"""
EMem-style near-verbatim event-unit builder (P1-a spectrum anchor).

EMem (Zhou & Han, 2025, arXiv:2511.17208) reports strong LoCoMo accuracy with
a training-free memory whose units are "enriched elementary discourse units
(EDUs) -- self-contained statements with normalized entities and source turn
attributions", built to "preserve information in a non-compressive form ...
rather than more lossy". The paper cites EMem as the high-fidelity end of the
extraction spectrum; this builder measures that end *inside* the controlled
harness, reproducing the EDU construction mechanism (the stored-representation
axis) while retrieval stays the fixed backbone -- exactly like the
Mem0/A-Mem/SeCom mechanism anchors.

Faithfulness constraints (what makes an EDU, per the paper):
  1. NON-COMPRESSIVE: every piece of information in the turn must be covered
     by some unit; nothing is filtered for salience.
  2. NEAR-VERBATIM: units copy source wording; the only permitted rewrite is
     entity normalization (pronoun -> referent name) plus the minimal glue
     needed for self-containment. No summarizing; numbers/dates/hedges kept.
  3. PROVENANCE: each unit carries the same dated turn marker the chunks and
     sentence-verbatim anchors print, so date grounding is identical across
     representations and only unit construction changes.

A non-compressive design must not silently drop a turn (that is the failure
mode of the typed-artifact extractor), so on JSON-parse failure -- after one
reformat retry -- the turn falls back to sentence-verbatim units and the
fallback is counted (``fallback_turns``).

Env knobs: EMEM_MODEL (default EXTRACTOR_MODEL -> gpt-4o-mini);
EMEM_CONTEXT_TURNS (default 3) prior turns shown for pronoun resolution only.
"""

import json
import os
import re
from typing import List

from experiments.data_gen import ConversationTurn
from experiments.agents.sentence_chunk_agent import _split_sentences

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_PROMPT = """Decompose the NEW UTTERANCE into elementary discourse units (EDUs).

Rules:
1. COVER EVERYTHING: every piece of information in the utterance must appear in some EDU. Do not filter by importance. Do not omit numbers, dates, names, quantities, hedges, or qualifiers.
2. STAY VERBATIM: copy the original wording. The only allowed edits are (a) replacing a pronoun with the name it refers to, and (b) adding the minimal words needed to make the EDU a self-contained statement. Never paraphrase, summarize, or generalize.
3. SELF-CONTAINED: each EDU must be understandable on its own (a reader who sees only this EDU knows who did what).
4. Output a JSON array of strings, nothing else. If the utterance carries no information (e.g. "haha", "ok"), output [].

CONTEXT (earlier turns, for resolving pronouns only -- do NOT extract from it):
{context}

NEW UTTERANCE by {speaker}:
{text}

JSON array:"""


class EMemBuilder:
    """LLM-built near-verbatim EDU store; append-only like the chunks anchor."""

    replaces_store = False
    anchor_label = "EMemEDU"

    def __init__(self, model: str = None, context_turns: int = None):
        self.model = (
            model
            or os.getenv("EMEM_MODEL")
            or os.getenv("EXTRACTOR_MODEL")
            or os.getenv("BUILDER_MODEL")
            or "gpt-4o-mini"
        )
        if context_turns is None:
            try:
                context_turns = int(os.getenv("EMEM_CONTEXT_TURNS", "3"))
            except ValueError:
                context_turns = 3
        self.context_turns = max(0, context_turns)
        self.fallback_turns = 0

        from openai import OpenAI

        ak = (
            os.getenv("EXTRACTOR_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        ab = (
            os.getenv("EXTRACTOR_API_BASE")
            or os.getenv("API_BASE")
            or os.getenv("OPENAI_API_BASE")
        )
        self._client = OpenAI(api_key=ak, base_url=ab) if ak else None

    # ---- LLM decomposition -------------------------------------------------

    @staticmethod
    def _parse_units(raw: str):
        raw = _FENCE.sub("", (raw or "").strip()).strip()
        data = json.loads(raw)
        if isinstance(data, str):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("not a JSON array")
        return [u.strip() for u in data if isinstance(u, str) and u.strip()]

    def _extract_units(self, context: str, speaker: str, text: str, verbose: int) -> List[str]:
        from experiments.llm_utils import call_llm_with_retry

        prompt = _PROMPT.format(context=context or "(none)", speaker=speaker, text=text)
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            raw = call_llm_with_retry(
                client=self._client,
                model=self.model,
                messages=messages,
                max_tokens=1200,
                temperature=0,
                verbose=False,
                call_type="extract",
            )
            try:
                units = self._parse_units(raw)
            except (ValueError, json.JSONDecodeError):
                if attempt == 0:
                    messages = [{"role": "user", "content": prompt
                                 + "\n\nReturn ONLY a valid JSON array of strings."}]
                    continue
                break
            # Coverage guard: an empty decomposition of a substantive utterance
            # violates the non-compressive contract -> verbatim fallback.
            if not units and len(text.strip()) > 60:
                break
            return units
        self.fallback_turns += 1
        if verbose:
            print(f"   [EMem] fallback to sentence-verbatim ({self.fallback_turns} turns so far)")
        return _split_sentences(text)

    # ---- builder interface (same as SentenceChunkBuilder) ------------------

    def build_items(self, turns: List[ConversationTurn], verbose: int = 0):
        items = []
        history: List[str] = []
        for t in turns:
            sdt = getattr(t, "session_datetime", None)
            dt_marker = f" (Session: {sdt})" if sdt else ""
            for role, text in (("User", t.user), ("Assistant", t.assistant)):
                text = (text or "").strip()
                if not text:
                    continue
                context = "\n".join(history[-(2 * self.context_turns):])
                prefix = f"[Turn {t.turn_id}{dt_marker}] {role}: "
                for unit in self._extract_units(context, role, text, verbose):
                    items.append({
                        "content": prefix + unit,
                        "source_turns": [t.turn_id],
                        "session_datetime": sdt,
                    })
                history.append(f"[Turn {t.turn_id}] {role}: {text}")
        return items
