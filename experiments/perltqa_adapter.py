"""
PerLTQA adapter: native-Chinese personal long-term memory QA (Du et al., SIGHAN
2024) loaded into the same LoCoMoConversation format the backbone consumes, so
the verbatim-chunks / sentence / artifact anchors run on it unchanged.

This is the M2 cross-lingual generalization probe. We use ONLY the dialogue
category of PerLT_QA (questions answerable from raw dialogue), discarding the
profile/event/relationship questions (those query pre-structured semantic memory,
not raw conversation, and would favor extraction by construction). Each
character's conversation = all of its dialogue records (PerLT_Mem), grouped by
the leading character index of the record key (e.g. "1_0_0#0" -> char "1").
Gold QA is third-party-annotated (not self-generated); the dialogues themselves
are ChatGPT-generated, so this is a clean-fact regime -- documented in the paper.
"""

import json
import os
import re
from typing import List, Dict

from experiments.data_gen import ConversationTurn
from experiments.locomo_adapter import LoCoMoConversation, LoCoMoQAPair


def _char_idx(ref) -> str:
    """Leading character index of a memory key/reference: '1_0_0#0' -> '1'."""
    if isinstance(ref, list):
        ref = ref[0] if ref else ""
    m = re.match(r"\s*(\d+)_", str(ref))
    return m.group(1) if m else None


def _build_turns(records: List, start_id: int = 1) -> List[ConversationTurn]:
    """records: list of (key, contents) where contents = {timestamp: [utterances]}.
    Pair consecutive utterances into user/assistant turns, verbatim, with the
    timestamp as session_datetime so temporal grounding survives."""
    turns = []
    tid = start_id
    for _key, contents in records:
        if not isinstance(contents, dict):
            continue
        for ts in contents:
            utts = [u for u in contents[ts] if isinstance(u, str) and u.strip()]
            i = 0
            while i < len(utts):
                user = utts[i].strip()
                assistant = utts[i + 1].strip() if i + 1 < len(utts) else "[No response]"
                turns.append(ConversationTurn(
                    turn_id=tid, user=user, assistant=assistant, session_datetime=ts,
                ))
                tid += 1
                i += 2
    return turns


def load_perltqa(path: str) -> List[LoCoMoConversation]:
    """path -> .../Dataset/zh/perltqa.json (perltmem.json expected alongside).
    Returns one LoCoMoConversation per character, with dialogue-category QA."""
    base = os.path.dirname(path)
    mem_path = os.path.join(base, "perltmem.json")
    qa_all = json.load(open(path, encoding="utf-8"))
    mem = json.load(open(mem_path, encoding="utf-8"))

    # Global index of every dialogue record, grouped by character index.
    dlg_by_char: Dict[str, list] = {}
    for rec in mem:
        if not isinstance(rec, dict):
            continue
        dialogues = rec.get("dialogues")
        if not isinstance(dialogues, dict):
            continue
        for key, val in dialogues.items():
            ci = _char_idx(key)
            if ci is None or not isinstance(val, dict) or "contents" not in val:
                continue
            dlg_by_char.setdefault(ci, []).append((key, val["contents"]))

    conversations = []
    for entry in qa_all:
        if not isinstance(entry, dict):
            continue
        for char_name, sections in entry.items():
            # dialogues QA is a dict: {dialogue_record_key: [QA, ...]}.
            dlg_qa = sections.get("dialogues") if isinstance(sections, dict) else None
            if not isinstance(dlg_qa, dict) or not dlg_qa:
                continue
            # Character index from the dialogue record keys (e.g. "4_0_0#0" -> "4").
            ci = next((_char_idx(k) for k in dlg_qa if _char_idx(k)), None)
            if ci is None or ci not in dlg_by_char:
                continue
            records = sorted(dlg_by_char[ci], key=lambda x: x[0])
            turns = _build_turns(records)
            if not turns:
                continue
            qa_pairs = []
            for rkey, qlist in dlg_qa.items():
                if not isinstance(qlist, list):
                    continue
                for q in qlist:
                    ques, ans = q.get("Question"), q.get("Answer")
                    if not ques or not ans:
                        continue
                    qa_pairs.append(LoCoMoQAPair(
                        question=ques.strip(),
                        answer=ans.strip(),
                        evidence=[rkey],
                        category=1,  # dialogue recall ~ single-hop; passes --categories 1,2,3
                    ))
            if not qa_pairs:
                continue
            conversations.append(LoCoMoConversation(
                id=f"perltqa_{ci}_{char_name}",
                speaker_a="User", speaker_b="AI",
                turns=turns, qa_pairs=qa_pairs,
                dialogue_id_to_turn={}, metadata={"source": "perltqa", "char": char_name},
            ))
    print(f"Loaded {len(conversations)} PerLTQA conversations "
          f"({sum(len(c.qa_pairs) for c in conversations)} dialogue QA)")
    return conversations
