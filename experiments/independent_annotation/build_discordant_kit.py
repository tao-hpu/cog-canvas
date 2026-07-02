"""Build a blind human-annotation kit focused on the HEADLINE-DRIVING items:
the chunk-vs-artifact DISCORDANT pairs (where the two representations disagree
and which therefore drive the +15.9pp / +22.0pp gaps) plus abstention cases.

Reviewer ask (⑧): enlarge the human-validation sample, concentrated on the
discordant pairs and abstention cases rather than a generic stratified sample,
so the LLM judge is validated exactly where the headline is decided.

Design. For each sampled discordant LoCoMo question we emit BOTH systems' answers
as separate, blind, shuffled correctness items (question + gold + one answer ->
human labels CORRECT/INCORRECT, never told which system). Abstention items come
from LongMemEval abstention questions (both systems), labeled ABSTAINED/ANSWERED.
The system identity and the LLM-judge verdict are written to a separate KEY file
(NOT embedded in the HTML) so we can later compute human-vs-judge agreement on
exactly the contested items.

Reuses the audited HTML template from build_independent_kit.py verbatim.

Usage:
    python build_discordant_kit.py
Outputs:
    annotate_discordant.html      -> send ONLY this to the annotator
    discordant_KEY.json           -> keep private; used by score_discordant.py
"""

import json
import random
from pathlib import Path

from build_independent_kit import HTML  # reuse the exact audited template

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
OUT_HTML = HERE / "annotate_discordant.html"
OUT_KEY = HERE / "discordant_KEY.json"

# Canonical headline runs (same files clustered_ci.py / the paper Table use).
LOCOMO_CHUNKS = "locomo_chunks_C_nograph_10_cat123.json"
LOCOMO_ARTIFACTS = "locomo_full_10_cat123_0609.json"
LME_CHUNKS = "lme_s_chunks_nograph_500.json"
LME_ARTIFACTS = "lme_s_cogcanvas_full_500.json"

N_DISCORDANT_QUESTIONS = 40   # each yields 2 blind correctness items (both systems)
N_ABSTENTION_QUESTIONS = 12   # each yields 2 blind abstention items (both systems)
SEED = 13


def load(fname):
    d = json.loads((RESULTS / fname).read_text())
    out = {}
    for c in d["conversations"]:
        for q in c["questions"]:
            key = (c["id"], q["question"], str(q["ground_truth"]))
            passed = str(q.get("passed", q.get("correct"))).lower() in ("true", "1")
            out[key] = {
                "answer": q.get("answer", ""),
                "gold": str(q["ground_truth"]),
                "question": q["question"],
                "category": q.get("category_name", q.get("category")),
                "is_abstention": str(q.get("is_abstention", "")).lower() in ("true", "1"),
                "passed": passed,
            }
    return out


def main():
    rng = random.Random(SEED)
    lch, lar = load(LOCOMO_CHUNKS), load(LOCOMO_ARTIFACTS)
    mch, mar = load(LME_CHUNKS), load(LME_ARTIFACTS)

    # --- discordant LoCoMo questions (judge says one right, one wrong) ---
    shared = [k for k in (set(lch) & set(lar)) if lch[k]["passed"] != lar[k]["passed"]]
    # stratify by category so temporal/multi-hop/single-hop all appear
    by_cat = {}
    for k in shared:
        by_cat.setdefault(lch[k]["category"], []).append(k)
    picked = []
    cats = sorted(by_cat)
    per = max(1, N_DISCORDANT_QUESTIONS // len(cats))
    for c in cats:
        rng.shuffle(by_cat[c])
        picked += by_cat[c][:per]
    rng.shuffle(picked)
    picked = picked[:N_DISCORDANT_QUESTIONS]

    main_items, key = [], {}
    for k in picked:
        for system, rec in (("chunks", lch[k]), ("artifacts", lar[k])):
            iid = f"d{len(main_items):04d}"
            main_items.append({"id": iid, "question": rec["question"],
                               "gold": rec["gold"], "answer": rec["answer"] or "(empty answer)"})
            key[iid] = {"phase": "main", "system": system,
                        "judge_correct": rec["passed"], "category": rec["category"]}
    rng.shuffle(main_items)

    # --- abstention: LongMemEval abstention questions, both systems ---
    abst_keys = [k for k in (set(mch) & set(mar)) if mch[k]["is_abstention"]]
    rng.shuffle(abst_keys)
    abst_keys = abst_keys[:N_ABSTENTION_QUESTIONS]
    cat5_items = []
    for k in abst_keys:
        for system, rec in (("chunks", mch[k]), ("artifacts", mar[k])):
            iid = f"a{len(cat5_items):04d}"
            cat5_items.append({"id": iid, "question": rec["question"],
                               "answer": rec["answer"] or "(empty answer)"})
            key[iid] = {"phase": "abstention", "system": system,
                        "judge_correct": rec["passed"]}
    rng.shuffle(cat5_items)

    html = (HTML
            .replace("__MAIN_JSON__", json.dumps(main_items, ensure_ascii=False))
            .replace("__CAT5_JSON__", json.dumps(cat5_items, ensure_ascii=False)))
    OUT_HTML.write_text(html, encoding="utf-8")
    OUT_KEY.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_HTML}")
    print(f"  discordant LoCoMo questions sampled: {len(picked)} "
          f"(from {len(shared)} total discordant) -> {len(main_items)} blind correctness items")
    print(f"  abstention questions: {len(abst_keys)} -> {len(cat5_items)} blind items")
    print(f"  total items: {len(main_items)+len(cat5_items)}")
    print(f"Wrote {OUT_KEY} (private; maps blind id -> system + judge verdict)")
    print("\nSend ONLY annotate_discordant.html. They return annotations_export.json.")


if __name__ == "__main__":
    main()
