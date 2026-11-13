#!/usr/bin/env python3
"""
Paraphrase-tolerant store-survival check (camera-ready, reviewer tEhS).

store_survival.py decides survival by exact content-token co-occurrence, so a
fact that an artifact restates in other words is counted as lost. This script
re-tests the same chunk-pass/artifact-fail questions without string matching:

  1. Shortlist: embed "question + gold answer" and every stored artifact unit
     (content + quote + context, the same unit text as store_survival.py) with
     bge-m3; keep the top-K units by cosine similarity, separately for all
     units and for deliberate (non-window) units.
  2. Judge: an LLM reads each shortlisted unit and answers whether the unit,
     on its own, states the information needed to give the gold answer,
     allowing paraphrase. A fact "survives" if any shortlisted unit is judged
     sufficient.

Two controls bound the judge's error:
  - positive: the single source turn in which the gold's content tokens
    co-occur verbatim (exists for every diagnosable question) must be judged
    sufficient -> sensitivity;
  - negative: the top-K units of ANOTHER conversation's store (where the fact
    cannot be) must be judged insufficient -> false-positive rate.

Reads cached extraction canvases; the only API cost is the judge calls
(gpt-4o-mini by default) and bge-m3 embeddings.

Usage:
    python -m experiments.store_survival_paraphrase \
        --survival experiments/results/store_survival_w5.json \
        --cache-dir experiments/cache/extraction/632f01f4 \
        -o experiments/results/camera_ready/store_survival_paraphrase.json
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

from experiments.locomo_adapter import load_locomo, convert_to_eval_format
from experiments.store_survival import tokens

JUDGE_PROMPT = """You are checking whether a stored memory passage preserves a fact.

Question: {question}
Gold answer: {gold}

Passage:
\"\"\"
{passage}
\"\"\"

Does the passage, read on its own, state the information needed to give the gold answer to the question? Paraphrases and equivalent wordings count. A passage that is only on the same topic, or that mentions the entities without the specific fact asked for, does not count. If the gold answer is a date or duration, the passage must let a reader derive it from what the passage itself says.

Reply with exactly one word: YES or NO."""


def unit_texts(cache_file: Path):
    """(text, is_window) per canvas object; same unit definition as store_survival."""
    with open(cache_file) as f:
        data = json.load(f)
    out = []
    for obj in data.get("objects", []):
        parts = [str(obj[k]) for k in ("content", "quote", "context") if obj.get(k)]
        if parts:
            out.append((" ".join(parts), "[Turn" in (obj.get("content") or "")))
    return out


def turn_texts(conv):
    out = []
    for t in conv.turns:
        parts = [str(getattr(t, k)) for k in ("user", "assistant", "session_datetime")
                 if getattr(t, k, None)]
        if parts:
            out.append(" ".join(parts))
    return out


class Embedder:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("EMBEDDING_API_KEY", "x"),
                             base_url=os.environ["EMBEDDING_API_BASE"])
        self.model = os.getenv("EMBEDDING_MODEL", "bge-m3")

    def __call__(self, texts, batch=32):
        vecs = []
        for i in range(0, len(texts), batch):
            # bge-m3 caps at 8192 tokens; window units can be long, so clip.
            chunk = [t[:6000] for t in texts[i:i + batch]]
            r = self.client.embeddings.create(model=self.model, input=chunk)
            vecs.extend(d.embedding for d in r.data)
        v = np.asarray(vecs, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class Judge:
    def __init__(self, model):
        self.client = OpenAI(api_key=os.environ["SCORE_API_KEY"],
                             base_url=os.environ["SCORE_API_BASE"])
        self.model = model
        self.calls = 0

    def __call__(self, question, gold, passage):
        prompt = JUDGE_PROMPT.format(question=question, gold=gold, passage=passage[:8000])
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=3,
                    messages=[{"role": "user", "content": prompt}])
                self.calls += 1
                return r.choices[0].message.content.strip().upper().startswith("YES")
            except Exception as e:  # noqa: BLE001 - retry any transport error
                if attempt == 5:
                    raise
                import time
                time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--survival", required=True, help="store_survival.py detail JSON")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dataset", default="experiments/data/locomo10.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--judge-model", default=os.getenv("SCORE_MODEL", "gpt-4o-mini"))
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    surv = json.load(open(args.survival))
    details = [d for d in surv["details"] if d["bucket"] != "trivial_gold"]
    convs = {c.id: c for c in convert_to_eval_format(load_locomo(args.dataset))}
    conv_ids = sorted(convs)
    cache_dir = Path(args.cache_dir)
    embed, judge = Embedder(), Judge(args.judge_model)

    stores = {}
    for cid in conv_ids:
        units = unit_texts(cache_dir / f"{cid}.json")
        stores[cid] = (units, embed([u for u, _ in units]))
        print(f"  embedded {cid}: {len(units)} units", flush=True)

    def shortlist(qvec, cid, deliberate_only):
        units, vecs = stores[cid]
        idx = [i for i, (_, w) in enumerate(units) if not (deliberate_only and w)]
        if not idx:
            return []
        sims = vecs[idx] @ qvec
        order = np.argsort(-sims)[: args.top_k]
        return [(units[idx[j]][0], float(sims[j])) for j in order]

    qvecs = embed([f"{d['question']} {d['ground_truth']}" for d in details])

    def run_one(i):
        d = details[i]
        cid, q, gold = d["conversation_id"], d["question"], str(d["ground_truth"])
        qv = qvecs[i]
        res = dict(d)
        any_units = shortlist(qv, cid, False)
        del_units = shortlist(qv, cid, True)
        verdict = {}

        def j(passage):
            if passage not in verdict:
                verdict[passage] = judge(q, gold, passage)
            return verdict[passage]

        res["para_any"] = any(j(u) for u, _ in any_units)
        res["para_deliberate"] = any(j(u) for u, _ in del_units)
        res["top_sim_any"] = round(any_units[0][1], 3) if any_units else None
        res["top_sim_deliberate"] = round(del_units[0][1], 3) if del_units else None

        # Negative control: the same shortlist against another conversation's store.
        other = conv_ids[(conv_ids.index(cid) + 1) % len(conv_ids)]
        res["neg_any"] = any(judge(q, gold, u) for u, _ in shortlist(qv, other, False))

        # Positive control (diagnosable only): the verbatim source turn.
        if d["bucket"] != "inferential_gold":
            gt = tokens(gold)
            src = [t for t in turn_texts(convs[cid]) if gt <= tokens(t)]
            res["pos_source_turn"] = any(judge(q, gold, t) for t in src[:3]) if src else None
        return res

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(run_one, range(len(details))))

    def rate(rows, key):
        rows = [r for r in rows if r.get(key) is not None]
        k = sum(1 for r in rows if r[key])
        return k, len(rows)

    # Token-based classes as reported in the paper.
    def cls(r):
        if r["bucket"] == "write_time_loss":
            return "write_time_loss"
        if r["bucket"] == "query_time_miss":
            return "extraction_failure" if r["store_coverage_nonwindow"] < 1.0 else "query_time_miss"
        return "inferential_gold"

    summary = {}
    print("\n=== Paraphrase-tolerant survival, by token-based class ===")
    for c in ("write_time_loss", "extraction_failure", "query_time_miss", "inferential_gold"):
        rows = [r for r in out if cls(r) == c]
        a, n = rate(rows, "para_any")
        dl, _ = rate(rows, "para_deliberate")
        summary[c] = {"n": n, "survives_any_unit": a, "survives_deliberate_unit": dl}
        print(f"  {c:20s} n={n:3d}  any unit: {a:3d} ({a/n:5.1%})   "
              f"deliberate unit: {dl:3d} ({dl/n:5.1%})")
    diag = [r for r in out if r["bucket"] != "inferential_gold"]
    pos = rate(diag, "pos_source_turn")
    neg_d = rate(diag, "neg_any")
    neg_all = rate(out, "neg_any")
    lost = sum(1 for r in diag if not r["para_any"])
    only_window = sum(1 for r in diag if r["para_any"] and not r["para_deliberate"])
    extracted = sum(1 for r in diag if r["para_deliberate"])
    summary["diagnosable_reclassified"] = {
        "n": len(diag), "no_unit_sufficient": lost,
        "window_unit_only": only_window, "deliberate_unit_sufficient": extracted}
    summary["controls"] = {
        "positive_source_turn": pos, "negative_other_conversation_diagnosable": neg_d,
        "negative_other_conversation_all": neg_all}
    print(f"\n  Diagnosable {len(diag)} under the paraphrase-tolerant test:")
    print(f"    no unit sufficient (write-time loss): {lost} ({lost/len(diag):.1%})")
    print(f"    only a raw-window unit sufficient   : {only_window} ({only_window/len(diag):.1%})")
    print(f"    a deliberate artifact sufficient    : {extracted} ({extracted/len(diag):.1%})")
    print(f"\n  Controls: source turn judged sufficient {pos[0]}/{pos[1]}; "
          f"other-conversation store judged sufficient {neg_d[0]}/{neg_d[1]} (diagnosable), "
          f"{neg_all[0]}/{neg_all[1]} (all)")
    print(f"  Judge calls: {judge.calls} ({args.judge_model})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"survival_file": args.survival, "cache_dir": str(cache_dir),
                   "top_k": args.top_k, "judge_model": args.judge_model,
                   "embedding_model": embed.model, "judge_calls": judge.calls,
                   "summary": summary, "details": out}, f, indent=2, ensure_ascii=False)
    print(f"\nDetail written to {args.output}")


if __name__ == "__main__":
    main()
