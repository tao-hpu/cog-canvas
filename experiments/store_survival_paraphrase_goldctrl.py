#!/usr/bin/env python3
"""
Second positive control for store_survival_paraphrase.py.

The first positive control used the turn in which the gold's content tokens
co-occur, which need not be the turn that answers the question. LoCoMo ships
gold evidence annotations (dialogue ids per question); this script judges the
concatenated gold-evidence utterances (each prefixed with its session date and
speaker) with the same judge and prompt, and adds `pos_gold_evidence` to every
detail row of an existing output file.

Usage:
    python -m experiments.store_survival_paraphrase_goldctrl \
        experiments/results/camera_ready/store_survival_paraphrase.json
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from experiments.store_survival_paraphrase import Judge


def main():
    path = sys.argv[1]
    out = json.load(open(path))
    raw = json.load(open("experiments/data/locomo10.json"))
    evidence = {}
    for i, conv in enumerate(raw):
        c = conv["conversation"]
        utt = {}
        for k, v in c.items():
            if k.startswith("session_") and isinstance(v, list):
                date = c.get(f"{k}_date_time", "")
                for d in v:
                    utt[d["dia_id"]] = f"[{date}] {d['speaker']}: {d['text']}"
        for qa in conv["qa"]:
            ids = [e.strip() for ev in qa.get("evidence", []) for e in str(ev).replace(";", ",").split(",")]
            evidence[(f"locomo_{i:03d}", qa["question"])] = "\n".join(utt[e] for e in ids if e in utt)
    judge = Judge(out["judge_model"])

    def run(r):
        passage = evidence.get((r["conversation_id"], r["question"]), "")
        r["gold_evidence_chars"] = len(passage)
        r["pos_gold_evidence"] = judge(r["question"], str(r["ground_truth"]), passage) if passage else None
        return r

    with ThreadPoolExecutor(8) as ex:
        out["details"] = list(ex.map(run, out["details"]))
    json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)

    def cls(r):
        if r["bucket"] == "write_time_loss":
            return "write_time_loss"
        if r["bucket"] == "query_time_miss":
            return "extraction_failure" if r["store_coverage_nonwindow"] < 1.0 else "query_time_miss"
        return "inferential_gold"

    rows = out["details"]
    for name, sel in (("diagnosable 58", [r for r in rows if r["bucket"] != "inferential_gold"]),
                      ("inferential 87", [r for r in rows if r["bucket"] == "inferential_gold"]),
                      ("all", rows)):
        have = [r for r in sel if r["pos_gold_evidence"] is not None]
        ok = [r for r in have if r["pos_gold_evidence"]]
        print(f"{name}: gold evidence judged sufficient {len(ok)}/{len(have)}")
        none = sum(1 for r in ok if not r["para_any"])
        win = sum(1 for r in ok if r["para_any"] and not r["para_deliberate"])
        dl = sum(1 for r in ok if r["para_deliberate"])
        print(f"   among those {len(ok)}: no unit {none}, window only {win}, extracted artifact {dl}")
        for c in ("write_time_loss", "extraction_failure", "query_time_miss"):
            R = [r for r in ok if cls(r) == c]
            if R:
                print(f"     {c}: n={len(R)} any={sum(r['para_any'] for r in R)} "
                      f"extracted={sum(r['para_deliberate'] for r in R)}")


if __name__ == "__main__":
    main()
