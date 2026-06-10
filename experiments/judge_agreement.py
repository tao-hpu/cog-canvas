"""Judge-human agreement study (paper W8/P0-3).

Samples 100 (question, gold, answer) triples from the same-batch Table 1 runs,
stratified 25 per (benchmark x pipeline) cell, and emits a *blind* annotation
sheet (no judge verdict, no pipeline label) plus a key file. After human
annotation, the score subcommand computes Cohen's kappa overall and stratified
by benchmark, pipeline, and answer length.

Usage:
    python -m experiments.judge_agreement sample
    # ... fill human_verdict column (CORRECT / INCORRECT) in the CSV ...
    python -m experiments.judge_agreement score
"""

import argparse
import csv
import json
import random
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OUT_DIR = RESULTS / "judge_agreement"

# (benchmark, pipeline, file, verdict_field)
SOURCES = [
    ("locomo", "chunks", "locomo_chunks_C_nograph_10_cat123.json", "passed"),
    ("locomo", "artifacts", "locomo_full_10_cat123_0609.json", "passed"),
    ("lme_s", "chunks", "lme_s_chunks_nograph_500.json", "correct"),
    ("lme_s", "artifacts", "lme_s_cogcanvas_full_500.json", "correct"),
]

PER_CELL = 25
SEED = 42
SHORT_ANSWER_TOKENS = 30  # whitespace tokens; review flags 16.7 vs 85.9 means


def load_items(benchmark, pipeline, fname, verdict_field):
    data = json.loads((RESULTS / fname).read_text())
    items = []
    for conv in data["conversations"]:
        for q in conv["questions"]:
            verdict = q.get(verdict_field)
            if verdict is None:
                continue
            items.append({
                "benchmark": benchmark,
                "pipeline": pipeline,
                "conversation_id": conv.get("id", ""),
                "category": q.get("category_name", ""),
                "question": q["question"],
                "ground_truth": str(q["ground_truth"]),
                "answer": str(q["answer"]),
                "judge_verdict": bool(verdict),
            })
    return items


def cmd_sample(_args):
    rng = random.Random(SEED)
    sampled = []
    for benchmark, pipeline, fname, field in SOURCES:
        items = load_items(benchmark, pipeline, fname, field)
        sampled.extend(rng.sample(items, PER_CELL))
        print(f"{benchmark}/{pipeline}: {len(items)} items -> {PER_CELL} sampled")
    rng.shuffle(sampled)

    OUT_DIR.mkdir(exist_ok=True)
    key = {}
    sheet_path = OUT_DIR / "annotation_sheet.csv"
    with sheet_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "benchmark", "category", "question",
                    "ground_truth", "answer", "human_verdict", "notes"])
        for i, it in enumerate(sampled, 1):
            qid = f"J{i:03d}"
            w.writerow([qid, it["benchmark"], it["category"], it["question"],
                        it["ground_truth"], it["answer"], "", ""])
            key[qid] = {
                "pipeline": it["pipeline"],
                "conversation_id": it["conversation_id"],
                "judge_verdict": it["judge_verdict"],
                "answer_tokens": len(it["answer"].split()),
            }
    (OUT_DIR / "key.json").write_text(json.dumps(key, indent=2))
    print(f"\nWrote {sheet_path} ({len(sampled)} rows, judge verdicts hidden)")
    print(f"Wrote {OUT_DIR / 'key.json'}")
    print("Fill human_verdict with CORRECT or INCORRECT, then run: "
          "python -m experiments.judge_agreement score")


def kappa(pairs):
    """Cohen's kappa for a list of (human_bool, judge_bool)."""
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    agree = sum(1 for h, j in pairs if h == j)
    po = agree / n
    ph = sum(1 for h, _ in pairs if h) / n
    pj = sum(1 for _, j in pairs if j) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, k


def cmd_score(_args):
    key = json.loads((OUT_DIR / "key.json").read_text())
    rows = list(csv.DictReader((OUT_DIR / "annotation_sheet.csv").open()))
    pairs, strata = [], {}
    skipped = 0
    for r in rows:
        hv = r["human_verdict"].strip().upper()
        if hv not in ("CORRECT", "INCORRECT"):
            skipped += 1
            continue
        k = key[r["id"]]
        h, j = hv == "CORRECT", bool(k["judge_verdict"])
        pairs.append((h, j))
        for name in (
            f"benchmark={r['benchmark']}",
            f"pipeline={k['pipeline']}",
            "answer=short" if k["answer_tokens"] <= SHORT_ANSWER_TOKENS else "answer=long",
        ):
            strata.setdefault(name, []).append((h, j))
    if skipped:
        print(f"WARNING: {skipped} rows without a valid human_verdict were skipped")

    po, k = kappa(pairs)
    tp = sum(1 for h, j in pairs if h and j)
    fn = sum(1 for h, j in pairs if h and not j)
    fp = sum(1 for h, j in pairs if not h and j)
    tn = sum(1 for h, j in pairs if not h and not j)
    print(f"\nOverall: n={len(pairs)} agreement={po:.1%} kappa={k:.3f}")
    print(f"Confusion (human x judge): both-correct={tp} human-only={fn} "
          f"judge-only={fp} both-incorrect={tn}")
    for name in sorted(strata):
        spo, sk = kappa(strata[name])
        print(f"  {name:22s} n={len(strata[name]):3d} "
              f"agreement={spo:.1%} kappa={sk:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample").set_defaults(func=cmd_sample)
    sub.add_parser("score").set_defaults(func=cmd_score)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
