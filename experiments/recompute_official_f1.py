"""Recompute LoCoMo official lexical F1 offline from result JSONs.

Reads per-question answer/ground_truth strings and re-applies the official
LoCoMo scoring (normalize -> Porter stem -> token F1), independent of any
stored f1_score field (which may be judge-derived in some run paths).

Usage:
    python -m experiments.recompute_official_f1 <result.json> [<result.json> ...]
"""

import json
import sys
from collections import defaultdict

from experiments.runner_locomo import compute_f1_score


def recompute(path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    f1s = []
    by_cat = defaultdict(list)
    stored = []
    for conv in data["conversations"]:
        for q in conv["questions"]:
            f1, _, _, _, _ = compute_f1_score(q["answer"], q["ground_truth"])
            f1s.append(f1)
            by_cat[q.get("category_name", str(q.get("category")))].append(f1)
            if "f1_score" in q:
                stored.append(q["f1_score"])

    n = len(f1s)
    mean_f1 = sum(f1s) / n if n else 0.0
    print(f"\n{path}")
    print(f"  questions: {n}")
    print(f"  official lexical F1 (per-question micro mean): {mean_f1 * 100:.1f}")
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        print(f"    {cat}: {sum(vals) / len(vals) * 100:.1f}  (n={len(vals)})")
    if stored:
        mean_stored = sum(stored) / len(stored)
        print(f"  stored f1_score mean (may be judge-derived): {mean_stored * 100:.1f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        recompute(p)
