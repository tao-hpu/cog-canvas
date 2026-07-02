"""
Paired analysis for the stateful-agent verbatim-vs-artifacts ablation.

Loads two runner_locomo result JSONs (Arm A = memgpt-lite verbatim archival,
Arm B = memgpt-lite-artifacts), pairs per-question by (conv id, question,
ground_truth) exactly like clustered_ci.py, and reports:
  - overall LLM-judge accuracy per arm + per category (1 single-hop / 2 temporal
    / 3 multi-hop)
  - the paired gap in pp
  - McNemar discordant b/c counts and a two-sided exact binomial p-value

Usage:
    python -m experiments.mcnemar_memgpt_arms <armA.json> <armB.json>
"""

import json
import sys
from math import comb
from pathlib import Path

CAT_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop"}


def load(fname):
    data = json.loads(Path(fname).read_text())
    out = {}
    for conv in data["conversations"]:
        for q in conv["questions"]:
            key = (conv["id"], q["question"], str(q.get("ground_truth")))
            passed = bool(q["passed"] if "passed" in q else q.get("correct"))
            cat = q.get("category")
            out[key] = (passed, cat)
    return out


def acc(d, keys, cat=None):
    sel = [k for k in keys if (cat is None or d[k][1] == cat)]
    if not sel:
        return float("nan"), 0
    return sum(d[k][0] for k in sel) / len(sel), len(sel)


def exact_binom_two_sided(b, c):
    """Two-sided exact binomial p on discordants (n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) under Binom(n, 0.5), doubled, capped at 1.
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    A, B = load(fa), load(fb)
    keys = sorted(set(A) & set(B))
    print(f"Arm A (verbatim) : {fa}")
    print(f"Arm B (artifacts): {fb}")
    print(f"Paired questions : {len(keys)}  (A had {len(A)}, B had {len(B)})\n")

    print(f"{'arm':<12}{'overall':>10}{'single-hop':>13}{'temporal':>12}{'multi-hop':>12}{'N':>7}")
    for label, d in (("verbatim", A), ("artifacts", B)):
        ov, n = acc(d, keys)
        c1, n1 = acc(d, keys, 1)
        c2, n2 = acc(d, keys, 2)
        c3, n3 = acc(d, keys, 3)
        print(f"{label:<12}{ov*100:>9.1f}%{c1*100:>12.1f}%{c2*100:>11.1f}%{c3*100:>11.1f}%{n:>7}")
    print(f"  (per-cat N: single={n1}, temporal={n2}, multi={n3})\n")

    ovA = acc(A, keys)[0]
    ovB = acc(B, keys)[0]
    gap = (ovA - ovB) * 100
    print(f"Gap (verbatim - artifacts): {gap:+.1f} pp\n")

    # McNemar
    b = sum(1 for k in keys if A[k][0] and not B[k][0])  # A pass, B fail
    c = sum(1 for k in keys if not A[k][0] and B[k][0])  # A fail, B pass
    both = sum(1 for k in keys if A[k][0] and B[k][0])
    neither = sum(1 for k in keys if not A[k][0] and not B[k][0])
    p = exact_binom_two_sided(b, c)
    print("McNemar contingency (paired):")
    print(f"  both pass   : {both}")
    print(f"  both fail   : {neither}")
    print(f"  A>B (b, verbatim only): {b}")
    print(f"  B>A (c, artifacts only): {c}")
    print(f"  discordant n = {b + c}")
    print(f"  two-sided exact binomial p = {p:.4g}")
    sig = "YES" if p < 0.05 else "NO"
    print(f"  significant at 0.05: {sig}")
    if gap > 0:
        verdict = "verbatim ahead" if p < 0.05 else "no significant difference (verbatim numerically ahead)"
    elif gap < 0:
        verdict = "artifacts ahead" if p < 0.05 else "no significant difference (artifacts numerically ahead)"
    else:
        verdict = "exact tie"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
