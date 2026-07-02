#!/usr/bin/env python3
"""Tier-2 abstention frontier analysis.

Builds the (cat1-3 answerable accuracy, cat5 abstention rate) frontier for the
verbatim-chunks pipeline and overlays the new LLM-based refusers (M1 evidence
sufficiency, M2 answer-then-verify) on the score-threshold gate baseline.

For each mechanism we report:
  - cat1-3 answerable accuracy (weighted mean over categories 1,2,3)
  - cat5 abstention rate (mean over category 5)
and we judge whether a new mechanism DOMINATES the baseline frontier
(strictly Pareto-better: >= on both axes, > on at least one, vs every gate point
at a comparable abstention level), plus McNemar tests against the no-gate
baseline (answerable loss on cat1-3, abstention gain on cat5).

Usage: python -m experiments.analyze_abstention_frontier
Paths are hard-coded to experiments/results/*.json (edit MECHANISMS below).
"""
import json
import glob
import os
from collections import defaultdict

RESULTS = "experiments/results"


def load_questions(path):
    """Return list of (category, passed_bool, question_text) for a result file."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for conv in d.get("conversations", []):
        for q in conv.get("questions", []):
            out.append((q.get("category"), bool(q.get("passed")), q.get("question", "")))
    return out


def cat_breakdown(questions):
    cats = defaultdict(list)
    for c, passed, _ in questions:
        cats[c].append(1 if passed else 0)
    return cats


def answerable_acc(questions):
    """Weighted accuracy over categories 1,2,3 (pooled per-question)."""
    vals = [1 if p else 0 for c, p, _ in questions if c in (1, 2, 3)]
    return (100.0 * sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def abstention_rate(questions):
    """Abstention (correct-refuse) rate over category 5."""
    vals = [1 if p else 0 for c, p, _ in questions if c == 5]
    return (100.0 * sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def mcnemar(paired):
    """paired: list of (a_pass, b_pass). Returns (b01, b10, chi2_cc, approx_p)."""
    import math

    b01 = sum(1 for a, b in paired if not a and b)  # b better
    b10 = sum(1 for a, b in paired if a and not b)  # a better
    n = b01 + b10
    if n == 0:
        return b01, b10, 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n  # continuity-corrected
    # survival of chi-square with 1 dof = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return b01, b10, chi2, p


def merge_by_question(qa, qb):
    """Pair two result sets by question text (within a category)."""
    # index b by (category, question)
    idx = {}
    for c, p, q in qb:
        idx[(c, q)] = p
    paired = []
    for c, p, q in qa:
        if (c, q) in idx:
            paired.append((p, idx[(c, q)]))
    return paired


def load_combo(path_or_pair):
    """Accept a single combined file path, or (cat123_path, cat5_path) tuple."""
    if isinstance(path_or_pair, (list, tuple)):
        qs = []
        for p in path_or_pair:
            qs += load_questions(os.path.join(RESULTS, p))
        return qs
    return load_questions(os.path.join(RESULTS, path_or_pair))


# ---- mechanism registry -----------------------------------------------------
# Baseline frontier (on-disk). Answerer noted in parens (mini vs 4o).
BASELINE = {
    # no-gate gpt-4o (on-disk Table-14 anchor)
    "no-gate (4o)": ("locomo_chunks_10_cat123.json", "locomo_chunks_nograph_10_cat5.json"),
    "gate t=0.1 (mini)": ("locomo_chunks_abstgate_t0.1_cat123.json", "locomo_chunks_abstgate_t0.1_cat5.json"),
    "gate t=0.2 (mini)": ("locomo_chunks_abstgate_t0.2_cat123.json", "locomo_chunks_abstgate_t0.2_cat5.json"),
    "gate t=0.3 (mini)": ("locomo_chunks_abstgate_t0.3_cat123.json", "locomo_chunks_abstgate_t0.3_cat5.json"),
}

# New mechanisms (combined cat 1,2,3,5 single files). Filled by the run matrix.
MECHANISMS = {
    "no-gate (mini)": "locomo_chunks_mini_cat1235.json",
    "M1 llm t=0.0 (mini)": "locomo_refuser_llm_t0.0_cat1235.json",
    "M1 llm t=0.1 (mini)": "locomo_refuser_llm_t0.1_cat1235.json",
    "M1 llm t=0.2 (mini)": "locomo_refuser_llm_t0.2_cat1235.json",
    "M2 verify (mini)": "locomo_refuser_verify_cat1235.json",
}


def main():
    print("=" * 74)
    print("Tier-2 abstention frontier  (cat1-3 answerable acc  |  cat5 abstain rate)")
    print("=" * 74)

    rows = []  # (label, ans_acc, abst_rate, questions)

    for label, src in list(BASELINE.items()):
        try:
            qs = load_combo(src)
        except FileNotFoundError:
            continue
        a, na = answerable_acc(qs)
        b, nb = abstention_rate(qs)
        rows.append((label, a, b, qs))

    new_rows = []
    for label, src in MECHANISMS.items():
        p = os.path.join(RESULTS, src)
        if not os.path.exists(p):
            print(f"  [pending] {label}: {src} not found")
            continue
        qs = load_questions(p)
        a, na = answerable_acc(qs)
        b, nb = abstention_rate(qs)
        rows.append((label, a, b, qs))
        new_rows.append((label, a, b, qs))

    print(f"\n{'mechanism':<26}{'cat1-3 ans':>12}{'cat5 abstain':>14}")
    print("-" * 74)
    for label, a, b, _ in rows:
        print(f"{label:<26}{a:>11.1f}%{b:>13.1f}%")

    # dominance: a new mechanism dominates the gate frontier if for the gate
    # point at the closest abstention level it keeps strictly more answerable
    # accuracy (or more abstention at equal answerable).
    gate_pts = [(a, b) for (label, a, b, _) in rows if label.startswith("gate")]
    print("\nDominance vs gate frontier (does the new point Pareto-beat any gate?)")
    print("-" * 74)
    for label, a, b, _ in new_rows:
        beats = []
        for ga, gb in gate_pts:
            # new point dominates gate point if a>=ga and b>=gb, strictly better on one
            if a >= ga and b >= gb and (a > ga or b > gb):
                beats.append((ga, gb))
        verdict = "DOMINATES" if beats else "does not dominate"
        print(f"  {label:<26} {verdict}  (beats {len(beats)}/{len(gate_pts)} gate pts)")

    # McNemar vs no-gate (mini) if available, else vs no-gate (4o)
    base_label = "no-gate (mini)"
    base_row = next((r for r in rows if r[0] == base_label), None)
    if base_row is None:
        base_row = next((r for r in rows if r[0] == "no-gate (4o)"), None)
    if base_row is not None:
        base_qs = base_row[3]
        print(f"\nMcNemar vs {base_row[0]} (paired by question text)")
        print("-" * 74)
        for label, a, b, qs in new_rows:
            # cat1-3 answerable loss
            pa = [(c, p, q) for c, p, q in base_qs if c in (1, 2, 3)]
            pb = [(c, p, q) for c, p, q in qs if c in (1, 2, 3)]
            paired = merge_by_question(pa, pb)
            b01, b10, chi2, pval = mcnemar(paired)
            # cat5 abstention gain
            pa5 = [(c, p, q) for c, p, q in base_qs if c == 5]
            pb5 = [(c, p, q) for c, p, q in qs if c == 5]
            paired5 = merge_by_question(pa5, pb5)
            c01, c10, chi5, pval5 = mcnemar(paired5)
            print(f"  {label}")
            print(f"    cat1-3 answerable: base_only_correct={b10} new_only_correct={b01} "
                  f"chi2={chi2:.2f} p={pval:.4g}")
            print(f"    cat5  abstention : base_only_correct={c10} new_only_correct={c01} "
                  f"chi2={chi5:.2f} p={pval5:.4g}")


if __name__ == "__main__":
    main()
