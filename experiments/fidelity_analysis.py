"""
Multi-schema fidelity-curve analysis.

Reads the six anchor result files produced by run_fidelity_locomo.sh (one flat
retrieve->rerank->reason pipeline, only the storage representation varies) and:

  1. Recomputes MICRO accuracy from the per-question array -- never trusts the
     summary.overall_accuracy field (known macro/micro contamination bug).
  2. Breaks accuracy down by LoCoMo question category.
  3. Cluster-bootstraps (resample conversations) a 95% CI for each anchor's
     accuracy and for its gap vs the verbatim-chunks anchor.
  4. Runs McNemar's exact test, verbatim-chunks vs each lossier anchor.
  5. Plots the fidelity curve (pdf.fonttype=42, no Type-3 fonts).

Usage:
    python -m experiments.fidelity_analysis \
        --dir experiments/results/fidelity_locomo --suffix _gpt4o \
        --fig ../paper/figures/fidelity_curve.pdf
"""

import argparse
import json
import random
from math import comb
from pathlib import Path

# Fidelity-curve order: most verbatim -> most distilled.
CURVE = [
    ("rag", "Verbatim\nchunks"),
    ("secom", "SeCom\n(seg+compress)"),
    ("summarization", "Summary"),
    ("mem0", "Mem0\n(facts)"),
    ("amem", "A-Mem\n(notes)"),
    ("artifacts-flat", "Typed\nartifacts"),
]

N_BOOT = 10000
SEED = 7


def load(path):
    """Return {(conv_id, question, gt): passed_bool} and a per-category map."""
    data = json.loads(Path(path).read_text())
    out, cat = {}, {}
    for conv in data["conversations"]:
        for q in conv["questions"]:
            key = (conv["id"], q["question"], str(q.get("ground_truth", "")))
            passed = bool(q["passed"] if "passed" in q else q.get("correct"))
            out[key] = passed
            cat[key] = q.get("category_name") or str(q.get("category", "?"))
    return out, cat


def micro_acc(d):
    return sum(d.values()) / len(d) if d else 0.0


def by_category(d, cat):
    groups = {}
    for k, v in d.items():
        groups.setdefault(cat.get(k, "?"), []).append(v)
    return {c: (sum(vs) / len(vs), len(vs)) for c, vs in sorted(groups.items())}


def cluster_ci_abs(d):
    """Cluster-bootstrap 95% CI for a single anchor's accuracy."""
    keys = list(d)
    convs = {}
    for k in keys:
        convs.setdefault(k[0], []).append(d[k])
    cids = sorted(convs)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        pairs = [v for cid in (rng.choice(cids) for _ in cids) for v in convs[cid]]
        boots.append(sum(pairs) / len(pairs))
    boots.sort()
    return boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]


def cluster_ci_gap(a, b):
    """Cluster-bootstrap 95% CI for gap (a - b) over shared questions."""
    keys = sorted(set(a) & set(b))
    convs = {}
    for k in keys:
        convs.setdefault(k[0], []).append((a[k], b[k]))
    cids = sorted(convs)
    gap = (sum(a[k] for k in keys) - sum(b[k] for k in keys)) / len(keys)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        pairs = [p for cid in (rng.choice(cids) for _ in cids) for p in convs[cid]]
        boots.append(sum(x for x, _ in pairs) / len(pairs)
                     - sum(y for _, y in pairs) / len(pairs))
    boots.sort()
    return gap, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)], len(keys)


def mcnemar_exact(a, b):
    """Exact McNemar: a wins where a passed & b failed, etc. Two-sided p."""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if (not a[k]) and b[k])   # a fail, b pass
    b10 = sum(1 for k in keys if a[k] and (not b[k]))   # a pass, b fail
    n = b01 + b10
    if n == 0:
        return b10, b01, 1.0
    # two-sided exact binomial p at q=0.5
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * p)
    return b10, b01, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/results/fidelity_locomo")
    ap.add_argument("--suffix", default="_gpt4o")
    ap.add_argument("--fig", default=None, help="Path to write the curve PDF")
    ap.add_argument("--title", default="LoCoMo")
    args = ap.parse_args()

    d = Path(args.dir)
    loaded, cats = {}, {}
    for key, _label in CURVE:
        f = d / f"{key}{args.suffix}.json"
        if f.exists():
            loaded[key], cats[key] = load(f)

    present = [(k, lbl) for k, lbl in CURVE if k in loaded]
    if not present:
        print(f"No result files found in {d} with suffix {args.suffix}")
        return

    base_key = "rag"
    base = loaded.get(base_key)

    print(f"\n=== Fidelity curve ({args.title}, builder-steelman) ===")
    print(f"{'anchor':16} {'acc%':>7} {'95% CI':>16} {'n':>6}   "
          f"{'gap vs verbatim (pp) [CI]':>30}   McNemar p")
    rows = []
    for key, lbl in present:
        dd = loaded[key]
        acc = micro_acc(dd)
        lo, hi = cluster_ci_abs(dd)
        line = f"{key:16} {acc*100:7.1f} [{lo*100:5.1f},{hi*100:5.1f}] {len(dd):6}"
        gap_str, p_str = "", ""
        if base is not None and key != base_key:
            gap, glo, ghi, n = cluster_ci_gap(base, dd)
            w, l, p = mcnemar_exact(base, dd)
            gap_str = f"  {gap*100:+6.1f} [{glo*100:+5.1f},{ghi*100:+5.1f}]"
            p_str = f"   p={p:.2e} (v wins {w}, loses {l})"
        print(line + f"  {gap_str:>30} {p_str}")
        rows.append((key, lbl, acc, lo, hi))

    # Per-category breakdown (for the monotonicity / Zeng-preemption argument).
    print("\n=== Per-category accuracy (%) ===")
    all_cats = sorted({c for key in loaded for c in by_category(loaded[key], cats[key])})
    header = "category".ljust(22) + "".join(f"{k[:8]:>9}" for k, _ in present)
    print(header)
    for c in all_cats:
        row = c[:22].ljust(22)
        for key, _ in present:
            bc = by_category(loaded[key], cats[key])
            row += (f"{bc[c][0]*100:8.1f}" if c in bc else "       -") + " "
        print(row)

    if args.fig:
        _plot(rows, args.fig, args.title)


def _plot(rows, fig_path, title):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, no Type-3 (ARR-safe)
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    labels = [lbl for _, lbl, _, _, _ in rows]
    accs = [a * 100 for _, _, a, _, _ in rows]
    los = [(a - lo) * 100 for _, _, a, lo, _ in rows]
    his = [(hi - a) * 100 for _, _, a, _, hi in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(rows)))
    ax.errorbar(x, accs, yerr=[los, his], marker="o", color="#2c3e50",
                ecolor="#95a5a6", capsize=4, linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("QA accuracy (%)")
    ax.set_xlabel("storage representation  (verbatim → distilled)")
    ax.set_title(f"Fidelity curve: accuracy falls as representation departs from source ({title})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path)
    print(f"\nFigure written: {fig_path}")


if __name__ == "__main__":
    main()
