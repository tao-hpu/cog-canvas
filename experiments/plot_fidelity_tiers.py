"""
Final fidelity figure: three-tier "verbatim vs extracted" cliff (LoCoMo,
fixed-context-budget, builder=gpt-4o steelman).

Bars sorted by accuracy, coloured by representation tier, with cluster-bootstrap
95% CIs and McNemar significance vs the verbatim-chunks anchor. pdf.fonttype=42
(no Type-3, ARR-safe).

Usage:
    python -m experiments.plot_fidelity_tiers \
        --dir experiments/results/fidelity_locomo_budget --suffix _gpt4o \
        --fig ../paper/figures/fidelity_curve.pdf
"""

import argparse
from pathlib import Path

from experiments.fidelity_analysis import (
    load, micro_acc, cluster_ci_abs, cluster_ci_gap, mcnemar_exact,
)


def filter_cats(data_cat, keep):
    """Restrict a (outcomes, category) pair to question categories in `keep`."""
    if not keep:
        return data_cat[0]
    out, cat = data_cat
    return {k: v for k, v in out.items() if cat.get(k) in keep}

# anchor -> (display label, tier)
META = {
    "rag":            ("Verbatim\nchunks",      "verbatim"),
    "secom":          ("SeCom\nseg+compress",   "verbatim"),
    "artifacts-flat": ("Typed\nartifacts",      "extracted"),
    "amem":           ("A-Mem\nnotes",          "extracted"),
    "mem0":           ("Mem0\nfacts",           "extracted"),
    "summarization":  ("Summary",               "summary"),
}
TIER_COLOR = {"verbatim": "#1b7837", "extracted": "#d9820b", "summary": "#b2182b"}
TIER_LABEL = {
    "verbatim": "Verbatim-grounded",
    "extracted": "Structured extraction",
    "summary": "Free-form summary",
}
BASE = "rag"


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/results/fidelity_locomo_budget")
    ap.add_argument("--suffix", default="_gpt4o")
    ap.add_argument("--fig", default="../paper/figures/fidelity_curve.pdf")
    ap.add_argument("--categories", default=None,
                    help="comma-separated category_names to keep (e.g. single-hop,temporal,multi-hop)")
    args = ap.parse_args()

    keep = set(args.categories.split(",")) if args.categories else None
    d = Path(args.dir)
    data = {}
    for key in META:
        f = d / f"{key}{args.suffix}.json"
        if f.exists():
            data[key] = filter_cats(load(f), keep)

    base = data[BASE]
    rows = []
    for key, dd in data.items():
        acc = micro_acc(dd)
        lo, hi = cluster_ci_abs(dd)
        if key == BASE:
            gap, p = 0.0, None
        else:
            gap, _, _, _ = cluster_ci_gap(base, dd)
            _, _, p = mcnemar_exact(base, dd)
        rows.append((key, META[key][0], META[key][1], acc, lo, hi, gap, p))

    rows.sort(key=lambda r: r[3], reverse=True)  # by accuracy desc

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = list(range(len(rows)))
    accs = [r[3] * 100 for r in rows]
    los = [(r[3] - r[4]) * 100 for r in rows]
    his = [(r[5] - r[3]) * 100 for r in rows]
    colors = [TIER_COLOR[r[2]] for r in rows]

    ax.bar(x, accs, yerr=[los, his], color=colors, capsize=4,
           edgecolor="black", linewidth=0.6, width=0.66, error_kw={"lw": 1})

    # verbatim-chunks reference line = the "cliff" edge
    base_acc = micro_acc(base) * 100
    ax.axhline(base_acc, color="#1b7837", ls="--", lw=1, alpha=0.7)
    ax.text(len(rows) - 0.4, base_acc + 0.6, "verbatim chunks",
            color="#1b7837", fontsize=7, ha="right")

    for xi, r in zip(x, rows):
        acc = r[3] * 100
        ax.text(xi, r[5] * 100 + 1.2, f"{acc:.1f}", ha="center", fontsize=8)
        # Only mark significance for bars BELOW verbatim (the extracted/summary
        # tiers); a star on the above-verbatim SeCom bar would read as a deficit.
        if r[7] is not None and r[2] != "verbatim":
            ax.text(xi, 2, stars(r[7]), ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], fontsize=8.5)
    ax.set_ylabel("LoCoMo QA accuracy (%)")
    ax.set_ylim(0, max(accs) + 8)
    ax.set_title("Verbatim-grounded memory beats every extracted representation\n"
                 "(fixed context budget; builder = GPT-4o steelman)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    handles = [Patch(facecolor=TIER_COLOR[t], edgecolor="black", label=TIER_LABEL[t])
               for t in ["verbatim", "extracted", "summary"]]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.fig)
    print(f"written: {args.fig}")
    print("order:", [(r[0], round(r[3] * 100, 1), stars(r[7]) if r[7] else "base") for r in rows])


if __name__ == "__main__":
    main()
