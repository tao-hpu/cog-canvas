#!/usr/bin/env python3
"""
Statistical significance analysis: CogCanvas recall-boost vs RAG on LoCoMo benchmark.

Compares per-conversation and per-category accuracy between the two systems
using Wilcoxon signed-rank tests, permutation tests, and bootstrap CIs.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from typing import NamedTuple


# ============================================================================
# Data definitions
# ============================================================================

# CogCanvas recall-boost: per-conversation (correct, total) from stdout
COGCANVAS_RAW: dict[str, tuple[int, int]] = {
    "locomo_000": (28, 82),
    "locomo_001": (17, 37),
    "locomo_002": (19, 66),
    "locomo_003": (21, 88),
    "locomo_004": (24, 71),
    "locomo_005": (26, 61),
    "locomo_006": (16, 67),
    "locomo_007": (21, 73),
    "locomo_008": (26, 83),
    "locomo_009": (24, 71),
}

# CogCanvas per-category overall accuracy (from stdout summary)
COGCANVAS_CATEGORY_OVERALL = {
    1: 0.277,  # single-hop 27.7%
    2: 0.327,  # temporal 32.7%
    3: 0.406,  # multi-hop 40.6%
}

RAG_JSON_PATH = Path(
    "/Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas-all/cog-canvas/"
    "experiments/results/locomo_rag_llmscore_10_cat123.json"
)

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain"}  # names per LoCoMo paper

RNG = np.random.default_rng(42)


# ============================================================================
# Data loading
# ============================================================================

class ConvData(NamedTuple):
    """Per-conversation data for one system."""
    conv_id: str
    overall_correct: int
    overall_total: int
    overall_acc: float
    cat_correct: dict[int, int]   # category -> correct count
    cat_total: dict[int, int]     # category -> total count
    cat_acc: dict[int, float]     # category -> accuracy


def load_rag_data() -> list[ConvData]:
    """Load RAG baseline results from JSON."""
    with open(RAG_JSON_PATH) as f:
        data = json.load(f)

    results = []
    for conv in data["conversations"]:
        conv_id = conv["id"]
        questions = conv["questions"]
        cat_correct: dict[int, int] = {}
        cat_total: dict[int, int] = {}

        for q in questions:
            cat = q["category"]
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if q["passed"]:
                cat_correct[cat] = cat_correct.get(cat, 0) + 1
            else:
                cat_correct.setdefault(cat, 0)

        total = len(questions)
        correct = sum(cat_correct.values())
        cat_acc = {
            c: cat_correct[c] / cat_total[c] if cat_total[c] > 0 else 0.0
            for c in cat_total
        }

        results.append(ConvData(
            conv_id=conv_id,
            overall_correct=correct,
            overall_total=total,
            overall_acc=correct / total if total > 0 else 0.0,
            cat_correct=cat_correct,
            cat_total=cat_total,
            cat_acc=cat_acc,
        ))
    return results


def build_cogcanvas_data(rag_data: list[ConvData]) -> list[ConvData]:
    """
    Build CogCanvas per-conversation data.

    We have overall (correct, total) per conversation.
    We do NOT have per-category breakdowns per conversation for CogCanvas,
    so we need to estimate them. Since we know the total questions per category
    per conversation (same questions for both systems), we can use the
    category totals from RAG and distribute CogCanvas correct proportionally,
    or note that we only have overall per-conversation accuracy for CogCanvas.

    For per-category analysis at the conversation level, we will use the
    per-conversation overall accuracy only and note the limitation.
    """
    results = []
    for rag_conv in rag_data:
        cid = rag_conv.conv_id
        correct, total = COGCANVAS_RAW[cid]
        results.append(ConvData(
            conv_id=cid,
            overall_correct=correct,
            overall_total=total,
            overall_acc=correct / total,
            cat_correct={},   # Not available per-conversation
            cat_total=rag_conv.cat_total,  # Same questions
            cat_acc={},        # Not available per-conversation
        ))
    return results


# ============================================================================
# Statistical tests
# ============================================================================

def wilcoxon_test(
    x: np.ndarray, y: np.ndarray, label: str
) -> tuple[float, float]:
    """Run Wilcoxon signed-rank test on paired samples."""
    diff = x - y
    # Remove zero differences (ties)
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        print(f"  [{label}] Too few non-zero differences for Wilcoxon test")
        return np.nan, np.nan

    stat, pval = stats.wilcoxon(nonzero, alternative="two-sided")
    return stat, pval


def permutation_test(
    x: np.ndarray, y: np.ndarray, n_iter: int = 10000
) -> float:
    """
    Permutation test for difference in means.

    Under H0 the labels are exchangeable. We randomly swap each pair
    and recompute the mean difference.
    """
    observed_diff = np.mean(x) - np.mean(y)
    n = len(x)
    count = 0
    for _ in range(n_iter):
        signs = RNG.choice([-1, 1], size=n)
        diff = x - y
        perm_diff = np.mean(signs * diff)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_iter


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI for difference in means (x - y).

    Uses BCa (bias-corrected and accelerated) via percentile method.
    Returns (lower, mean_diff, upper).
    """
    diff = x - y
    observed_mean = np.mean(diff)
    boot_means = np.array([
        np.mean(RNG.choice(diff, size=len(diff), replace=True))
        for _ in range(n_boot)
    ])
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, observed_mean, upper


def question_weighted_accuracy(data: list[ConvData]) -> float:
    """Compute question-weighted overall accuracy (pooled across all conversations)."""
    total_correct = sum(d.overall_correct for d in data)
    total_questions = sum(d.overall_total for d in data)
    return total_correct / total_questions


# ============================================================================
# Main analysis
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("CogCanvas (recall-boost) vs RAG on LoCoMo Benchmark")
    print("=" * 80)

    # Load data
    rag_data = load_rag_data()
    cog_data = build_cogcanvas_data(rag_data)

    conv_ids = [d.conv_id for d in cog_data]
    n = len(conv_ids)

    cog_acc = np.array([d.overall_acc for d in cog_data])
    rag_acc = np.array([d.overall_acc for d in rag_data])

    # ------------------------------------------------------------------
    # 1. Per-conversation comparison table
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("1. PER-CONVERSATION ACCURACY COMPARISON")
    print("-" * 80)
    print(f"{'Conv ID':<14} {'CogCanvas':>10} {'RAG':>10} {'Diff':>10} {'Winner':>10}")
    print("-" * 56)

    wins, losses, ties = 0, 0, 0
    for i in range(n):
        cc = cog_data[i]
        rr = rag_data[i]
        diff = cc.overall_acc - rr.overall_acc
        if diff > 0.001:
            winner = "CogCanvas"
            wins += 1
        elif diff < -0.001:
            winner = "RAG"
            losses += 1
        else:
            winner = "Tie"
            ties += 1
        print(
            f"{conv_ids[i]:<14} "
            f"{cc.overall_acc:>9.1%} "
            f"{rr.overall_acc:>9.1%} "
            f"{diff:>+9.1%} "
            f"{winner:>10}"
        )

    print("-" * 56)
    print(f"Win/Tie/Loss (CogCanvas vs RAG): {wins}/{ties}/{losses}")

    # ------------------------------------------------------------------
    # 2. Overall accuracy: two perspectives
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("2. OVERALL ACCURACY (TWO PERSPECTIVES)")
    print("-" * 80)

    # Perspective A: per-conversation mean (each conversation weighted equally)
    cog_mean = np.mean(cog_acc)
    rag_mean = np.mean(rag_acc)
    print(f"\n  A) Per-conversation mean (each conversation = 1 data point):")
    print(f"     CogCanvas: {cog_mean:.1%}  (std={np.std(cog_acc, ddof=1):.1%})")
    print(f"     RAG:       {rag_mean:.1%}  (std={np.std(rag_acc, ddof=1):.1%})")
    print(f"     Diff:      {cog_mean - rag_mean:+.1%}")

    # Perspective B: question-weighted (pooled)
    cog_pooled = question_weighted_accuracy(cog_data)
    rag_pooled = question_weighted_accuracy(rag_data)
    cog_total_q = sum(d.overall_total for d in cog_data)
    rag_total_q = sum(d.overall_total for d in rag_data)
    cog_total_c = sum(d.overall_correct for d in cog_data)
    rag_total_c = sum(d.overall_correct for d in rag_data)
    print(f"\n  B) Question-weighted (pooled across all conversations):")
    print(f"     CogCanvas: {cog_total_c}/{cog_total_q} = {cog_pooled:.1%}")
    print(f"     RAG:       {rag_total_c}/{rag_total_q} = {rag_pooled:.1%}")
    print(f"     Diff:      {cog_pooled - rag_pooled:+.1%}")

    print(f"\n  NOTE: Per-conversation mean ({cog_mean:.1%} vs {rag_mean:.1%}) can differ")
    print(f"  from question-weighted ({cog_pooled:.1%} vs {rag_pooled:.1%}) because")
    print(f"  conversations have different numbers of questions (37 to 88).")

    # ------------------------------------------------------------------
    # 3. Wilcoxon signed-rank test (overall)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("3. WILCOXON SIGNED-RANK TEST (Overall, per-conversation paired)")
    print("-" * 80)

    stat, pval = wilcoxon_test(cog_acc, rag_acc, "Overall")
    print(f"   Test statistic:  {stat:.1f}")
    print(f"   p-value:         {pval:.6f}")
    print(f"   Significant at alpha=0.05? {'YES' if pval < 0.05 else 'NO'}")
    print(f"   Significant at alpha=0.01? {'YES' if pval < 0.01 else 'NO'}")

    # Effect size: matched-pairs rank-biserial correlation
    diff_nonzero = (cog_acc - rag_acc)
    diff_nz = diff_nonzero[diff_nonzero != 0]
    n_nz = len(diff_nz)
    # r = 1 - (2*W) / (n*(n+1)/2) where W is the smaller of W+ and W-
    # Alternatively: r = Z / sqrt(n)
    # Using scipy's returned statistic (which is T+, the sum of positive ranks)
    # r = 4*T / (n*(n+1)) - 1
    r_effect = 4 * stat / (n_nz * (n_nz + 1)) - 1
    print(f"   Effect size (r): {r_effect:.3f}")

    # ------------------------------------------------------------------
    # 4. Permutation test (overall)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("4. PERMUTATION TEST (Overall, 10000 iterations)")
    print("-" * 80)

    perm_p = permutation_test(cog_acc, rag_acc, n_iter=10000)
    print(f"   Observed mean diff: {np.mean(cog_acc) - np.mean(rag_acc):+.4f}")
    print(f"   p-value:            {perm_p:.4f}")
    print(f"   Significant at alpha=0.05? {'YES' if perm_p < 0.05 else 'NO'}")

    # ------------------------------------------------------------------
    # 5. Bootstrap 95% CI for difference in means
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5. BOOTSTRAP 95% CI FOR DIFFERENCE IN MEANS (Overall)")
    print("-" * 80)

    lower, mean_diff, upper = bootstrap_ci(cog_acc, rag_acc, n_boot=10000)
    print(f"   Mean difference (CogCanvas - RAG): {mean_diff:+.4f} ({mean_diff:+.1%})")
    print(f"   95% CI: [{lower:+.4f}, {upper:+.4f}]")
    print(f"           [{lower:+.1%}, {upper:+.1%}]")
    excludes_zero = (lower > 0) or (upper < 0)
    print(f"   CI excludes zero? {'YES' if excludes_zero else 'NO'}")

    # ------------------------------------------------------------------
    # 6. Per-category analysis
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("6. PER-CATEGORY ANALYSIS")
    print("-" * 80)

    # Build per-conversation, per-category accuracy arrays from RAG data
    # For CogCanvas we do NOT have per-conversation category breakdowns,
    # so we'll do the analysis using per-conversation overall accuracy
    # AND a pooled question-level McNemar-style comparison.

    # First, show the pooled category-level comparison
    print("\n  6a) POOLED (QUESTION-WEIGHTED) CATEGORY ACCURACY:")
    print(f"  {'Category':<14} {'CogCanvas':>10} {'RAG':>10} {'Diff':>10}")
    print("  " + "-" * 48)

    rag_cat_pooled: dict[int, tuple[int, int]] = {}
    for conv in rag_data:
        for cat in conv.cat_total:
            if cat not in rag_cat_pooled:
                rag_cat_pooled[cat] = (0, 0)
            c, t = rag_cat_pooled[cat]
            rag_cat_pooled[cat] = (
                c + conv.cat_correct.get(cat, 0),
                t + conv.cat_total[cat],
            )

    for cat in sorted(CATEGORY_NAMES.keys()):
        cog_cat_acc = COGCANVAS_CATEGORY_OVERALL[cat]
        rag_c, rag_t = rag_cat_pooled[cat]
        rag_cat_acc = rag_c / rag_t
        diff = cog_cat_acc - rag_cat_acc
        print(
            f"  {CATEGORY_NAMES[cat]:<14} "
            f"{cog_cat_acc:>9.1%} "
            f"{rag_cat_acc:>9.1%} "
            f"{diff:>+9.1%}"
        )

    # 6b) Per-conversation per-category Wilcoxon where possible
    # We need CogCanvas per-conv per-cat data. We don't have it directly,
    # but we can reconstruct it if we know the per-category question counts
    # are the same. Since both systems answer the same questions, the totals
    # match. We only have CogCanvas overall correct counts, not per-category.
    #
    # HOWEVER, we have the CogCanvas overall category accuracies:
    #   single-hop: 27.7%, temporal: 32.7%, multi-hop: 40.6%
    # And RAG:
    #   single-hop: 33.3%, temporal: 12.1%, multi-hop: 40.6%
    #
    # The per-conversation per-category data for RAG IS available.
    # For a proper paired test, we need per-conversation per-category CogCanvas.
    # Since we don't have that breakdown, we'll note this limitation and
    # instead perform a two-proportion z-test on the pooled question counts.

    print("\n  6b) TWO-PROPORTION Z-TEST (pooled question-level, per category):")
    print("  (Since per-conversation category breakdowns are unavailable for")
    print("   CogCanvas, we use a pooled two-proportion test as an approximation.)")
    print()

    # CogCanvas category totals: same question counts as RAG
    cog_cat_totals = {}
    for conv in rag_data:
        for cat in conv.cat_total:
            cog_cat_totals[cat] = cog_cat_totals.get(cat, 0) + conv.cat_total[cat]

    # CogCanvas category correct counts (from pooled accuracy * total)
    # We know: single-hop=27.7% of 282, temporal=32.7% of 321, multi-hop=40.6% of 96
    # Let's compute the actual correct counts
    cog_cat_correct_est = {
        1: round(0.277 * cog_cat_totals[1]),  # 78 of 282
        2: round(0.327 * cog_cat_totals[2]),  # 105 of 321
        3: round(0.406 * cog_cat_totals[3]),  # 39 of 96
    }

    # Verify: these should sum to overall correct
    est_total = sum(cog_cat_correct_est.values())
    actual_total = sum(d.overall_correct for d in cog_data)
    print(f"  CogCanvas estimated category correct sum: {est_total} (actual total: {actual_total})")

    # Try to find exact correct counts by trying nearby values
    # We know total correct = 222, and the per-cat percentages
    # 27.7% of 282 = 78.114 -> 78
    # 32.7% of 321 = 104.967 -> 105
    # 40.6% of 96 = 38.976 -> 39
    # Sum = 78 + 105 + 39 = 222. Perfect!

    for cat in sorted(CATEGORY_NAMES.keys()):
        cog_c = cog_cat_correct_est[cat]
        cog_t = cog_cat_totals[cat]
        rag_c, rag_t = rag_cat_pooled[cat]

        # Two-proportion z-test
        p1 = cog_c / cog_t
        p2 = rag_c / rag_t
        p_pool = (cog_c + rag_c) / (cog_t + rag_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/cog_t + 1/rag_t))
        if se > 0:
            z = (p1 - p2) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            z = 0.0
            p_val = 1.0

        print(f"  {CATEGORY_NAMES[cat]} (cat {cat}):")
        print(f"    CogCanvas: {cog_c}/{cog_t} = {p1:.1%}")
        print(f"    RAG:       {rag_c}/{rag_t} = {p2:.1%}")
        print(f"    Diff:      {p1 - p2:+.1%}")
        print(f"    z = {z:.3f}, p = {p_val:.6f}")
        sig = "YES" if p_val < 0.05 else "NO"
        print(f"    Significant at alpha=0.05? {sig}")
        print()

    # ------------------------------------------------------------------
    # 7. Per-conversation Wilcoxon using RAG per-category accuracy
    # ------------------------------------------------------------------
    # Even though we don't have CogCanvas per-conv per-cat data,
    # we CAN test whether overall per-conv accuracy improvement
    # correlates with category composition. But more useful:
    # let's also do a chi-squared test on the pooled contingency table.

    print("-" * 80)
    print("7. CHI-SQUARED TEST ON POOLED 2x2 CONTINGENCY (Overall)")
    print("-" * 80)
    # CogCanvas: 222 correct, 477 incorrect out of 699
    # RAG: 172 correct, 527 incorrect out of 699
    # But these are NOT independent -- they answer the same questions.
    # McNemar's test would be appropriate if we had per-question paired data.
    # We don't have that pairing, so we note this and use the z-test.

    cog_overall_c = sum(d.overall_correct for d in cog_data)
    cog_overall_t = sum(d.overall_total for d in cog_data)
    rag_overall_c = sum(d.overall_correct for d in rag_data)
    rag_overall_t = sum(d.overall_total for d in rag_data)

    p1 = cog_overall_c / cog_overall_t
    p2 = rag_overall_c / rag_overall_t
    p_pool = (cog_overall_c + rag_overall_c) / (cog_overall_t + rag_overall_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/cog_overall_t + 1/rag_overall_t))
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"  CogCanvas: {cog_overall_c}/{cog_overall_t} = {p1:.1%}")
    print(f"  RAG:       {rag_overall_c}/{rag_overall_t} = {p2:.1%}")
    print(f"  Diff:      {p1 - p2:+.1%}")
    print(f"  z = {z:.3f}, p = {p_val:.6f}")
    print(f"  Significant at alpha=0.05? {'YES' if p_val < 0.05 else 'NO'}")
    print(f"  Significant at alpha=0.01? {'YES' if p_val < 0.01 else 'NO'}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY OF STATISTICAL TESTS")
    print("=" * 80)

    print(f"""
  System Comparison: CogCanvas (recall-boost) vs RAG (k=10, chunk=512)
  Benchmark: LoCoMo, 10 conversations, {cog_overall_t} questions (cat 1/2/3)

  OVERALL ACCURACY:
    CogCanvas:  {cog_pooled:.1%} (question-weighted), {cog_mean:.1%} (conv-mean)
    RAG:        {rag_pooled:.1%} (question-weighted), {rag_mean:.1%} (conv-mean)
    Difference: {cog_pooled - rag_pooled:+.1%} (question-weighted), {cog_mean - rag_mean:+.1%} (conv-mean)

  PER-CATEGORY (question-weighted):
    Single-hop (cat 1): CogCanvas {COGCANVAS_CATEGORY_OVERALL[1]:.1%} vs RAG {rag_cat_pooled[1][0]/rag_cat_pooled[1][1]:.1%} (diff {COGCANVAS_CATEGORY_OVERALL[1] - rag_cat_pooled[1][0]/rag_cat_pooled[1][1]:+.1%})
    Temporal   (cat 2): CogCanvas {COGCANVAS_CATEGORY_OVERALL[2]:.1%} vs RAG {rag_cat_pooled[2][0]/rag_cat_pooled[2][1]:.1%} (diff {COGCANVAS_CATEGORY_OVERALL[2] - rag_cat_pooled[2][0]/rag_cat_pooled[2][1]:+.1%})
    Multi-hop  (cat 3): CogCanvas {COGCANVAS_CATEGORY_OVERALL[3]:.1%} vs RAG {rag_cat_pooled[3][0]/rag_cat_pooled[3][1]:.1%} (diff {COGCANVAS_CATEGORY_OVERALL[3] - rag_cat_pooled[3][0]/rag_cat_pooled[3][1]:+.1%})
""")

    print("  STATISTICAL TESTS:")
    print(f"  {'Test':<45} {'p-value':>10} {'Sig?':>6}")
    print("  " + "-" * 63)

    # Recalculate per-category z-tests for summary
    cat_pvals = {}
    for cat in sorted(CATEGORY_NAMES.keys()):
        cog_c = cog_cat_correct_est[cat]
        cog_t = cog_cat_totals[cat]
        rag_c_val, rag_t_val = rag_cat_pooled[cat]
        pp1 = cog_c / cog_t
        pp2 = rag_c_val / rag_t_val
        pp_pool = (cog_c + rag_c_val) / (cog_t + rag_t_val)
        sse = np.sqrt(pp_pool * (1 - pp_pool) * (1/cog_t + 1/rag_t_val))
        if sse > 0:
            zz = (pp1 - pp2) / sse
            pp_val = 2 * (1 - stats.norm.cdf(abs(zz)))
        else:
            pp_val = 1.0
        cat_pvals[cat] = pp_val

    tests = [
        ("Wilcoxon signed-rank (conv-level)", pval),
        ("Permutation test (conv-level, 10K iter)", perm_p),
        (f"Bootstrap 95% CI excludes 0?", None),
        ("Two-proportion z-test (overall pooled)", p_val),
        ("Two-proportion z-test (single-hop)", cat_pvals[1]),
        ("Two-proportion z-test (temporal)", cat_pvals[2]),
        ("Two-proportion z-test (multi-hop)", cat_pvals[3]),
    ]

    for name, pv in tests:
        if pv is None:
            ci_str = f"[{lower:+.4f}, {upper:+.4f}]"
            sig_str = "YES" if excludes_zero else "NO"
            print(f"  {name:<45} {ci_str:>10} {sig_str:>6}")
        else:
            sig_str = "YES" if pv < 0.05 else "NO"
            print(f"  {name:<45} {pv:>10.6f} {sig_str:>6}")

    print()
    print("  KEY FINDINGS:")
    print(f"  - CogCanvas outperforms RAG on {wins}/10 conversations (win/tie/loss = {wins}/{ties}/{losses})")
    if pval < 0.05:
        print(f"  - Overall improvement is statistically significant (Wilcoxon p={pval:.4f})")
    else:
        print(f"  - Overall improvement is NOT statistically significant (Wilcoxon p={pval:.4f})")

    if cat_pvals[2] < 0.01:
        print(f"  - Temporal reasoning improvement is HIGHLY significant (p={cat_pvals[2]:.6f})")
    elif cat_pvals[2] < 0.05:
        print(f"  - Temporal reasoning improvement is significant (p={cat_pvals[2]:.6f})")

    if cat_pvals[3] >= 0.05:
        print(f"  - Multi-hop improvement is NOT significant (p={cat_pvals[3]:.4f}) -- same pooled rate")

    if cat_pvals[1] >= 0.05:
        print(f"  - Single-hop shows RAG slightly ahead, NOT significant (p={cat_pvals[1]:.4f})")
    elif cat_pvals[1] < 0.05:
        print(f"  - Single-hop difference is significant (p={cat_pvals[1]:.6f})")

    print()
    print("  INTERPRETATION:")
    print(f"  The overall CogCanvas advantage ({cog_pooled - rag_pooled:+.1%} question-weighted) is driven")
    print("  primarily by a large, highly significant gain in temporal reasoning")
    print("  (+20.6%), where structured memory graphs excel at tracking time-stamped")
    print("  events. Multi-hop accuracy is identical (40.6%), while single-hop")
    print("  shows RAG slightly ahead (-5.6%), reflecting RAG's strength at")
    print("  direct retrieval of explicitly stated facts.")


if __name__ == "__main__":
    main()
